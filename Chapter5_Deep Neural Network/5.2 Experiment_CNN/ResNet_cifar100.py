import os
import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


mode = 'train'
model_name = 'ResNet18'  # ResNet18, ResNet34, ResNet50
pretrained_model_path = './cifar100/{}_best.pth'.format(model_name)
if not os.path.exists(os.path.dirname(pretrained_model_path)):
    os.makedirs(os.path.dirname(pretrained_model_path))

# 定义超参数
batch_size = 128
learning_rate = 0.001
num_epochs = 50
validation_split = 0.1  # 验证集占比


# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4  # 对于瓶颈块，输出通道数是输入通道数的 4 倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * Bottleneck.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 实例化ResNet-18模型
def resnet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=100):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=100):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def train(model, train_loader, val_loader, device, pretrained_model_path: str):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_acc = 0
    acc = 0
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm.tqdm(train_loader, desc=f"[Train]: Epoch {epoch}/{num_epochs}", unit="batch")
        for i, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            running_loss += loss.item()
            train_bar.set_postfix(
                loss=f"{running_loss / len(train_loader):.3f}", acc_val=f"{acc:.3f}", acc_best=f"{best_acc:.3f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.6f}"
            )
        # 学习率调整
        lr_scheduler.step()
        # 模型验证
        acc = val(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            # 保存模型
            torch.save(model.state_dict(), pretrained_model_path)
    return


def val(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        val_bar = tqdm.tqdm(val_loader, desc=f"[Val]: ", unit="batch")
        for i, (images, labels) in enumerate(val_bar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    images_list = []
    labels_list = []
    predictions_list = []

    with torch.no_grad():
        test_bar = tqdm.tqdm(test_loader, desc=f"[Test]: ", unit="batch")
        for i, (images, labels) in enumerate(test_bar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 存储部分样本用于可视化
            if i % 10 == 0:
                images_list.extend(images.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                predictions_list.extend(predicted.cpu().numpy())
    acc = correct / total
    print(f'Accuracy of the model on the test set: {acc * 100:.2f}%')
    return acc, images_list, labels_list, predictions_list


def visualize_results(images, labels, predictions, num_images=10):
    plt.figure(figsize=(16, 6))
    num_images = num_images // 5 * 5
    for i in range(num_images):
        idx = random.randint(0, len(images))
        plt.subplot(num_images // 5, 5, i + 1)
        plt.imshow(images[idx].transpose(1, 2, 0) / 2 + 0.5)  # 反归一化
        plt.title(f'True: {classes[labels[idx]]}, Pred: {classes[predictions[idx]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# 数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# 下载数据集
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
classes = test_dataset.classes


# 计算验证集的大小
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(validation_split * num_train))
# 随机打乱索引
np.random.seed(42)
np.random.shuffle(indices)
# 划分训练集和验证集
train_indices, val_indices = indices[split:], indices[:split]
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)


# 数据加载
train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
if model_name == 'ResNet18':
    model = resnet18(num_classes=100)
elif model_name == 'ResNet34':
    model = resnet34(num_classes=100)
elif model_name == 'ResNet50':
    model = resnet50(num_classes=100)
else:
    raise ValueError('Unsupported model type! only ResNet18, ResNet34 and ResNet50 are supported!')

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if mode == 'test' and os.path.exists(pretrained_model_path):
    # 加载预训练模型
    model.load_state_dict(torch.load(pretrained_model_path))
    model = model.to(device)
    # 模型测试
    acc, images_list, labels_list, predictions_list = test(model, test_loader, device)
    # 调用可视化函数
    visualize_results(images_list, labels_list, predictions_list, 20)
else:
    model = model.to(device)
    # 模型训练
    train(model, train_loader, val_loader, device, pretrained_model_path)
    acc, images_list, labels_list, predictions_list = test(model, test_loader, device)
    # 调用可视化函数
    visualize_results(images_list, labels_list, predictions_list, 20)








import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 5
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 数据加载
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(128, 64)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, 10)  # 隐藏层2到输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入
        x = torch.relu(self.fc1(x))  # 第一层激活
        x = torch.relu(self.fc2(x))  # 第二层激活
        x = self.fc3(x)  # 输出层
        return x


# 实例化模型、损失函数和优化器
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm.tqdm(train_loader, desc=f"[Train]: Epoch {epoch}/{num_epochs}", unit="batch")
    for i, (images, labels) in enumerate(train_bar):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        running_loss += loss.item()
        train_bar.set_postfix(loss=running_loss/len(train_loader))
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 测试模型
model.eval()
correct = 0
total = 0
images_list = []
labels_list = []
predictions_list = []

with torch.no_grad():
    test_bar = tqdm.tqdm(test_loader, desc=f"[Test]: ", unit="batch")
    for i, (images, labels) in enumerate(test_bar):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 存储部分样本用于可视化
        images_list.extend(images.numpy())
        labels_list.extend(labels.numpy())
        predictions_list.extend(predicted.numpy())

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')


# 可视化部分测试结果
def visualize_results(images, labels, predictions, num_images=10):
    plt.figure(figsize=(12, 4))
    num_images = num_images // 5 * 5
    for i in range(num_images):
        idx = random.randint(0, len(images))
        plt.subplot(num_images // 5, 5, i + 1)
        plt.imshow(images[idx][0], cmap='gray')
        plt.title(f'True: {labels[idx]}, Pred: {predictions[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# 调用可视化函数
visualize_results(images_list, labels_list, predictions_list, 20)

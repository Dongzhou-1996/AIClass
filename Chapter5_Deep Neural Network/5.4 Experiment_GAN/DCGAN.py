import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from IPython.display import clear_output


# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_and_display_images(generator, fixed_noise, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    generated_images = generator(fixed_noise).permute(0, 2, 3, 1).detach().cpu()  # 生成图像
    generated_images = 0.5 * generated_images + 0.5  # 将像素值归一化到 [0, 1]

    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i])  # 显示生成的图像
        plt.axis('off')
    plt.tight_layout()
    plt.title(f'Epoch: {epoch}')
    # plt.show()
    path = 'run/results/{:06d}.png'.format(epoch)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 42
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    dataset_root = "data/celeba"
    num_epochs = 10
    lr = 2e-4
    batch_size = 128
    image_size = 64
    nc = 3  # Number of channels in the training images. For color images this is 3
    nz = 100  # Size of z latent vector (i.e. size of generator input)
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    beta1 = 0.5   # Beta1 hyperparameter for Adam optimizers

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    dataset = dset.ImageFolder(
        root=dataset_root,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Create the generator
    netG = Generator().to(device)
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # For each epoch
    for epoch in range(num_epochs):
        generate_and_display_images(netG, fixed_noise, epoch)
        # For each batch in the dataloader
        train_bar = tqdm.tqdm(dataloader, desc='[Train]: {}/{}'.format(epoch, num_epochs))
        for i, data in enumerate(train_bar):
            ############################
            # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through Discriminator
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for Discriminator in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with Generator
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with Discriminator
            output = netD(fake.detach()).view(-1)
            # Calculate Discriminator's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of Discriminator as sum over the fake and the real batches
            errD = errD_real + errD_fake
            optimizerD.step()  # Update 

            ############################
            # (2) Update Generator network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated Discriminator, perform another forward pass of all-fake batch through Discriminator
            output = netD(fake).view(-1)
            # Calculate Generator's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for Generator
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()  # Update Generator

            train_bar.set_postfix({
                'D_loss': errD.item(),
                'G_loss': errG.item(),
                'D(x)': D_x,
                'D(G(z1))': D_G_z1,
                'D(G(z2))': D_G_z2,
            })

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            # with torch.no_grad():
            #     fake = netG(fixed_noise).detach().cpu()
            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('run/losses.png', dpi=150)


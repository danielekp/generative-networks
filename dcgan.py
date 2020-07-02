skip_training = False

import os
import numpy as np
import matplotlib.pyplot as pl

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision.utils as utils


data_dir = "."

if torch.cuda.is_available() and not skip_training:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),  # Transform to tensor
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

batch_size = 100
data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

def save_model(model, filename):
    try:
        do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
        if do_save == 'yes':
            torch.save(model.state_dict(), filename)
            print('Model saved to %s.' % (filename))
        else:
            print('Model not saved.')
    except:
        raise Exception('error')

def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()

class Generator(nn.Module):
    def __init__(self, nz=10, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(nn.ConvTranspose2d(nz,4*ngf,kernel_size=4,stride=2,bias=False), nn.BatchNorm2d(4*ngf), nn.ReLU())
        self.block2 = nn.Sequential(nn.ConvTranspose2d(4*ngf,2*ngf,kernel_size=4,stride=2,bias=False,padding=1), nn.BatchNorm2d(2*ngf), nn.ReLU())
        self.block3 = nn.Sequential(nn.ConvTranspose2d(2*ngf,ngf,kernel_size=4,stride=2,bias=False,padding=2), nn.BatchNorm2d(ngf), nn.ReLU())
        self.block4 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1,bias=False))

    def forward(self, z, verbose=False):
        x = self.block1(z)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.tanh(x)

real_label = 1
fake_label = 0

def generator_loss(D, fake_images):
    dev = fake_images.device
    D = D.to(dev)
    batch_size = fake_images.size(0)
    loss = F.binary_cross_entropy(D(fake_images),torch.ones(batch_size,device=device)*real_label,reduction='mean')
    return loss

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator,self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,ndf,kernel_size=4,stride=2,bias=False,padding=1), nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(ndf,2*ndf,kernel_size=4,stride=2,bias=False,padding=2), nn.LeakyReLU(0.2))
        self.block3 = nn.Sequential(nn.Conv2d(2*ndf,4*ndf,kernel_size=4,stride=2,bias=False,padding=1), nn.LeakyReLU(0.2))
        self.block4 = nn.Sequential(nn.Conv2d(4*ndf,nc,kernel_size=4,stride=2,bias=False))

    def forward(self, x, verbose=False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.sigmoid(x).squeeze()


def discriminator_loss(D, real_images, fake_images):
    batch_size = real_images.size(0)
    out_real = D(real_images)
    d_loss_real = F.binary_cross_entropy(out_real,torch.ones(batch_size,device=device)*real_label)
    out_fake = D(fake_images)
    d_loss_fake = F.binary_cross_entropy(out_fake,torch.ones(batch_size,device=device)*fake_label)
    return d_loss_real, torch.mean(out_real), d_loss_fake, torch.mean(out_fake)

nz = 10
netG = Generator(nz=nz, ngf=64, nc=1)
netD = Discriminator(nc=1, ndf=64)

netD = netD.to(device)
netG = netG.to(device)

if not skip_training:
    lr = 0.0002
    epochs = 50
    optim1 = torch.optim.Adam(netG.parameters(),lr=lr,betas=(0.5, 0.999))
    optim2 = torch.optim.Adam(netD.parameters(),lr=lr,betas=(0.5, 0.999))
    for n in range(epochs):
        netG.train()
        netD.train()
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            fake_images = netG(torch.randn(real_images.size(0), nz, 1, 1, device=device))

            optim2.zero_grad()
            d_loss_real, d, d_loss_fake, f = discriminator_loss(netD, real_images, fake_images.detach())
            d_loss_real.backward()
            d_loss_fake.backward()
            optim2.step()

            optim1.zero_grad()
            g_loss = generator_loss(netD, fake_images)
            g_loss.backward()
            optim1.step()

        with torch.no_grad():
            netG.eval()
                  # Plot generated images
            z = torch.randn(144, nz, 1, 1, device=device)
            samples = netG(z)
            utils.save_image(samples, "generatedDCGAN/{}.png".format(n), nrow=10, normalize=True)
            print(g_loss)

if not skip_training:
    save_model(netG, 'dcgan_g.pth')
    save_model(netD, 'dcgan_d.pth')
else:
    nz = 10
    netG = Generator(nz=nz, ngf=64, nc=1)
    netD = Discriminator(nc=1, ndf=64)

    load_model(netG, 'dcgan_g.pth', device)
    load_model(netD, 'dcgan_d.pth', device)

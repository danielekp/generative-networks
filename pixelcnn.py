skip_training = False

import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils

data_dir = "."

if torch.cuda.is_available() and not skip_training:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_binary_receptive_field(net, image_size, i, j):
    inputs = torch.randn(32, 1, image_size[0], image_size[1], requires_grad=True)
    net.eval()
    net.to('cpu')
    outputs = net(inputs)
    loss = outputs[0,0,i,j]
    loss.backward()
    rfield = torch.abs(inputs.grad[0, 0]) > 0
    return rfield

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


trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, blind_center=False):
        super(MaskedConv2d,self).__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,bias=False,padding=p)
        self.register_buffer('mask',self.conv.weight.data.clone())
        self.typem = blind_center
        self.mask.fill_(1)
        self.mask[:, :, kernel_size // 2, kernel_size // 2 + (not self.typem):] = 0
        self.mask[:, :, kernel_size // 2 + 1:] = 0

    def forward(self, x):
        self.conv.weight.data *= self.mask
        x = self.conv(x)
        return x

layer = MaskedConv2d(in_channels=1, out_channels=2, kernel_size=5, blind_center=False)

net = nn.Sequential(
    MaskedConv2d(in_channels=1, out_channels=2, kernel_size=5, blind_center=True),
    MaskedConv2d(in_channels=2, out_channels=2, kernel_size=5, blind_center=False),
    MaskedConv2d(in_channels=2, out_channels=2, kernel_size=5, blind_center=False),
    MaskedConv2d(in_channels=2, out_channels=2, kernel_size=5, blind_center=False),
    MaskedConv2d(in_channels=2, out_channels=2, kernel_size=5, blind_center=False),
    MaskedConv2d(in_channels=2, out_channels=2, kernel_size=5, blind_center=False),
    MaskedConv2d(in_channels=2, out_channels=2, kernel_size=5, blind_center=False),
    MaskedConv2d(in_channels=2, out_channels=2, kernel_size=5, blind_center=False),
    nn.Conv2d(2, 256, 1)
)

class ConditionalPixelCNN(nn.Module):
    def __init__(self, n_channels=64, kernel_size=7):
        super(ConditionalPixelCNN, self).__init__()
        self.nc = n_channels
        self.first_block = nn.Sequential(MaskedConv2d(1,n_channels,kernel_size,blind_center=True), nn.BatchNorm2d(n_channels), nn.ReLU())
        self.second_block = nn.Sequential(MaskedConv2d(n_channels,n_channels,kernel_size), nn.BatchNorm2d(n_channels), nn.ReLU())
        self.third_block = nn.Sequential(MaskedConv2d(n_channels,n_channels,kernel_size), nn.BatchNorm2d(n_channels), nn.ReLU())
        self.fourth_block = nn.Sequential(MaskedConv2d(n_channels,n_channels,kernel_size), nn.BatchNorm2d(n_channels), nn.ReLU())
        self.fifth_block = nn.Sequential(MaskedConv2d(n_channels,n_channels,kernel_size), nn.BatchNorm2d(n_channels), nn.ReLU())
        self.sixth_block = nn.Sequential(MaskedConv2d(n_channels,n_channels,kernel_size), nn.BatchNorm2d(n_channels), nn.ReLU())
        self.seventh_block = nn.Sequential(MaskedConv2d(n_channels,n_channels,kernel_size), nn.BatchNorm2d(n_channels), nn.ReLU())
        self.eighth_block = nn.Sequential(MaskedConv2d(n_channels,n_channels,kernel_size), nn.BatchNorm2d(n_channels), nn.ReLU())
        self.out_block = nn.Conv2d(n_channels,256,1)
        self.M = nn.Linear(10,n_channels,bias=False)

    def forward(self, x, labels):
        one_hot = F.one_hot(labels,10).float().to(device)
        x = self.first_block(x)
        Wh = self.M(one_hot).reshape(x.size(0),self.nc,1,1)
        x = x + Wh
        x = self.second_block(x)
        x = x + Wh
        x = self.third_block(x)
        x = x + Wh
        x = self.fourth_block(x)
        x = x + Wh
        x = self.fifth_block(x)
        x = x + Wh
        x = self.sixth_block(x)
        x = x + Wh
        x = self.seventh_block(x)
        x = x + Wh
        x = self.eighth_block(x)
        return self.out_block(x)


def loss_fn(logits, x):
    return F.cross_entropy(logits,(x.reshape(x.size(0),x.size(2),-1) * 255).long())

def generate(net, labels, n_samples, image_size=(28, 28), device='cpu', save=False):
    net.eval()
    labels = labels.to(device)
    samples = torch.zeros(n_samples,1,image_size[0]*image_size[1]).to(device)
    for n in range(samples.size(2)):
        prob = F.softmax(net(samples.reshape(n_samples,1,image_size[0],image_size[1]),labels).reshape(n_samples,256,image_size[0]*image_size[1]), dim=1)
        samples[:,:,n] = torch.multinomial(prob[:,:,n],1).true_divide(255)
        if save:
            utils.save_image(samples.reshape(n_samples,1,image_size[0],image_size[1]),'generatedPixel/{}.png'.format(n),nrow=10, normalize=True)
    return samples.reshape(n_samples,1,image_size[0],image_size[1])

net = ConditionalPixelCNN(n_channels=64, kernel_size=7)
net.to(device)

if not skip_training:
    epochs = 80
    lr = 0.005
    optimi = optim.Adam(net.parameters(), lr)
    for n in range(epochs):
        net.train()
        loss_tot = 0
        for data, target in trainloader:
            data = data.to(device)
            target = target.to(device)
            optimi.zero_grad()
            t = net(data,target)
            loss = loss_fn(t,data)
            loss.backward()
            optimi.step()
            loss_tot += loss
        lo = str(loss_tot/len(trainloader)).replace('.','_')
        print(lo)
        with torch.no_grad():
            labels = []
            for n in range(10):
                labels.append(torch.ones(10)*n)
            labels = torch.cat(labels)
            labels = labels.to(torch.int64)
            gens = generate(net, labels, n_samples=100,device=device)
            utils.save_image(gens, "imagesConditionalPixelCNN/{}.png".format(lo), nrow=10, normalize=True)

if not skip_training:
    save_model(net, 'pixelcnn.pth')
else:
    net = ConditionalPixelCNN(n_channels=64, kernel_size=7)
    load_model(net, 'pixelcnn.pth', device)

with torch.no_grad():
    labels = []
    for n in range(10):
        labels.append(torch.ones(10)*n)
    labels = torch.cat(labels)
    labels = labels.to(torch.int64)
    gens = generate(net, labels, n_samples=100,device=device,save=True)

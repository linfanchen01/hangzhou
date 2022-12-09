"""""""""""
https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_MNIST_DCGAN.py
"""""""""""
import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.datasets as dset
import numpy as np
import pandas as pd


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))

        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)  # fixed noise
with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.to(device))


def show_result(num_epoch, show=False, save=False, path='result.png', isFix=False):
    z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        z_= Variable(z_.to(device))

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_size = 128
dlr = 0.0002
glr = 0.0001
train_epoch = 10000

# data_loader
# My dataset
# Counting how many data loaded
N_folder = 1
k = 0
for i in range(0, N_folder):
    for j in range(11010000, 12000000, 100):
        if os.path.exists("./data/compress/compress" + str(i + 1) + "/w3_" + str(j + 100) + ".csv"):
            k = k + 1
            N_filecase = k
        else:
            #print("The root does not exist!")
            pass
        continue
print("Totally " + str(N_filecase) + "data, start loading data......")

Mi = np.zeros(N_filecase)
Ma = np.zeros(N_filecase)
l = 0
k = 0
sub_sampling = 1
resolution = 32
if sub_sampling == 0:
    data_set = np.zeros([N_filecase, 512, 512, 1])
else:
    data_set = np.zeros([N_filecase, resolution, resolution, 1])

for i in range(0, N_folder):
    for j in range(11010000, 12000000, 100):
        if os.path.exists("./data/compress/compress" + str(i + 1) + "/w3_" + str(j + 100) + ".csv"):
            data_file = pd.read_csv("./data/compress/compress" + str(i + 1) + "/w3_" + str(j + 100) + ".csv")
            # data_inter=np.array(data_file.values.T[0])
            data_inter = np.array(data_file.values)
            Maxx = np.max(data_inter)
            Minn = np.min(data_inter)
            # data_inter=data_inter.reshape([512,512])
            data_inter = data_inter.reshape([resolution, resolution])
            if sub_sampling == 0:
                for m in range(0, 512):
                    for n in range(0, 512):
                        data_set[k][m][n][0] = data_inter[m][n]
            else:
                for m in range(0, resolution):
                    for n in range(0, resolution):
                        data_set[k][m][n][0] = data_inter[m][n]
            # can also replace 4 by 2
            Ma[k] = Maxx
            Mi[k] = Minn
            k = k + 1
        else:
            pass
        # print(k, "./data/compress/compress" + str(i + 1) + "/w3_" + str(j + 100) + ".csv")
        if k % 1000 == 0:
            print("loaded : {} in {} data".format(k, N_filecase))

Maxtotal = abs(max(Ma))
Mintotal = abs(min(Mi))
if Maxtotal > Mintotal:
    MMAX = Maxtotal
else:
    MMAX = Mintotal
MMAX = MMAX + 0.01
for l in range(0, N_filecase):
    data_set[l] = (data_set[l] + MMAX) / (2 * MMAX)
train_dataset = data_set.transpose(0, 3, 1, 2)
# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.to(device)
D.to(device)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=glr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=dlr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('../FLU32_DCGAN_results'):
    os.mkdir('../FLU32_DCGAN_results')
if not os.path.isdir('../FLU32_DCGAN_results/Random_results'):
    os.mkdir('../FLU32_DCGAN_results/Random_results')
if not os.path.isdir('../FLU32_DCGAN_results/Fixed_results'):
    os.mkdir('../FLU32_DCGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for _, x_ in enumerate(train_loader):
        # train discriminator D
        D.zero_grad()
        x_ = x_.type(torch.cuda.FloatTensor)

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        
        x_, y_real_, y_fake_ = Variable(x_.to(device)), Variable(y_real_.to(device)), Variable(y_fake_.to(device))
        D_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.to(device))
        G_result = G(z_)

        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # D_losses.append(D_train_loss.data[0])
        D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.to(device))

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.item())

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
    torch.mean(torch.FloatTensor(G_losses))))
    p = '../FLU32_DCGAN_results/Random_results/FLU32_DCGAN_' + str(epoch + 1) + '.png'
    fixed_p = '../FLU32_DCGAN_results/Fixed_results/FLU32_DCGAN_' + str(epoch + 1) + '.png'
    #######################
    #show results#
    if (epoch + 1)%1 == 0:
        show_result((epoch + 1), save=True, path=p, isFix=False)
        show_result((epoch + 1), save=True, path=fixed_p, isFix=True)
    #######################
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "../FLU32_DCGAN_results/generator_param.pkl")
torch.save(D.state_dict(), "../FLU32_DCGAN_results/discriminator_param.pkl")
with open('../FLU32_DCGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='../FLU32_DCGAN_results/FLU32_DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = '../FLU32_DCGAN_results/Fixed_results/FLU32_DCGAN_' + str(e + 1) + '.png'
    if os.path.exists(img_name):
        images.append(imageio.imread(img_name))
imageio.mimsave('../FLU32_DCGAN_results/generation_animation.gif', images, fps=5)

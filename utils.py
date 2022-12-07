import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import itertools
import imageio
import natsort
from glob import glob
import numpy as np
import pandas as pd
import os

def get_data_loader(batch_size):
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, ))])

    #train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    N_filecase=45623
    Mi=np.zeros(N_filecase)
    Ma=np.zeros(N_filecase)
    k=0
    l=0
    sub_sampling=1
    if sub_sampling==0:
        data_set=np.zeros([N_filecase,512,512,1])
    else:
        data_set=np.zeros([N_filecase,32,32,1])
    for i in range(0,5):
        for j in range(11010000,12000000,100):
          if os.path.exists("./data/compress/compress"+str(i+1)+"/w3_"+str(j+100)+".csv"):
            data_file=pd.read_csv("./data/compress/compress"+str(i+1)+"/w3_"+str(j+100)+".csv")
            #data_inter=np.array(data_file.values.T[0])
            data_inter=np.array(data_file.values)
            Maxx=np.max(data_inter)
            Minn=np.min(data_inter)
            #data_inter=data_inter.reshape([512,512])
            data_inter=data_inter.reshape([32,32])
            if sub_sampling==0:
                for m in range(0,512):
                    for n in range(0,512):
                        data_set[k][m][n][0]=data_inter[m][n]
            else:
                for m in range(0,32):
                    for n in range(0,32):
                        data_set[k][m][n][0]=data_inter[m][n]
            #can also replace 4 by 2
            Ma[k]=Maxx
            Mi[k]=Minn
            k=k+1
            print(k,"./data/compress/compress"+str(i+1)+"/w3_"+str(j+100)+".csv")
          else:
            print("The root does not exist!")
    Maxtotal=abs(max(Ma))
    Mintotal=abs(min(Mi))
    if Maxtotal>Mintotal:
        MMAX=Maxtotal
    else:
        MMAX = Mintotal
    MMAX = MMAX+0.01
    for l in range(0,N_filecase):
        data_set[l]=(data_set[l]+MMAX)/(2*MMAX)
    
    train_dataset = data_set.transpose(0,3,1,2)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def generate_images(epoch, path, fixed_noise, num_test_samples, netG, device, use_fixed=False):
    z = torch.randn(num_test_samples, 100, 1, 1, device=device)
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(z)
        path += 'variable_noise/'
        title = 'Variable Noise'
  
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k//4
        j = k%4
        ax[i,j].cla()
        ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28,28), cmap='Greys')
    label = 'Epoch_{}'.format(epoch+1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')

def save_gif(path, fps, fixed_noise=False):
    if fixed_noise==True:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path+'animated.gif', gif, fps=fps)

    

    

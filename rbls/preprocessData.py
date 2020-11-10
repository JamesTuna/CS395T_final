#! /usr/bin/env python3
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch,os,shutil
from torch.autograd import Variable
import argparse
import numpy as np

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='data',train = True,transform = transforms.ToTensor(),download = True)
test_dataset = dsets.MNIST(root ='data',train = False,transform = transforms.ToTensor())
# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = 1000,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = 1000,shuffle = False)

A = []
y = []
for img,label in train_loader:
    A.append(img.numpy().reshape(1000,784))
    y.append(label.numpy().reshape(1000,1))

A_test = []
y_test = []
for img,label in test_loader:
    A_test.append(img.numpy().reshape(1000,784))
    y_test.append(label.numpy().reshape(1000,1))

A = np.concatenate(A,axis=0)
y = np.concatenate(y,axis=0)
A_test = np.concatenate(A_test,axis=0)
y_test = np.concatenate(y_test,axis=0)

np.save('data/A.npy',A)
np.save('data/A_test.npy',A_test)
np.save('data/y.npy',y)
np.save('data/y_test.npy',y_test)

# one hot encoding
y_ = np.zeros((y.shape[0],10))
for i in range(y.shape[0]):
    y_[i,y[i]] = 1

y_test_ = np.zeros((y_test.shape[0],10))
for i in range(y_test.shape[0]):
    y_test_[i,y_test[i]] = 1

np.save('data/y_onehot.npy',y_)
np.save('data/y_test_onehot.npy',y_test_)

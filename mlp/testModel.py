#! /usr/bin/env python3
from trainer import *
from cMLP import *
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch,os,shutil
from torch.autograd import Variable
import argparse
# parser
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, help='model to test')
parser.add_argument('--layer', type=int, help='number of layers in MLP to be tested')
parser.add_argument('--hidden',type=int, default=32,help='how many hidden neurons in the model')
parser.add_argument('--batch-size',type=int, default=32,help='batch size')
parser.add_argument('--samples',type=int, default=100,help='number of samples to test')
parser.add_argument('--noise',type=float, default=0,help='noise threshold on weight matrices')
parser.add_argument('--logdir',type=str, default='./',help='where to save the test results')
args = parser.parse_args()

PERTURBATION = args.noise
SAVEDIR = args.load
batch_size = args.batch_size

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='../rbls/data',train = True,transform = transforms.ToTensor(),download = True)
test_dataset = dsets.MNIST(root ='../rbls/data',train = False,transform = transforms.ToTensor())
# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = False)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = cMLP(input_dim=28*28,output_dim=10,num_layers=args.layer,num_hidden_neurons=args.hidden)
# load saved model
st_dict = torch.load(args.load)
remove_para = []
for name in st_dict:
    if name.endswith('C_W') or name.endswith('C_b'):
        remove_para.append(name)
for name in remove_para:
    st_dict.__delitem__(name)
model.load_state_dict(st_dict)
model.to(device)
loss = nn.CrossEntropyLoss()
# optimizer
trainer = RobustTrainer(model,train_loader=train_loader,test_loader=test_loader,loss = loss,noise_scale=PERTURBATION)
# Test for random sampled noise
trainer.test(noise_scale=PERTURBATION,repeat=args.samples,use_cuda = torch.cuda.is_available(),logdir=args.logdir)

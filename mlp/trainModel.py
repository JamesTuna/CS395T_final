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
# model arch
parser.add_argument('--layer',type=int, default=5,help='how many layers in the model')
parser.add_argument('--hidden',type=int, default=32,help='how many hidden neurons in the model')
# model loading
parser.add_argument('--load', type=str, default=None, help='resume training from saved model')
# noise level & number
parser.add_argument('--noise',type=float, default=0,help='noise std')
parser.add_argument('--n', type=int, default=1, help='# of noises used each step')
# optimizer related
parser.add_argument('--opt',type=str, default = 'Adam', help='optimizer to do training')
parser.add_argument('--lr',type=float, default=0.001,help='learning rate')
parser.add_argument('--batch_size',type=int, default=32,help='batch size')
parser.add_argument('--epoch',type=int, default=120,help='epochs to train')
parser.add_argument('--lr_decay_epoch',type=int, default=100,help='how many epochs to reduce lr')
parser.add_argument('--lr_decay_rate',type=float, default=0.1,help='lr decay ratio')
# train log & model save
parser.add_argument('--ps',type=int, default=500,help='how many steps to print information in console')
parser.add_argument('--logdir',type=str, default='./',help='where to store results')
parser.add_argument('--save_as',type=str, default='./newModel',help='path to save the model')

args = parser.parse_args()

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='../rbls/data',train = True,transform = transforms.ToTensor(),download = True)
test_dataset = dsets.MNIST(root ='../rbls/data',train = False,transform = transforms.ToTensor())
# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = args.batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = args.batch_size,shuffle = False)
# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = cMLP(input_dim=28*28,output_dim=10,num_layers=args.layer,num_hidden_neurons=args.hidden)
# load saved model or not
if args.load is not None:
    model.load_state_dict(torch.load(args.load))
model.to(device)
loss = nn.CrossEntropyLoss()


# optimizer
trainer = RobustTrainer(model,train_loader=train_loader,test_loader=test_loader,loss = loss,
                noise_scale=args.noise,n_noises=args.n,train_epochs=args.epoch,lr=args.lr)

trainer.train_n_noises(device=device,print_step=args.ps,optimizer=args.opt,logdir=args.logdir,reduce_lr_per_epochs=args.lr_decay_epoch,reduce_rate=args.lr_decay_rate)
torch.save(model.state_dict(),args.save_as)

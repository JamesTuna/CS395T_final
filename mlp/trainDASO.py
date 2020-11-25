#! /usr/bin/env python3
from trainer import *
from cMLP import *
import torchvision
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
parser.add_argument('--save-per-epochs',type=int, default=100,help='how many epochs to store the model')
parser.add_argument('--cuda',type=int, default=None,help='cuda index if use cuda')

args = parser.parse_args()

######################################################################
# For the CIFAR10 dataset
###############################
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
######################################################################
# For the MNIST dataset
###############################
transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
######################################################################

# MNIST Dataset (Images and Labels)
train_dataset = torchvision.datasets.MNIST(root ='./rbls/data',train=True,transform=transforms, download=True)
test_dataset = torchvision.datasets.MNIST(root ='./rbls/data',train=False,transform=transforms, download=True)
# train_dataset = torchvision.datasets.CIFAR10(root='../rbls/data', train=True, download=True, transform=transform_train)
# test_dataset = torchvision.datasets.CIFAR10(root='../rbls/data', train=False,download=True, transform=transform_test)

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = args.batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=1000, shuffle = False)
# GPU
device = torch.device("cuda:%s" % args.cuda if args.cuda is not None else "cpu")
model = cMLP(input_dim=28*28,output_dim=10,num_layers=args.layer,num_hidden_neurons=args.hidden)
# load saved model or not
if args.load is not None:
    model.load_state_dict(torch.load(args.load))
    print(args.load + " loaded")
model.to(device)
loss = nn.CrossEntropyLoss()


# optimizer
trainer = RobustTrainer(model, train_loader=train_loader, test_loader=test_loader,loss = loss,
                noise_scale=args.noise, n_noises=args.n,train_epochs=args.epoch,lr=args.lr,
                        cuda=None if args.cuda is None else "cuda:%s" % args.cuda)

trainer.train_n_noises(device=device,print_step=args.ps,optimizer=args.opt,logdir=args.logdir,reduce_lr_per_epochs=args.lr_decay_epoch,reduce_rate=args.lr_decay_rate)

torch.save(model.state_dict(), args.save_as)

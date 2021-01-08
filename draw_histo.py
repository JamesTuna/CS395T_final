#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file1',type=str,help="the first acc npy file")
parser.add_argument('--file2',type=str,help="the second acc npy file")
parser.add_argument('--bins',default=20,type=int,help="number of bins")
parser.add_argument('--dasoN',type=int,help="DASO parameter used")
parser.add_argument('--title',type=str,default='',help='title of ploted graph')
parser.add_argument('--left',type=float,default=None,help='x axis left boundary')
parser.add_argument('--right',type=float,default=None,help='x axis right boundary')
args = parser.parse_args()

BINS=args.bins
accfile1 = '/home/jamestuna/Desktop/StochasticOptimization/CNN/MonteCarlo/noise0.2/n1_iter1000_lr0.001/acc_10000samples.npy'
accfile2 = '/home/jamestuna/Desktop/StochasticOptimization/CNN/MonteCarlo/noise0.2/n20_iter1000_lr0.001/acc.npy'

accfile1 = '/home/jamestuna/Desktop/StochasticOptimization/MonteCarlo/n_method/n1_iter3000/acc.npy'

accfile2 = '/home/jamestuna/Desktop/StochasticOptimization/MonteCarlo/n_method/n20_iter3000/acc.npy'

accfile1 = '/Users/zihaojames/Desktop/StochasticOptimization/CNN/MonteCarlo/noise0.2/n20_iter1000_lr0.001/acc.npy'
accfile2 = '/Users/zihaojames/Desktop/StochasticOptimization/CNN/MonteCarlo/noise0.2/n1_iter1000_lr0.001/acc_10000samples.npy'


acc1 = np.load(args.file1)
acc2 = np.load(args.file2)

print(acc1.shape)
print(acc2.shape)

bins=np.histogram(np.hstack((acc1,acc2)), bins=BINS)[1] #get the bin edges
plt.hist(acc1,bins,label='NI',alpha=0.5,density=False)
plt.hist(acc2,bins,label='DASO-n%s'%(args.dasoN),alpha=0.5,density=False)
#plt.title('ResNet-18 on CIFAR-10 with noise 0.2')
#plt.title('MLP-5-128 on MNIST with noise 2.0')
plt.ylabel('Count')
plt.xlabel('Accuracy')
plt.title(args.title)
plt.legend(loc='best')
if args.left is not None and args.right is not None:
    plt.xlim([args.left,args.right])
plt.show()

plt.hist(acc1,500*BINS,density=True,label='NI',histtype='step',cumulative=True)
plt.hist(acc2,500*BINS,density=True,label='DASO-n%s'%(args.dasoN),histtype='step',cumulative=True)
plt.ylabel('CDF')
plt.xlabel('Accuracy')
plt.legend(loc='best')
plt.title(args.title)
if args.left is not None and args.right is not None:
    plt.xlim([args.left,args.right])
plt.show()

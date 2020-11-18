#! /usr/bin/env python3
# pytorch based variability-aware training of a linear model
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from trainVortex import *

if __name__ == "__main__":
    import argparse
    from copy import deepcopy
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None, help='resume training from saved model')
    parser.add_argument('--noise',type=float, default=0,help='noise std in test')
    parser.add_argument('--logdir',type=str, default=0,help='where to save tested results')
    parser.add_argument('--samples',type=int, default=2000,help='how many samples of noises')
    args = parser.parse_args()

    A_test = np.load('../rbls/data/A_test.npy') # [10000,784]
    y_test = np.load('../rbls/data/y_test.npy') #[10000,1]

    # test
    W = np.load(args.load)
    noise_sample=args.samples
    accuracy = []
    print("test under noise %.4f"%(args.noise))
    for test in range(noise_sample):
        W_ = W * np.random.normal(1,args.noise,W.shape)
        y_hat = A_test.dot(W_)
        pred = y_hat.argmax(axis=1)
        acc = np.count_nonzero(pred == y_test.reshape(-1))/y_test.shape[0]
        if test % 100 == 99:
            print("test %s/%s acc %.4f"%(test+1,noise_sample,acc))
        accuracy.append(acc)
    accuracy = np.array(accuracy)
    print("avg acc %.4f"%(accuracy.mean()))

    # statistics
    acc_list = accuracy
    np.save(args.logdir+'/acc.npy',acc_list)
    with open(args.logdir+'/statistics.txt','w') as f:
        for q in range(1,20):
            qtl = 0.05 * q
            qtl_acc = np.quantile(acc_list,qtl)
            f.write('%.3f quantile:\tacc: %.4f\t\n'%(qtl,qtl_acc))
        f.write('Expectation:\tacc: %.4f\n'%(acc_list.mean()))
        f.write('Variance:\tacc: %.4f'%(acc_list.var()))
    # histogram
    #ax3 = plt.subplot(3,1,3)
    num_bins = 10
    acc_mean = acc_list.mean()
    heights,bins = np.histogram(acc_list,num_bins)
    freq = heights/sum(heights)
    width = (max(bins)-min(bins))/(len(bins))
    plt.bar(bins[:-1]+1/2*width,freq,width=width,color='blue',alpha=0.5)
    plt.xticks(np.around(bins,decimals=3))
    plt.title('Accuracy distribution of %s perturbations with mean %s'%(acc_list.shape[0],np.around(acc_mean,decimals=4)))
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.rc('grid', linestyle="-", color='black')
    plt.grid()
    plt.savefig(args.logdir+'/test_%s_perturbation_%s.pdf'%(args.samples,args.noise))

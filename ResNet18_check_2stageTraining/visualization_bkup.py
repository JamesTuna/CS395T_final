#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


####################################### accuracy ########################################
high_qtl = 0.95
low_qtl = 0.05
daso_n=50

noise_list = [0.1,0.2,0.3,0.4,0.5]

NI_low_qtl_list = []
NI_high_qtl_list = []
NI_mean_list = []

daso_low_qtl_list = []
daso_high_qtl_list = []
daso_mean_list = []

for i in range(len(noise_list)):
    noise = noise_list[i]
    ni_acc_file = np.load('test_results/noise%s_daso%s/acc_10000samples.npy'%(noise,1))
    daso_acc_file = np.load('test_results/noise%s_daso%s/acc_10000samples.npy'%(noise,daso_n))
    NI_low_qtl_list.append(np.quantile(ni_acc_file,low_qtl))
    NI_high_qtl_list.append(np.quantile(ni_acc_file,high_qtl))
    NI_mean_list.append(ni_acc_file.mean())

    daso_low_qtl_list.append(np.quantile(daso_acc_file,low_qtl))
    daso_high_qtl_list.append(np.quantile(daso_acc_file,high_qtl))
    daso_mean_list.append(daso_acc_file.mean())

# error bar plot
noise_list = np.array(noise_list)
NI_low_qtl_list = np.array(NI_low_qtl_list)
NI_high_qtl_list = np.array(NI_high_qtl_list)
NI_mean_list =np.array(NI_mean_list)
daso_low_qtl_list = np.array(daso_low_qtl_list)
daso_high_qtl_list = np.array(daso_high_qtl_list)
daso_mean_list = np.array(daso_mean_list)

ytop = NI_high_qtl_list - NI_mean_list
ybot = NI_mean_list - NI_low_qtl_list
plt.errorbar(noise_list,NI_mean_list,(ybot,ytop),fmt='-o',label="NI")

ytop = daso_high_qtl_list - daso_mean_list
ybot = daso_mean_list - daso_low_qtl_list
print(ytop)
print(daso_mean_list)
print(ybot)
plt.errorbar(noise_list+0.01,daso_mean_list,(ybot,ytop),fmt='-o',label="daso%s"%daso_n)

plt.xlabel("Noise Standard Deviation")
plt.ylabel("Test Accuracy")
plt.legend()
plt.xticks(noise_list)
plt.title("ResNet18 on CIFAR10")
plt.savefig("3.pdf")
plt.show()

f = plt.figure(figsize=(15,8))
# mean plot
ax1 = f.add_subplot(2,3,1)
ax1.plot(noise_list,NI_mean_list,label='noise injection',color='black')
ax1.scatter(noise_list,NI_mean_list,color='black')
ax1.plot(noise_list,daso_mean_list,label='daso%s'%daso_n,color='red')
ax1.scatter(noise_list,daso_mean_list,color='red')
ax1.set_title("mean accuracy")
ax1.set_ylabel("accuracy")
#ax1.set_xlabel("noise std")
ax1.legend()

# quantile plot
ax2 = f.add_subplot(2,3,2)
ax2.plot(noise_list,NI_high_qtl_list,label='noise injection',color='black')
ax2.scatter(noise_list,NI_high_qtl_list,color='black')
ax2.plot(noise_list,daso_high_qtl_list,label='daso%s'%daso_n,color='red')
ax2.scatter(noise_list,daso_high_qtl_list,color='red')
ax2.set_title("%s-quantile accuracy"%high_qtl)
ax2.set_ylabel("accuracy")
#ax2.set_xlabel("noise std")
ax2.legend()

# low quantile plot
ax3 = f.add_subplot(2,3,3)
ax3.plot(noise_list,NI_low_qtl_list,label='noise injection',color='black')
ax3.scatter(noise_list,NI_low_qtl_list,color='black')
ax3.plot(noise_list,daso_low_qtl_list,label='daso%s'%daso_n,color='red')
ax3.scatter(noise_list,daso_low_qtl_list,color='red')
ax3.set_title("%s-quantile accuracy"%low_qtl)
ax3.set_ylabel("accuracy")
#ax3.set_xlabel("noise std")
ax3.legend()


####################################### loss ########################################
NI_low_qtl_list = []
NI_high_qtl_list = []
NI_mean_list = []

daso_low_qtl_list = []
daso_high_qtl_list = []
daso_mean_list = []

for i in range(len(noise_list)):
    noise = noise_list[i]
    ni_acc_file = np.load('test_results/noise%s_daso%s/loss_10000samples.npy'%(noise,1))
    daso_acc_file = np.load('test_results/noise%s_daso%s/loss_10000samples.npy'%(noise,daso_n))
    NI_low_qtl_list.append(np.quantile(ni_acc_file,low_qtl))
    NI_high_qtl_list.append(np.quantile(ni_acc_file,high_qtl))
    NI_mean_list.append(ni_acc_file.mean())

    daso_low_qtl_list.append(np.quantile(daso_acc_file,low_qtl))
    daso_high_qtl_list.append(np.quantile(daso_acc_file,high_qtl))
    daso_mean_list.append(daso_acc_file.mean())

# mean plot
ax1 = f.add_subplot(2,3,4)
ax1.plot(noise_list,NI_mean_list,label='noise injection',color='black')
ax1.scatter(noise_list,NI_mean_list,color='black')
ax1.plot(noise_list,daso_mean_list,label='daso%s'%daso_n,color='red')
ax1.scatter(noise_list,daso_mean_list,color='red')
ax1.set_title("mean loss")
ax1.set_ylabel("loss")
ax1.set_xlabel("noise std")
ax1.legend()

# quantile plot
ax2 = f.add_subplot(2,3,5)
ax2.plot(noise_list,NI_high_qtl_list,label='noise injection',color='black')
ax2.scatter(noise_list,NI_high_qtl_list,color='black')
ax2.plot(noise_list,daso_high_qtl_list,label='daso%s'%daso_n,color='red')
ax2.scatter(noise_list,daso_high_qtl_list,color='red')
ax2.set_title("%s-quantile loss"%high_qtl)
ax2.set_ylabel("loss")
ax2.set_xlabel("noise std")
ax2.legend()

# low quantile plot
ax3 = f.add_subplot(2,3,6)
ax3.plot(noise_list,NI_low_qtl_list,label='noise injection',color='black')
ax3.scatter(noise_list,NI_low_qtl_list,color='black')
ax3.plot(noise_list,daso_low_qtl_list,label='daso%s'%daso_n,color='red')
ax3.scatter(noise_list,daso_low_qtl_list,color='red')
ax3.set_title("%s-quantile loss"%low_qtl)
ax3.set_ylabel("loss")
ax3.set_xlabel("noise std")
ax3.legend()
plt.savefig("1.pdf")
plt.show()


################################# histograms ######################################
BINS=25
f = plt.figure(figsize=(20,10))
for i in range(len(noise_list)):
    noise = noise_list[i]
    ni_acc_file = np.load('test_results/noise%s_daso%s/acc_10000samples.npy'%(noise,1))
    daso_acc_file = np.load('test_results/noise%s_daso%s/acc_10000samples.npy'%(noise,daso_n))

    ni_loss_file = np.load('test_results/noise%s_daso%s/loss_10000samples.npy'%(noise,1))
    daso_loss_file = np.load('test_results/noise%s_daso%s/loss_10000samples.npy'%(noise,daso_n))


    ax = f.add_subplot(5,4,i*4+1)
    # unified bin width
    bins=np.histogram(np.hstack((ni_acc_file,daso_acc_file)), bins=BINS)[1] #get the bin edges
    plt.hist(ni_acc_file,bins,label='NI',alpha=0.5,density=False)
    plt.hist(daso_acc_file,bins,label='DASO-n%s'%(daso_n),alpha=0.5,density=False)
    '''
    # un unified bin width
    ax.hist(ni_acc_file,bins=BINS,label='NI',alpha=0.5,density=False)
    ax.hist(daso_acc_file,bins=BINS,label='DASO-n%s'%(daso_n),alpha=0.5,density=False)
    '''

    ax.set_ylabel('Count')
    ax.set_xlabel('Accuracy')
    #ax.title(args.title)
    ax.legend(loc='best')
    #if args.left is not None and args.right is not None:
    #    plt.xlim([args.left,args.right])
    ax = f.add_subplot(5,4,i*4+2)
    ax.hist(ni_acc_file,500*BINS,density=False,label='NI',histtype='step',cumulative=True)
    ax.hist(daso_acc_file,500*BINS,density=False,label='DASO-n%s'%(daso_n),histtype='step',cumulative=True)
    ax.set_ylabel('CDF')
    ax.set_xlabel('Accuracy')
    ax.legend(loc='best')
    #ax.title(args.title)
    #if args.left is not None and args.right is not None:
    #    ax.xlim([args.left,args.right])

    ax = f.add_subplot(5,4,i*4+3)
    # unified bin width
    bins=np.histogram(np.hstack((ni_loss_file,daso_loss_file)), bins=BINS)[1] #get the bin edges
    plt.hist(ni_loss_file,bins,label='NI',alpha=0.5,density=False)
    plt.hist(daso_loss_file,bins,label='DASO-n%s'%(daso_n),alpha=0.5,density=False)
    '''
    # un-unified bin width
    ax.hist(ni_loss_file,bins=BINS,label='NI',alpha=0.5,density=False)
    ax.hist(daso_loss_file,bins=BINS,label='DASO-n%s'%(daso_n),alpha=0.5,density=False)
    '''
    ax.set_ylabel('Count')
    ax.set_xlabel('loss')
    #ax.title(args.title)
    ax.legend(loc='best')
    #if args.left is not None and args.right is not None:
    #    plt.xlim([args.left,args.right])

    ax = f.add_subplot(5,4,i*4+4)
    ax.hist(ni_loss_file,500*BINS,density=False,label='NI',histtype='step',cumulative=True)
    ax.hist(daso_loss_file,500*BINS,density=False,label='DASO-n%s'%(daso_n),histtype='step',cumulative=True)
    ax.set_ylabel('CDF')
    ax.set_xlabel('loss')
    ax.legend(loc='best')
    #ax.title(args.title)
    #if args.left is not None and args.right is not None:
    #    ax.xlim([args.left,args.right])
plt.savefig("2.pdf")
plt.show()

#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

high_qtl = 0.950
low_qtl = 0.050
hyper_vortex_list = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6]


noise_list = [0.2,0.4,0.6,0.8,1.0]

NI_low_qtl_list = []
NI_high_qtl_list = []
NI_mean_list = []

Vortex_low_qtl_list = []
Vortex_high_qtl_list = []
Vortex_mean_list = []

for i in range(len(noise_list)):
    noise = noise_list[i]
    ni_acc_file = np.load('NI%s/noise%s/acc.npy'%(noise,noise))

    best_performance_vortex = 0
    vortex_acc_file = None
    for hyper in hyper_vortex_list:
        vortex_mean_acc = np.load("w%s/noise%s/acc.npy"%(hyper,noise)).mean()
        if vortex_mean_acc > best_performance_vortex:
            vortex_acc_file = "w%s/noise%s/acc.npy"%(hyper,noise)
            best_performance_vortex = vortex_mean_acc

    print("noise %s, best hyper for vortex is %s"%(noise,vortex_acc_file.split('/')[0].split("w")[1]))
    vortex_acc_file = np.load(vortex_acc_file)
    NI_low_qtl_list.append(np.quantile(ni_acc_file,low_qtl))
    NI_high_qtl_list.append(np.quantile(ni_acc_file,high_qtl))
    NI_mean_list.append(ni_acc_file.mean())

    Vortex_low_qtl_list.append(np.quantile(vortex_acc_file,low_qtl))
    Vortex_high_qtl_list.append(np.quantile(vortex_acc_file,high_qtl))
    Vortex_mean_list.append(vortex_acc_file.mean())

# mean plot
plt.plot(noise_list,NI_mean_list,label='noise injection',color='black')
plt.scatter(noise_list,NI_mean_list,color='black')
plt.plot(noise_list,Vortex_mean_list,label='vortex',color='red')
plt.scatter(noise_list,Vortex_mean_list,color='red')
plt.title("mean accuracy")
plt.ylabel("accuracy")
plt.xlabel("noise std")
plt.legend()
plt.show()

# quantile plot
plt.plot(noise_list,NI_high_qtl_list,label='noise injection',color='black')
plt.scatter(noise_list,NI_high_qtl_list,color='black')
plt.plot(noise_list,Vortex_high_qtl_list,label='vortex',color='red')
plt.scatter(noise_list,Vortex_high_qtl_list,color='red')
plt.title("%s-quantile accuracy"%high_qtl)
plt.ylabel("accuracy")
plt.xlabel("noise std")
plt.legend()
plt.show()

# low quantile plot
plt.plot(noise_list,NI_low_qtl_list,label='noise injection',color='black')
plt.scatter(noise_list,NI_low_qtl_list,color='black')
plt.plot(noise_list,Vortex_low_qtl_list,label='vortex',color='red')
plt.scatter(noise_list,Vortex_low_qtl_list,color='red')
plt.title("%s-quantile accuracy"%low_qtl)
plt.ylabel("accuracy")
plt.xlabel("noise std")
plt.legend()
plt.show()

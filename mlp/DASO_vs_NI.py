#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def getStatistics(file,high_qtl=0.95,low_qtl=0.05):
    high_qtl_acc = None
    low_qtl_acc = None
    mean_acc = None
    with open(file,'r') as f:
        for l in f:
            if l.startswith('Expectation'):
                #print(l.split(" "))
                mean_acc = float(l.split(" ")[1].split('\t')[0])
                continue
            if l.startswith("Variance"):
                continue
            qtl = float(l.split(" ")[0])
            if np.abs(qtl - high_qtl) < 1e-6:
                #print(qtl)
                #print(l.split(" "))
                high_qtl_acc = float(l.split(" ")[2].split('\t')[0])
                #print(high_qtl_acc)
            elif np.abs(qtl - low_qtl) < 1e-6:
                low_qtl_acc = float(l.split(" ")[2].split('\t')[0])
    return low_qtl_acc,mean_acc,high_qtl_acc



if __name__ == "__main__":
    high_qtl = 0.90
    low_qtl = 0.10
    daso_n = 10
    noise_list = [0.2,0.4,0.8,1.0]

    NI_low_qtl_list = []
    NI_high_qtl_list = []
    NI_mean_list = []

    DASO_low_qtl_list = []
    DASO_high_qtl_list = []
    DASO_mean_list = []

    for i in range(len(noise_list)):
        noise = noise_list[i]
        ni_stat_file = "logs/l2h32noise%sn1_lr0.01ep400decay80rate0.1/noise%s/statistics.txt"%(noise,noise)
        daso_stat_file = "logs/l2h32noise%sn%s_lr0.01ep400decay80rate0.1/noise%s/statistics.txt"%(noise,daso_n,noise)

        low,mean,high = getStatistics(ni_stat_file,high_qtl=high_qtl,low_qtl=low_qtl)
        NI_low_qtl_list.append(low)
        NI_high_qtl_list.append(high)
        NI_mean_list.append(mean)

        low,mean,high = getStatistics(daso_stat_file,high_qtl=high_qtl,low_qtl=low_qtl)
        DASO_low_qtl_list.append(low)
        DASO_high_qtl_list.append(high)
        DASO_mean_list.append(mean)

    # mean plot
    plt.plot(noise_list,NI_mean_list,label='noise injection',color='black')
    plt.scatter(noise_list,NI_mean_list,color='black')
    plt.plot(noise_list,DASO_mean_list,label='daso',color='red')
    plt.scatter(noise_list,DASO_mean_list,color='red')
    plt.title("mean accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("noise std")
    plt.legend()
    plt.show()

    # quantile plot
    plt.plot(noise_list,NI_high_qtl_list,label='noise injection',color='black')
    plt.scatter(noise_list,NI_high_qtl_list,color='black')
    plt.plot(noise_list,DASO_high_qtl_list,label='daso',color='red')
    plt.scatter(noise_list,DASO_high_qtl_list,color='red')
    plt.title("%s-quantile accuracy"%high_qtl)
    plt.ylabel("accuracy")
    plt.xlabel("noise std")
    plt.legend()
    plt.show()

    # low quantile plot
    plt.plot(noise_list,NI_low_qtl_list,label='noise injection',color='black')
    plt.scatter(noise_list,NI_low_qtl_list,color='black')
    plt.plot(noise_list,DASO_low_qtl_list,label='daso',color='red')
    plt.scatter(noise_list,DASO_low_qtl_list,color='red')
    plt.title("%s-quantile accuracy"%low_qtl)
    plt.ylabel("accuracy")
    plt.xlabel("noise std")
    plt.legend()
    plt.show()

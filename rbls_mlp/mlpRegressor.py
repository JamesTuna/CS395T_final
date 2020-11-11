#! /usr/bin/env python3
# Given (1) a trained MLP (2) a dataset (3) std of weight variation
# use RBLS to get a set of robustified parameters
# that fight against the specific multiplicative Gaussian noise on MLP parameters
from numpyMLP import *
from copy import deepcopy

def regressMLP(model,X,noise):
    assert type(model) == npMLP, 'input model must be converted to npMLP first'
    newModel = deepcopy(model)
    n_layers = len(model.Ws)
    for i in range(n_layers):
        layer = i + 1
        print("Regress layer %s"%i)
        # construct mean input matrix A and estimate its variance
        meanA, varA = newModel.estimate_input_nlayer(X,layer,noise=noise,samples=100) # meanA, varA, and X are of the same shape [#samples, in_dim]
        varA = varA.mean()/(meanA**2).mean()
        print("estimated variance of A",varA)
        #varA = varA.mean()
        # construct Y matrix
        Y = model.forward_nlayer(X,layer,noise=0) # Y is of shape [#samples, out_dim]
        # robust least square to find new weights
        meanA = np.concatenate([meanA,np.ones((meanA.shape[0],1))],axis=1) # add one column to A for bias term

        sol = rbSolve(meanA,Y,varA,varx=noise**2)
        newW = sol[:-1,:]
        newb = sol[-1,:]
        newModel.Ws[i] = newW
        newModel.bs[i] = newb

    return newModel


if __name__ == "__main__":

    A_train = np.load('../rbls/data/A.npy') # [60000,784]
    A_test = np.load('../rbls/data/A_test.npy') # [10000,784]
    y_train_onehot = np.load('../rbls/data/y_onehot.npy') # [60000,10]
    y_test_onehot = np.load('../rbls/data/y_test_onehot.npy') # [10000,10]
    y_test = np.load('../rbls/data/y_test.npy') #[10000,1]

    noise = 0.5

    model = npMLP("l2h32noise0n1_lr0.1ep40decay10rate0.1.ckpt")
    layer = 2

    newModel = regressMLP(model,A_train,noise)

    # performance of original model under noise
    noise_sample=2000
    accuracy = []
    print("test orginal model, under noise %.4f"%(noise))
    for test in range(noise_sample):
        y_hat = model.forward_nlayer(A_test,layer,noise=noise)
        pred = y_hat.argmax(axis=1)
        acc = np.count_nonzero(pred == y_test.reshape(-1))/y_test.shape[0]
        if test % 100 == 99:
            print("test %s/%s acc %.4f"%(test+1,noise_sample,acc))
        accuracy.append(acc)
    accuracy = np.array(accuracy)
    print("avg acc %.4f"%(accuracy.mean()))

    # performance of regressed model under noise
    accuracy = []
    print("test regressed model, under noise %.4f"%(noise))
    for test in range(noise_sample):
        y_hat = newModel.forward_nlayer(A_test,layer,noise=noise)
        pred = y_hat.argmax(axis=1)
        acc = np.count_nonzero(pred == y_test.reshape(-1))/y_test.shape[0]
        if test % 100 == 99:
            print("test %s/%s acc %.4f"%(test+1,noise_sample,acc))
        accuracy.append(acc)
    accuracy = np.array(accuracy)
    print("avg acc %.4f"%(accuracy.mean()))

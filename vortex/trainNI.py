#! /usr/bin/env python3
# baseline noise injection training
# same loss function as vortex
import torch.nn as nn
import torch
import numpy as np
import time

class NILinear(nn.Module):
    def __init__(self,in_dim,out_dim,init_std=0.1,noise=0,cuda=False):
        super(NILinear, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_dim,out_dim).normal_(0,init_std))
        self.cuda = cuda
        if cuda:
            self.to('cuda:0')
        self.noise = noise
        self.inDim = in_dim
        self.outDim = out_dim

    def forward(self,X):
        if self.cuda:
            if self.noise == 0:
                CW = torch.cuda.FloatTensor(self.inDim,self.outDim).zero_()
            else:
                CW = torch.cuda.FloatTensor(self.inDim,self.outDim).normal_(0,self.noise)
        else:
            if self.noise == 0:
                CW = torch.FloatTensor(self.inDim,self.outDim).zero_()
            else:
                CW = torch.FloatTensor(self.inDim,self.outDim).normal_(0,self.noise)

        CW += 1
        PW = torch.mul(CW,self.W)
        out = torch.matmul(X, PW)

        return out

    def loss(self,X,Y):

        out = self.forward(X) # [batch, 10]
        convention_term = torch.mul(Y,out) # shape [10]
        epsilon =  - convention_term + 1 # shape [10]
        epsilon = torch.clamp(epsilon,min=0) # shape [10]
        loss = torch.sum(epsilon) # shape [1]
        loss /= X.size()[0]
        return loss

    def fit(self,X,Y,batch=128,epoch=100,ps=100,optimizer=None):
        # X,Y: numpy arrays
        # X: [60000,784]
        # Y: [60000,10], labels are from {+1,-1}
        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        n_samples = X.shape[0]
        decay_rate = 0.1
        dacay_epoch = 40

        for ep in range(epoch):

            start = time.time()

            epoch_loss = 0

            if ep % dacay_epoch == dacay_epoch - 1:
                for g in optimizer.param_groups:
                    g['lr'] *= decay_rate

            print("epoch %s/%s"%(ep+1,epoch))

            shuffle = np.random.choice(n_samples,n_samples,replace=False)
            X = X[shuffle] # shuffle
            Y = Y[shuffle]
            n_steps = (n_samples//batch + 1) if (n_samples%batch > 0) else n_samples//batch

            for iter in range(n_steps):
                batch_X = X[iter * batch:min(iter * batch + batch, n_samples),:]
                batch_Y = Y[iter * batch:min(iter * batch + batch, n_samples),:]
                batch_X = torch.Tensor(batch_X)
                batch_Y = torch.Tensor(batch_Y)

                if self.cuda:
                    batch_X = batch_X.to('cuda:0')
                    batch_Y = batch_Y.to('cuda:0')


                optimizer.zero_grad()
                loss = self.loss(batch_X,batch_Y)
                loss.backward()
                optimizer.step()

                batch_loss = loss.data.item()
                epoch_loss += batch_loss

                if iter % ps == ps - 1:
                    print("step %s, batch loss %.4f"%(iter+1,batch_loss))
            end = time.time()
            print("[%.2f seconds]epoch averaged loss %.4f"%(end-start,epoch_loss/n_steps))


if __name__ == "__main__":
    import argparse
    from copy import deepcopy
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise',type=float, default=0,help='noise injected in training')
    parser.add_argument('--saveas', type=str, help='full path to save the trained model')
    parser.add_argument('--cuda', action="store_true", default=False)
    args = parser.parse_args()

    A_train = np.load('../rbls/data/A.npy') # [60000,784]
    y_train_onehot = np.load('../rbls/data/y_onehot.npy') # [60000,10]
    y_train_onehot[y_train_onehot==0]=-1


    model = NILinear(in_dim=784,out_dim=10,noise=args.noise,init_std=0.1,cuda=args.cuda)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # train
    print('NI training with noise %.4f'%(args.noise))
    model.fit(A_train,y_train_onehot,batch=128,epoch=200,ps=1000,optimizer=optimizer)

    # save
    W = model.W.cpu().detach().numpy()
    np.save(args.saveas,W)

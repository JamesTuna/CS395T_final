#! /usr/bin/env python3
# pytorch based variability-aware training of a linear model
import torch.nn as nn
import torch
import numpy as np
import time

class VortexLinear(nn.Module):
    def __init__(self,in_dim,out_dim,init_std=0.1,cuda=False):
        super(VortexLinear, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_dim,out_dim).normal_(0,init_std))
        self.cuda = cuda
        self.to('cuda:0')

    def forward(self,X):
        return torch.matmul(X,self.W)

    def VortexLoss(self,X,Y,ro,gamma,alpha0=1,alpha1=1):

        # equation 10 in paper
        # https://users.ece.cmu.edu/~xinli/papers/2015_DAC_vortex.pdf
        # dimension comments are based on MNIST

        out = self.forward(X) # [batch, 10]
        loss = 0
        batch_size = X.size()[0]
        # epsilon for ith sample in the batch
        # epsilon should be dimension-10
        for i in range(batch_size):
            y = Y[i] # shape [10]
            #V = torch.diag(X[i]) # shape [784,784]
            #V = torch.matmul(V,self.W) # shape [784,10]
            V = torch.transpose(X[i] * torch.transpose(self.W, 0, 1), 0, 1) # shape [784,10], faster than above
            t = torch.norm(V,dim=0) # shape [10]
            penalization_term = torch.abs(alpha1 * y) # shape [10]
            penalization_term = torch.mul(penalization_term,t) # shape [10]
            penalization_term *= ro * gamma # shape [10]
            convention_term = alpha0 * torch.mul(y,out[i]) # shape [10]
            epsilon = penalization_term - convention_term + 1 # shape [10]
            epsilon = torch.clamp(epsilon,min=0) # shape [10]
            loss += torch.sum(epsilon) # shape [1]
        loss /= batch_size
        return loss

    def fit(self,X,Y,ro,gamma,batch=128,epoch=100,ps=100,optimizer=None):
        # X,Y: numpy arrays
        # X: [60000,784]
        # Y: [60000,10], labels are from {+1,-1}
        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        n_samples = X.shape[0]
        decay_rate = 0.1
        dacay_epoch = 20

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
                loss = self.VortexLoss(batch_X,batch_Y,ro,gamma)
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
    parser.add_argument('--gamma',type=float, default=0,help='gamma')
    parser.add_argument('--ro',type=float, default=0,help='ro')
    parser.add_argument('--saveas', type=str, help='full path to save the trained model')
    parser.add_argument('--cuda', action="store_true", default=False)
    args = parser.parse_args()

    A_train = np.load('../rbls/data/A.npy') # [60000,784]
    y_train_onehot = np.load('../rbls/data/y_onehot.npy') # [60000,10]
    y_train_onehot[y_train_onehot==0]=-1


    model = VortexLinear(in_dim=784,out_dim=10,init_std=0.1,cuda=args.cuda)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # train
    print('Vortex training with ro %.4f gamma %.4f'%(args.ro,args.gamma))
    model.fit(A_train,y_train_onehot,ro=args.ro,gamma=args.gamma,batch=128,epoch=100,ps=1000,optimizer=optimizer)

    # save
    W = model.W.cpu().detach().numpy()
    np.save(args.saveas,W)

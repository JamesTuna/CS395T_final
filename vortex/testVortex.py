#! /usr/bin/env python3
# pytorch based variability-aware training of a linear model
import torch.nn as nn
import torch
import numpy as np

class VortexLinear(nn.Module):
    def __init__(self,in_dim,out_dim,init_std=0.1):
        super(VortexLinear, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_dim,out_dim).normal_(0,init_std))

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

                optimizer.zero_grad()
                loss = self.VortexLoss(batch_X,batch_Y,ro,gamma)
                loss.backward()
                optimizer.step()

                batch_loss = loss.data.item()
                epoch_loss += batch_loss

                if iter % ps == ps - 1:
                    print("step %s, batch loss %.4f"%(iter+1,batch_loss))
            print("epoch averaged loss %.4f"%(epoch_loss/n_steps))


if __name__ == "__main__":
    import argparse
    from copy import deepcopy
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None, help='resume training from saved model')
    parser.add_argument('--noise',type=float, default=0,help='noise std in test')
    args = parser.parse_args()

    A_test = np.load('../rbls/data/A_test.npy') # [10000,784]
    y_test = np.load('../rbls/data/y_test.npy') #[10000,1]

    # test
    W = np.load(args.load)
    noise_sample=2000
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

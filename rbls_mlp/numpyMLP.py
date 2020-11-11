# convert cMLP model saved under ../mlp folder to numpy Matrices
# for later rbSolver to retreive A y
#! /usr/bin/env python3

import sys
sys.path.insert(1,'../rbls')
sys.path.insert(1,'../mlp')

from cMLP import *
from trainer import *
from rbls import *
from numpy.random import normal

class npMLP():
    def __init__(self,cMLP_model_path=None):
        self.Ws = []
        self.bs = []
        if cMLP_model_path is None:
            return
        st_dict = torch.load(cMLP_model_path)
        print("loading saved model: %s"%cMLP_model_path)
        for name in st_dict:
            print(name)
            if name.endswith(".W"):
                self.Ws.append(st_dict[name].cpu().numpy().T)
            elif name.endswith(".b"):
                self.bs.append(st_dict[name].cpu().numpy().T)
        print("model converted to numpy array")

    def act(self,x):
        # activation function used, default ReLU()
        x[x<0] = 0
        return x

    def forward_nlayer(self,x,n,noise=0):
        # x shape: [n samples, 784]
        # return the outputs of n th layer(no activation of nth layer)
        assert n>=0 and n<=len(self.Ws), 'layer index out of range, should be in range [0,%s]'%(len(self.Ws))
        if n>0:
            for i in range(0,n-1):
                x = x.dot(self.Ws[i]*normal(1,noise,self.Ws[i].shape)) + self.bs[i]*normal(1,noise,self.bs[i].shape)
                x = self.act(x)
            x = x.dot(self.Ws[n-1]*normal(1,noise,self.Ws[n-1].shape)) + self.bs[n-1]*normal(1,noise,self.bs[n-1].shape)
        return x


    def estimate_input_nlayer(self,x,n,noise,samples=100):
        # estimate mean and var of A matrix in rbls, shape [#samples, #feature]
        assert n>=1 and n<=len(self.Ws), 'layer index out of range, should be in range [1,%s]'%(len(self.Ws))

        sample_inputs = []


        for i in range(samples):
            noisy_out = self.forward_nlayer(x,n-1,noise=noise)
            noisy_out = self.act(noisy_out) # [n samples, out_dim]
            sample_inputs.append(noisy_out)

        sample_inputs = np.array(sample_inputs) # [100, #samples, #features]

        mean = sample_inputs.mean(axis=0)
        var = sample_inputs.var(axis=0)

        return mean, var














if __name__ == "__main__":
    '''
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
    args = parser.parse_args()
    '''

    A_train = np.load('../rbls/data/A.npy') # [60000,784]
    A_test = np.load('../rbls/data/A_test.npy') # [10000,784]
    y_train_onehot = np.load('../rbls/data/y_onehot.npy') # [60000,10]
    y_test_onehot = np.load('../rbls/data/y_test_onehot.npy') # [10000,10]
    y_test = np.load('../rbls/data/y_test.npy') #[10000,1]

    # debug: check correctness of function forward_nlayer() and estimate_input()
    np_model = npMLP("l2h32noise0n1_lr0.1ep40decay10rate0.1.ckpt")
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_model = cMLP(input_dim=28*28,output_dim=10,num_layers=2,num_hidden_neurons=32).to(device)
    torch_model.load_state_dict(torch.load("l2h32noise0n1_lr0.1ep40decay10rate0.1.ckpt"))
    # random sample a batch of inputs
    x = A_train[:5,:]
    np_out = np_model.forward_nlayer(x,1,noise=0)
    torch_out = torch_model(torch.from_numpy(x).to(device))
    print("np")
    print(np_out)
    print("true")
    print(torch_out)
    # estimate mean input variance of nth layer
    print("estimate input mean and variance for layer 2")
    mean,var = np_model.estimate_input_nlayer(x,n=2,noise=0.1,samples=1000)
    print('mean')
    print(mean)
    print('var')
    print(var)
    # estimate_input_nlayer() time on whole MNIST dataset,sample=1000
    start = time.time()
    mean,var = np_model.estimate_input_nlayer(A_train,n=2,noise=0.5,samples=100)
    end = time.time()
    print("%.4f seconds elasped "%(end-start))
    # check if it's reasonable to estimate var using a small fraction of x
    print("var on whole train: %.4f",var.mean())
    for i in range(10):
        choosen_samples = np.random.choice(A_train.shape[0],1000)
        var_ = var[choosen_samples]
        print("var on randomly choosen 1000 samples: %.4f"%var_.mean())

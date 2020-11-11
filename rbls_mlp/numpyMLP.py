# convert cMLP model saved under ../mlp folder to numpy Matrices
# for later rbSolver to retreive A y
#! /usr/bin/env python3

import sys
sys.path.insert(1,'../rbls')
sys.path.insert(1,'../mlp')

from cMLP import *
from trainer import *
from rbls import *

class npMLP():
    def __init__(self,cMLP_model_path):
        st_dict = torch.load(cMLP_model_path)
        self.Ws = []
        self.bs = []
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

    def forward_n(self,x,n):
        # x shape: [n samples, 784]
        # return the outputs of n th layer(no activation of nth layer)
        assert n>=0 and n<=len(self.Ws), 'layer index out of range'
        if n>0:
            for i in range(0,n-1):
                x = x.dot(self.Ws[i]) + self.bs[i]
                x = self.act(x)
            x = x.dot(self.Ws[n-1]) + self.bs[n-1]
        return x







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

    # debug: check correctness of npMLP.forward_n
    np_model = npMLP("l2h32noise0n1_lr0.1ep40decay10rate0.1.ckpt")
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_model = cMLP(input_dim=28*28,output_dim=10,num_layers=2,num_hidden_neurons=32).to(device)
    torch_model.load_state_dict(torch.load("l2h32noise0n1_lr0.1ep40decay10rate0.1.ckpt"))
    # random sample a batch of inputs
    x = A_train[:5,:]
    np_out = np_model.forward_n(x,2)
    torch_out = torch_model(torch.from_numpy(x).to(device))
    print("np")
    print(np_out)
    print("true")
    print(torch_out)

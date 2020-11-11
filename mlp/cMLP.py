#! /usr/bin/env python3
import torch.nn as nn
import torch
import numpy as np

class cLinear(nn.Module):
    '''
        Linear Layer with coeficient(variation) matirx(mask)
    '''
    def __init__(self,inDim,outDim,use_bias = True):
        super(cLinear,self).__init__()
        layer = nn.Linear(inDim,outDim)
        self.W = layer.weight
        self.b = layer.bias
        self.inDim = inDim
        self.outDim = outDim
        self.use_cuda = torch.cuda.is_available()
        self.C_W = None
        self.C_b = None
        self.use_bias = use_bias

    def generate_random_mask(self,noise_scale):
        # generate random Gaussian mask: C_W and C_b
        if self.use_cuda:
            if noise_scale == 0:
                CW = torch.cuda.FloatTensor(self.inDim,self.outDim).zero_()
                Cb = torch.cuda.FloatTensor(self.outDim).zero_()
            else:
                CW = torch.cuda.FloatTensor(self.inDim,self.outDim).normal_(0,noise_scale)
                Cb = torch.cuda.FloatTensor(self.outDim).normal_(0,noise_scale)
        else:
            CW = torch.FloatTensor(self.inDim,self.outDim).normal_(0,noise_scale)
            Cb = torch.FloatTensor(self.outDim).normal_(0,noise_scale)

        self.C_W = CW.transpose(1,0) + 1
        self.C_b = Cb + 1

    def forward(self,x):
        # C_W is the coefficient matrix for weight
        # C_b is the coefficient matrix for bias
        C_W = self.C_W
        C_b = self.C_b
        if self.use_cuda:
            if C_W == None or C_b == None:
                CW = torch.cuda.FloatTensor(self.outDim,self.inDim).zero_() + 1
                Cb = torch.cuda.FloatTensor(self.outDim).zero_() + 1
            else:
                CW = C_W
                Cb = C_b
        else:
            if C_W == None or C_b == None:
                CW = torch.FloatTensor(self.outDim,self.inDim).zero_() + 1
                Cb = torch.FloatTensor(self.outDim).zero_() + 1
            else:
                CW = C_W
                Cb = C_b

        PW = torch.mul(CW,self.W)
        Pb = torch.mul(Cb,self.b)
        out = torch.matmul(x, PW.transpose(1,0))
        if self.use_bias:
            out = out + Pb
        return out

class cMLP(nn.Module):
    def __init__(self,input_dim=1,output_dim=1,num_layers=1,num_hidden_neurons=1):
        super(cMLP, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.num_layers = num_layers
        self.layers = nn.Sequential()
        self.mode = 0
        if num_layers == 1:
            self.layers.add_module('layer1',cLinear(input_dim,output_dim))
        else:
            self.layers.add_module('layer1',cLinear(input_dim,num_hidden_neurons))
            self.layers.add_module('relu1',nn.ReLU())
            for i in range(num_layers-2):
                self.layers.add_module('layer%s'%(i+2),cLinear(num_hidden_neurons,num_hidden_neurons))
                self.layers.add_module('relu%s'%(i+2),nn.ReLU())
            self.layers.add_module('layer%s'%(num_layers),cLinear(num_hidden_neurons,output_dim))
        self.C = [None for i in range(num_layers)]

    def print_mask(self):
        for m in self.modules():
            if isinstance(m,cLinear):
                print(m)
                print(m.C_W)
                print(m.C_b)

    def generate_mask(self,noise_scale):
        for m in self.modules():
            if isinstance(m,cLinear):
                if self.mode == 0:
                    m.generate_random_mask(noise_scale)
                if self.mode == 1: #train mask mode
                    m.generate_random_mask_asparas(noise_scale)

    def clear_mask(self):
        # clear all masks used in cLinear layers
        for m in self.modules():
            if isinstance(m,cLinear):
                m.C_W = None
                m.C_b = None

    def forward(self,x):
        return self.layers(x)

    def print_grad(self):
        for name, para in self.named_parameters():
            print(name,'shape',(para.size()))
            print(para.grad)

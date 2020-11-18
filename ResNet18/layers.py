#! /usr/bin/env python3
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class cLinear(nn.Module):
    '''
        Linear Layer with coeficient matirx
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
        # generate random Gaussian mask for C_W and C_b
        if self.use_cuda:
            if noise_scale == 0:
                CW = torch.cuda.FloatTensor(self.inDim,self.outDim).zero_()
                Cb = torch.cuda.FloatTensor(self.outDim).zero_()
            else:
                CW = torch.cuda.FloatTensor(self.inDim,self.outDim).normal_(0,noise_scale)
                Cb = torch.cuda.FloatTensor(self.outDim).normal_(0,noise_scale)
        else:
            if noise_scale == 0:
                CW = torch.FloatTensor(self.inDim,self.outDim).zero_()
                Cb = torch.FloatTensor(self.outDim).zero_()
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


class cConv2d(nn.Module):
    '''
        conv2d Layer with coeficient matirx
    '''
    def __init__(self,in_channels,out_channels,kernel_size, stride=1,
                    padding=0, dilation=1, groups=1,bias=True,
                    padding_mode='zeros',noise_scale = 0):
        super(cConv2d,self).__init__()
        layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                            kernel_size = kernel_size, stride = stride, padding = padding,
                            dilation = dilation, groups = groups, bias = bias, padding_mode = padding_mode)
        self.W = layer.weight
        if bias:
            self.b = layer.bias
        if type(kernel_size) == type((1,1)):
            # tuple
            self.weight_shape = (out_channels,in_channels,kernel_size[0],kernel_size[1])
        else:
            self.weight_shape = (out_channels,in_channels,kernel_size,kernel_size)

        self.bias_shape = (out_channels)

        self.padding = padding
        self.use_bias = bias
        self.stride = stride
        self.use_cuda = torch.cuda.is_available()
        self.C_W = None
        self.C_b = None

    def generate_random_mask(self,noise_scale):
        # generate random Gaussian mask for C_W and C_b
        o,i,h,w = self.weight_shape
        if self.use_cuda:
            if noise_scale == 0:
                CW = torch.cuda.FloatTensor(o,i,h,w).zero_()
                Cb = torch.cuda.FloatTensor(self.bias_shape).zero_()
            else:
                CW = torch.cuda.FloatTensor(o,i,h,w).normal_(0,noise_scale)
                Cb = torch.cuda.FloatTensor(self.bias_shape).normal_(0,noise_scale)
        else:
            if noise_scale == 0:
                CW = torch.FloatTensor(o,i,h,w).zero_()
                Cb = torch.FloatTensor(self.bias_shape).zero_()
            else:
                CW = torch.FloatTensor(o,i,h,w).normal_(0,noise_scale)
                Cb = torch.FloatTensor(self.bias_shape).normal_(0,noise_scale)

        self.C_W = CW + 1
        self.C_b = Cb + 1

    def forward(self,x):
        # C_W is the coefficient matrix for weight
        # C_b is the coefficient matrix for bias
        o,i,h,w = self.weight_shape
        C_W = self.C_W
        C_b = self.C_b
        if self.use_cuda:
            if C_W == None or C_b == None:
                CW = torch.cuda.FloatTensor(o,i,h,w).zero_() + 1
                Cb = torch.cuda.FloatTensor(self.bias_shape).zero_() + 1
            else:
                CW = C_W
                Cb = C_b
        else:
            if C_W == None or C_b == None:
                CW = torch.FloatTensor(o,i,h,w).zero_() + 1
                Cb = torch.FloatTensor(self.bias_shape).zero_() + 1
            else:
                CW = C_W
                Cb = C_b

        PW = torch.mul(CW,self.W)

        if self.use_bias:
            out = F.conv2d(x,PW,bias = torch.mul(Cb,self.b),padding=self.padding, stride=self.stride)
        else:
            out = F.conv2d(x,PW,bias = None,stride = self.stride,padding=self.padding)
        return out

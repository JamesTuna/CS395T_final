#! /usr/bin/env python3
from trainer import *
from cResNet import *
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import argparse
# parser
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, help='model to test')
parser.add_argument('--samples', type=int, default=1000, help='number of devices simulated')
parser.add_argument('--noise',type=float, default=0,help='noise std')
parser.add_argument('--logdir',type=str, default='./',help='where to store test results')
parser.add_argument('--cuda',type=int, default=None,help='cuda index if use cuda')
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root='../rbls/data', train=False,download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,shuffle=False)

# GPU
device = torch.device("cuda:%s"%args.cuda if args.cuda is not None else "cpu")
model = rresnet18(num_classes=10)
# load saved model
print("try load %s"%(args.load))
model.load_state_dict(torch.load(args.load,map_location=torch.device(device)))
print(args.load+" loaded")
model.to(device)
loss = nn.CrossEntropyLoss()
# trainer obj
trainer = RobustTrainer(model,train_loader=None,test_loader=test_loader,loss=loss,
                noise_scale=args.noise,epochs=None,daso_n=None,
                    lr=None,cuda=None if args.cuda is None else "cuda:%s"%args.cuda)

trainer.test(noise_scale=args.noise,repeat=args.samples,logdir=args.logdir)

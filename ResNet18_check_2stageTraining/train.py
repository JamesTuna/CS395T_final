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
parser.add_argument('--load', type=str, default=None, help='resume training from saved model')
parser.add_argument('--n', type=int, default=1, help='number of noises sampled each step')
parser.add_argument('--opt',type=str, default = 'Adam', help='optimizer to do training')
parser.add_argument('--lr',type=float, default=0.001,help='learning rate')
parser.add_argument('--batch-size',type=int, default=32,help='batch size')
parser.add_argument('--epoch',type=int, default=120,help='epochs to train')
parser.add_argument('--noise',type=float, default=0,help='noise std')
parser.add_argument('--ps',type=int, default=1000,help='how many steps to print information in console')
parser.add_argument('--logdir',type=str, default='./',help='where to store training log information')
parser.add_argument('--saveas',type=str, default='./',help='where to store model')
parser.add_argument('--save-per-epochs',type=int, default=100,help='how many epochs to store the model')
parser.add_argument('--decay-epochs',type=int, default=400,help='how many epochs to decay lr')
parser.add_argument('--decay-rate',type=float, default=0.1,help='lr decay ratio')
parser.add_argument('--cuda',type=int, default=None,help='cuda index if use cuda')
parser.add_argument('--continueEp',type=int, default=0,help='if load a trained model and wish to keep train it, specify the break point epoch')
args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root='../rbls/data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='../rbls/data', train=False,download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,shuffle=False)

# GPU
device = torch.device("cuda:%s"%args.cuda if args.cuda is not None else "cpu")
model = rresnet18(num_classes=10)
# load saved model or not
if args.load is not None:
    model.load_state_dict(torch.load(args.load,map_location=device))
    print(args.load+" loaded")
model.to(device)
loss = nn.CrossEntropyLoss()

if (not args.continueEp == 0) and args.load is not None:
    try:
        trained = int((args.load).split("epoch")[1])
        print("trained %s"%trained)
        if trained == args.continueEp:
            print("safely keep training after %s epochs"%trained)
        else:
            print('warning: %s already trained %s epochs, not match args.continueEp=%s'%(args.load,trained,args.continueEp))
    except:
        print("trained epoch unknown for model %s"%args.load)

if (not args.continueEp == 0) and args.load is None:
    print("when train from scratch, argument --continue must be 0")
    exit(1)



# optimizer
trainer = RobustTrainer(model,train_loader=train_loader,test_loader=test_loader,loss=loss,
                noise_scale=args.noise,epochs=args.epoch,daso_n=args.n,
                    lr=args.lr,cuda=None if args.cuda is None else "cuda:%s"%args.cuda)

trainer.train_daso_n(saveas=args.saveas,continueEp=args.continueEp,save_per_epochs=args.save_per_epochs,logdir=args.logdir,print_step=args.ps,
                        optimizer=args.opt,reduce_lr_per_epochs=args.decay_epochs,reduce_rate=args.decay_rate)

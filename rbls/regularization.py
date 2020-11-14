from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 512) # 4
        self.fc2 = nn.Linear(512, 64) # 6
        self.fc3 = nn.Linear(64, 10) # 8

    def forward(self, x):
        # feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # Dense layers, regularization only apply on this part
        # The structure is:
        # x -(fc1)-> lin1 -(fc2)-> lin2 -(fc3)-> lin3/output
        lin1 = self.fc1(x)
        lin1 = F.relu(lin1)
        lin1 = self.dropout2(lin1)
        lin2 = self.fc2(lin1)
        lin2 = F.relu(lin2)
        lin2 = self.dropout3(lin2)
        lin3 = self.fc3(lin2)
        output = F.log_softmax(lin3, dim=1)
        return output, x, lin1, lin2


def reg_loss(model, data, target, args):
    output, x, lin1, lin2 = model(data)
    classification_loss = F.nll_loss(output, target)
    weight = np.sqrt(args.noise_data_train**2 + args.noise_train**2 + (args.noise_data_train**2)*(args.noise_train**2))
    regularization = 0.0
    for i, param in enumerate(model.parameters()):
        if i == 4:  # fc1
            regularization = regularization + torch.norm(weight * torch.norm(x, dim=0) * param) ** 2
        if i == 6:  # fc2
            regularization = regularization + torch.norm(weight * torch.norm(lin1, dim=0) * param) ** 2
        if i == 8:  # fc3
            regularization = regularization + torch.norm(weight * torch.norm(lin2, dim=0) * param) ** 2

    return classification_loss + args.lamb * regularization


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = reg_loss(model, data, target, args)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), flush=True)
            if args.dry_run:
                break


def test(model, device, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data * np.random.normal(1, args.noise_data_test, data.shape)
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output, _, _, _ = model(data)
            test_loss = test_loss + F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = correct + pred.eq(target.view_as(pred)).sum().item()

    test_loss = test_loss / len(test_loader.dataset)

    print(' - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), flush=True)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Robust training via regularization')
    parser.add_argument('--noise-data-train', type=float, default=0,
                        help='variation std of the training data matrix (default: 0)')
    parser.add_argument('--noise-train', type=float, default=0,
                        help='variation std assumed in solving RBLS (default: 0)')
    parser.add_argument('--noise-data-test', type=float, default=0,
                        help='variation std of the test data matrix (default: 0)')
    parser.add_argument('--noise-test', type=float, default=2,
                        help='variation std during test (default: 2)')
    parser.add_argument('--lamb', type=float, default=1.0,
                        help='penalty coefficient for the regularization term (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight-decay parameter in the optimizer (default: 0.0)')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enable CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=2020, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        print('Use GPU:', flush=True)
        print(f"- Name of GPU is: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"- Number of CPU threads {torch.get_num_threads()}", flush=True)
    else:
        print('args.cuda: %s, torch.cuda: %s' % (args.cuda, torch.cuda.is_available()))

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if not os.path.exists('./data'):
        os.makedirs('./data')

    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, args)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_robust_reg.pt")


if __name__ == '__main__':
    main()

#! /usr/bin/env python3

from cMLP import *
import copy
import numpy as np
import matplotlib.pyplot as plt
import time

class RobustTrainer():
    def __init__(self,model,train_loader,test_loader=None,loss = nn.MSELoss(),
                    noise_scale=0.5,train_epochs=100,n_noises=1,lr=1e-3,cuda=None):
        self.model = model
        self.model_copy = copy.deepcopy(model)
        self.train_loader = train_loader
        self.noise_scale = noise_scale
        self.train_epochs = train_epochs
        self.n_noises = n_noises
        self.lr = lr
        self.loss = loss
        self.test_loader = test_loader
        self.cuda = cuda  # none: no cuda otherwise be string like "cuda:0"

        self.model.to("cpu" if cuda is None else cuda)
        self.model_copy.to("cpu" if cuda is None else cuda)

    def train_n_noises(self,device,print_step=1000,optimizer='Adam',logdir='./',reduce_lr_per_epochs=None,reduce_rate=0.5):
        # self.n_noises is the number of random noise matrix
        # gradient was calculated w.r.t the noise matrix resulting in largest loss
        if optimizer=='SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        log_train_loss = []
        for epoch in range(self.train_epochs):
            lr = self.lr
            if reduce_lr_per_epochs is not None:
                lr = (reduce_rate ** (epoch//reduce_lr_per_epochs))*self.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            running_loss = 0
            for i, data in enumerate(self.train_loader, 0):
                self.model.train()
                self.model_copy = copy.deepcopy(self.model) # self.model stores parameters, model_copy is to find the worst (weight+noise) among n noises
                x, label = data
                x = x.view(-1,28*28).to(device)
                label = label.to(device)
                # find the largest loss among self.n_noises variations
                max_loss = 0
                for inner_iter in range(self.n_noises):
                    self.model_copy.generate_mask(self.noise_scale)
                    output_ = self.model_copy(x)
                    l_ = self.loss(output_, label).cpu().data.item()
                    if l_ > max_loss:
                        max_loss = l_
                        for layer_id in range(self.model.num_layers):
                            layer_name = "layers.layer%s"%(layer_id+1)
                            layer = eval("self.model."+layer_name)
                            layer_copy = eval("self.model_copy."+layer_name)
                            layer.C_W = layer_copy.C_W
                            layer.C_b = layer_copy.C_b
                # forward a batch using model with the worst variation to calculate gradient
                optimizer.zero_grad()
                output = self.model(x)
                l = self.loss(output,label)
                l.backward()
                optimizer.step()
                running_loss = 0.9 * running_loss + 0.1 * l.data.item()

            # log training and test loss at the end of each epoch
            print("at the end of epoch %s, lr %s"%(epoch,lr))
            print("running loss %.4f"%(running_loss))
            log_train_loss.append(running_loss)

            correct = 0
            total = 0
            accumulative_loss = 0
            count = 0
            self.model.eval()
            self.model.clear_mask()
            for t_images, t_labels in self.test_loader:
                count += 1
                t_images = t_images.view(-1,28*28).to(device)
                t_outputs = self.model(t_images)
                t_labels = t_labels.to(device)
                t_loss = self.loss(t_outputs,t_labels)
                accumulative_loss += t_loss.data.item()
                _, t_predicted = torch.max(t_outputs.data, 1)
                total += t_labels.size(0)
                correct += (t_predicted == t_labels).sum()
            acc = (correct.data.item()/ total)
            print('test loss: %.4f, test acc: %.4f'%(accumulative_loss/count, acc))

        # plot curves and save under logdir
        log_train_loss = np.array(log_train_loss)
        plt.plot(log_train_loss)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Training Loss with n=%s'%self.n_noises)
        plt.savefig(logdir+'/training_loss_SD.pdf')

    def test(self,noise_scale,repeat=100,use_cuda = torch.cuda.is_available(),logdir="./"):
        # test performance under random sampled noise
        # save figures and statistics in logdir
        self.model.eval()
        loss_list = []
        acc_list = []
        for test in range(repeat):
            self.model_copy = copy.deepcopy(self.model)
            print('test no.%s'%(test))
            model_sdict = self.model_copy.state_dict()
            for name in model_sdict:
                para = model_sdict[name]
                shape = para.size()
                if use_cuda:
                    para *= (torch.cuda.FloatTensor(shape).normal_(0,noise_scale) + 1)
                else:
                    para *= (torch.FloatTensor(shape).normal_(0,noise_scale) + 1)
                model_sdict[name] = para
            self.model_copy.load_state_dict(model_sdict)
            # performance on testset
            correct = 0
            total = 0
            accumulative_loss = 0
            count = 0
            self.model_copy.eval()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for t_images, t_labels in self.test_loader:
                count += 1
                t_images = t_images.view(-1,28*28).to(device)
                t_outputs = self.model_copy(t_images)
                t_labels = t_labels.to(device)
                t_loss = self.loss(t_outputs,t_labels)
                accumulative_loss += t_loss.data.item()
                _, t_predicted = torch.max(t_outputs.data, 1)
                total += t_labels.size(0)
                correct += (t_predicted == t_labels).sum()
            acc = (correct.data.item()/ total)
            print('test loss: %.4f, test acc: %.4f'%(accumulative_loss/count, acc))
            loss_list.append(accumulative_loss/count)
            acc_list.append(acc)

        loss_list = np.array(loss_list)
        acc_list = np.array(acc_list)
        # statistics
        np.save(logdir+'/loss.npy',loss_list)
        np.save(logdir+'/acc.npy',acc_list)
        with open(logdir+'/statistics.txt','w') as f:
            for q in range(1,20):
                qtl = 0.05 * q
                qtl_loss = np.quantile(loss_list,qtl)
                qtl_acc = np.quantile(acc_list,qtl)
                f.write('%.3f quantile:\tacc: %.4f\tloss: %.4f\n'%(qtl,qtl_acc,qtl_loss))
            f.write('Expectation:\tacc: %.4f\tloss: %.4f\n'%(acc_list.mean(),loss_list.mean()))
            f.write('Variance:\tacc: %.4f\tloss: %.4f'%(acc_list.var(),loss_list.var()))
        # histogram
        #ax3 = plt.subplot(3,1,3)
        num_bins = 10
        acc_mean = acc_list.mean()
        heights,bins = np.histogram(acc_list,num_bins)
        freq = heights/sum(heights)
        width = (max(bins)-min(bins))/(len(bins))
        plt.bar(bins[:-1]+1/2*width,freq,width=width,color='blue',alpha=0.5)
        plt.xticks(np.around(bins,decimals=3))
        plt.title('Accuracy distribution of %s perturbations with mean %s'%(acc_list.shape[0],np.around(acc_mean,decimals=4)))
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.rc('grid', linestyle="-", color='black')
        plt.grid()
        plt.savefig(logdir+'/test_%s_perturbation_%s.pdf'%(repeat,noise_scale))

    def mean_var_optimization(self, lamb=10, print_step=1000, optimizer='Adam', reduce_lr_per_epochs=None,
                              reduce_rate=0.5, logdir='./', saveas='unamedModel.ckpt', save_per_epochs=100):

        # optimze a lower bound of [mean(loss) + lambda * std(loss)]
        device = "cpu" if self.cuda is None else self.cuda
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)

        log_train_loss = []
        for epoch in range(self.epochs):
            start = time.time()
            if (optimizer == 'SGD') and (reduce_lr_per_epochs is not None):
                lr = (reduce_rate ** (epoch // reduce_lr_per_epochs)) * self.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            running_loss = 0
            avg_loss = 0
            for i, data in enumerate(self.train_loader, start=0):
                self.model.train()

                x, label = data
                x = x.to(device)
                label = label.to(device)
                # step 1, sample 2 variations
                # estimate mean(loss) + lambda std(loss)

                # loss 1
                self.model_copy = copy.deepcopy(self.model)
                self.model_copy.generate_mask(self.noise_scale)
                output1 = self.model_copy(x)
                l1 = self.loss(output1, label)
                # loss 2
                self.model_copy = copy.deepcopy(self.model)
                self.model_copy.generate_mask(self.noise_scale)
                output2 = self.model_copy(x)
                l2 = self.loss(output2, label)

                est_mean = (l1 + l2) / 2
                est_std = 0.7071 * torch.norm(l1 - l2, p=1)
                l = est_mean + lamb * est_std

                # step 2
                # backprop using the new loss
                optimizer.zero_grad()
                output = self.model(x)
                l.backward()
                optimizer.step()

                running_loss = 0.9 * running_loss + 0.1 * l.data.item()
                avg_loss = (avg_loss * i + l.data.item()) / (i + 1)

                if (i % print_step) == (print_step - 1):
                    print("epoch %s step %s avg loss per batch %.4f" % (epoch, i + 1, avg_loss))

            # log training and test loss at the end of each epoch
            print("at the end of epoch %s[%.2f seconds]" % (epoch, time.time() - start))
            print("running_loss %.4f avg loss %.4f" % (running_loss, avg_loss))
            log_train_loss.append(avg_loss)

            correct = 0
            total = 0
            accumulative_loss = 0
            count = 0

            self.model.eval()
            self.model.clear_mask()
            for t_images, t_labels in self.test_loader:
                count += 1
                t_images = t_images.to(device)
                t_outputs = self.model(t_images)
                t_labels = t_labels.to(device)
                t_loss = self.loss(t_outputs, t_labels)
                accumulative_loss += t_loss.cpu().data.item()
                _, t_predicted = torch.max(t_outputs.data, 1)
                total += t_labels.size(0)
                correct += (t_predicted == t_labels).sum()

            acc = correct.data.item() / total
            print('test loss: %.4f, test acc: %.4f' % (accumulative_loss / count, acc))

            # save model
            if ((epoch % save_per_epochs) == (save_per_epochs - 1)) or (epoch == (self.epochs - 1)):
                torch.save(self.model.state_dict(), saveas + '__epoch%s' % (epoch + 1))

        # plot curves and save under logdir
        log_train_loss = np.array(log_train_loss)
        plt.plot(log_train_loss)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Training Loss with lambda=%s' % lamb)
        plt.savefig(logdir + '/training_loss.pdf')


























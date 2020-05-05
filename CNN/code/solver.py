from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optim = self.optim(model.parameters(), **self.optim_args)
        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        
        self._reset_histories()
       
        for epoch in range(num_epochs):
            counter = 0
            correct = 0
            val_correct = 0
            val_counter = 0
            test_loss = 0
            val_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                
                data, target = data.to(device), target.to(device)
               
               #data = data.view(-1, 28*28)
                optim.zero_grad()
                net_out = model(data)
                loss = self.loss_func(net_out, target)
                loss.backward()
                optim.step()
                self.train_loss_history.append(loss.data[0])
                if batch_idx % log_nth == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx /iter_per_epoch, loss.data[0]))
              
                t = (epoch*iter_per_epoch) + counter
                first_it = (t == 0)
                last_it = (t == epoch * iter_per_epoch +1)
                epoch_end = (counter == iter_per_epoch-1)
                test_loss += loss.item()
                pred_out = F.log_softmax(net_out, dim=1)
                pred = net_out.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                    
            self.train_acc_history.append(correct/len(train_loader.dataset))
            print("[EPOCH {}/{}] TRAIN loss/acc: {:.6f}/{:.6f}".format(epoch,num_epochs,test_loss/iter_per_epoch,
                                                                              self.train_acc_history[-1]))
            for batch_idx, (data, target) in enumerate(val_loader):
                
                data, target = data.to(device), target.to(device)
               
                optim.zero_grad()
                net_out = model(data)
                loss = self.loss_func(net_out, target)
                loss.backward()
                optim.step()
                self.val_loss_history.append(loss.data[0])
               
                t = (epoch*iter_per_epoch) + val_counter
                first_it = (t == 0)
                last_it = (t == epoch * iter_per_epoch +1)
                epoch_end = (val_counter == iter_per_epoch-1)
                val_loss += loss.item()
                pred_out = F.log_softmax(net_out, dim=1)
                pred = net_out.max(1, keepdim=True)[1]
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                
            self.val_acc_history.append(val_correct/len(val_loader.dataset))
            print("[EPOCH {}/{}] VAL loss/acc: {:.6f}/{:.6f}".format(epoch,num_epochs,val_loss/iter_per_epoch,
                                                                              self.val_acc_history[-1]))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

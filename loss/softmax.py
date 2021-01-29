#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

import numpy as np
from collections import OrderedDict

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised Softmax Loss')
# -------
        self.n_classes = nClasses
        self.dropmode = False # Default is the normal behaviour
        self.set_ignored_classes([])

        self.W = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_uniform_(self.W)
# -------

    def forward(self, x, label=None):
        '''
        input (x): (batch_size, num_features): FloatTensor
        label (optional): (batch_size): LongTensor
        '''
        if self.dropmode:		
            W = self.W[self.rem_classes]
            label = self.get_mini_labels(label.tolist()).cuda()
        else:
            W = self.W

        x = F.linear(x, W)
        nloss   = self.criterion(x, label)
        prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1

# -------
    def drop(self):
        self.dropmode = True
    
    def nodrop(self):
        self.dropmode = False

    def set_ignored_classes(self, ignored:list):
        if len(ignored) != 0:
            assert min(ignored) >= 0
            assert max(ignored) < self.n_classes
        self.ignored = sorted(list(set(ignored)))
        self.rem_classes = sorted(set(np.arange(self.n_classes)) - set(ignored))
        self.ldict = OrderedDict({k:v for v, k in enumerate(self.rem_classes)}) #mapping of original label to new index
        self.idict = OrderedDict({k:v for k, v in enumerate(self.rem_classes)}) #mapping of remaining indexes to original label

    def get_mini_labels(self, label:list):
        # convert list of labels into new indexes for ignored classes
        mini_labels = torch.LongTensor(list(map(lambda x: self.ldict[x], label)))
        return mini_labels
# -------

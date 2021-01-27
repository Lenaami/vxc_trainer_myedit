#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

import numpy as np
from collections import OrderedDict

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

# -------
        self.nOut = nOut
        self.n_classes = nClasses
        self.s = scale
        self.m = margin
        self.W = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_normal_(self.W, gain=1)

        self.ce = nn.CrossEntropyLoss()

        self.dropmode = False # Default is the normal behaviour
        self.set_ignored_classes([])
# -------
        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        if self.dropmode:		
        	W = self.W[self.rem_classes]
			label = self.get_mini_labels(label.detach()).to(device)
        else:
			W = self.W

        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(W)
        # dot product
        logits = F.linear(x, W)
        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        loss    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


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

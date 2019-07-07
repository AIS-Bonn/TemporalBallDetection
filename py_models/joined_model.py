#!/usr/bin/env python

""" Join sweatynet and sequential part
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'

import torch
import torch.nn as nn
import time
import numpy as np

from arguments import opt
from py_models.model import SweatyNet1, SweatyNet2, SweatyNet3
from py_models.lstm import LSTM
from py_utils.logging_setup import logger
from py_models.tcn_ed import TCN_ED
from py_models.gru import GRU

def init_weights(m):
    if type(m) == nn.Conv1d:
        m.weight.data.normal_(0, 0.01)


class Weight(nn.Module):
    def __init__(self):
        super(Weight, self).__init__()
        self.multp = nn.Parameter(torch.Tensor([0.5]))


class JoinedModel(nn.Module):
    def __init__(self):
        super(JoinedModel, self).__init__()
        self.timea = []
        if opt.net == 'net1':
            self.sweaty = SweatyNet1(1, opt.drop_p, finetune=True)
        if opt.net == 'net2':
            self.sweaty = SweatyNet2(1, opt.drop_p, finetune=True)
        if opt.net == 'net3':
            self.sweaty = SweatyNet3(1, opt.drop_p, finetune=True)
        if opt.seq_model == 'lstm':
            opt.seq_predict = 2
            self.seq = LSTM()
        if opt.seq_model == 'tcn':
            opt.seq_predict = 1
            n_nodes = [64, 96]
            self.seq = TCN_ED(n_nodes, opt.hist, opt.seq_predict, opt.ksize).to(opt.device)
            self.seq.apply(init_weights)
        if opt.seq_model == 'gru':
            opt.seq_predict = 1
            self.seq = GRU()
            #
        self.conv1 = nn.Sequential(nn.Conv2d(112, 1, 7, padding=3),
                                   nn.BatchNorm2d(1),
                                   nn.LeakyReLU())
        self.alpha = Weight()

    def forward(self, x):
        start = time.time()
        x, out23 = self.sweaty(x)
        end = time.time()
        out23 = self.alpha.multp * self.conv1(out23).squeeze()
        x = x + out23
        if opt.seq_model == 'tcn':
            x = x.view(-1, opt.hist, opt.map_size_x * opt.map_size_y)
        if opt.seq_model == 'lstm':
            x = x.view(-1, opt.hist, opt.map_size_x, opt.map_size_y)
        if opt.seq_model == 'gru':
            x = x.view(-1, opt.hist, opt.map_size_x, opt.map_size_y)
        #start = time.time()
        x = self.seq(x)
        #end = time.time()
        #logger.debug('time again: %s' % (end - start))
        self.timea.append(end-start)
        if len(self.timea) == 10:
            logger.debug('average time per 10 images %s' % str(np.mean(self.timea)))
            self.timea = []
        return x

    def test(self, x, ret_out23=False):
        start = time.time()
        x, out23 = self.sweaty(x)
        end = time.time()
        # logger.debug('time: %s' % str(end - start))
        self.timea.append(end-start)
        if len(self.timea) == 30:
            # logger.debug('average time %s' % str(np.mean(self.timea)))
            self.timea = []
        if ret_out23:
            out23 = self.conv1(out23).squeeze()
            return x, out23
        return x

    def resume_sweaty(self, path):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict_model']
        self.sweaty.load_state_dict(checkpoint)

    def resume_seq(self, path):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict']
        self.seq.load_state_dict(checkpoint)

    def resume_both(self, path):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict_model']
        self.load_state_dict(checkpoint)

    def off_sweaty(self):
        for param in self.sweaty.parameters():
            param.requires_grad = False

    def on_sweaty(self):
        for param in self.sweaty.parameters():
            param.requires_grad = True


def create_model():
    torch.manual_seed(opt.manualSeed)
    model = JoinedModel()
    if not opt.seq_resume:
        model.off_sweaty()
    if opt.device == 'cuda':
        model = model.cuda()
    loss = nn.MSELoss(reduction='sum')
    # loss = nn.BCELoss()
    optimizer_seq = torch.optim.Adam(list(model.seq.parameters()) +
                                 list(model.conv1.parameters()) +
                                 list(model.alpha.parameters()),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    params = list(model.seq.parameters()) + list(model.conv1.parameters()) + list(model.alpha.parameters())
    both_lr = opt.lr if opt.seq_resume else opt.lr * 0.1
    optimizer_both = torch.optim.Adam([
        {'params': model.sweaty.parameters(), 'lr': both_lr},
        {'params': params}
                                       ],
                                      lr=opt.lr,
                                      weight_decay=opt.weight_decay)
    # logger.debug(str(model))
    # logger.debug(str(loss))
    # logger.debug(str(optimizer_seq))
    # logger.debug(str(optimizer_both))

    return model, loss, optimizer_seq, optimizer_both






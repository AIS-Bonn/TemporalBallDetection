#!/usr/bin/env python

__author__ = 'Anna Kukleva'
__date__ = 'February 2019'


import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from arguments import opt
from py_utils.logging_setup import logger
from py_utils.util_functions import dir_check


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, x):
        max_vals, _ = torch.max(torch.abs(x), 2, keepdim=True)
        max_vals  = max_vals + 1e-5
        x = x / max_vals
        return x


def init_weights(m):
    if type(m) == nn.Conv1d:
        m.weight.data.normal_(0, 0.01)


class TCN_ED(nn.Module):
    def __init__(self, n_nodes, n_inputs, n_outputs, kernel):
        super(TCN_ED, self).__init__()
        n_layers = len(n_nodes)
        layers = []

        # encoder
        for i in range(n_layers):
            input_dim = n_inputs if i == 0 else n_nodes[i-1]
            dilation = i + 1
            padding = (dilation) * (kernel - 1)
            conv = nn.Conv1d(input_dim, n_nodes[i], kernel,
                             padding=padding,
                             dilation=dilation)
            chomp = Chomp1d(padding)
            dropout = nn.Dropout(opt.dropout)
            relu = nn.ReLU()
            channel_norm = ChannelNorm()

            layers += [conv, chomp, dropout, relu, channel_norm]

        # decoder
        for i in range(n_layers):
            output_dim = n_nodes[-i-2] if i < n_layers - 1 else n_outputs
            dilation = n_layers - i
            padding = (dilation) * (kernel - 1)
            conv = nn.Conv1d(n_nodes[-i-1], output_dim, kernel,
                             padding=padding,
                             dilation=dilation)
            chomp = Chomp1d(padding)
            dropout = nn.Dropout(opt.dropout)
            if i < n_layers-1:
                relu = nn.ReLU()
                channel_norm = ChannelNorm()
                layers += [conv, chomp, dropout, relu, channel_norm]
            else:
                layers += [conv, chomp, dropout, nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, opt.hist,  opt.map_size_x * opt.map_size_y)
        x = self.net(x)
        x = x.view(-1, opt.seq_predict, opt.map_size_x, opt.map_size_y)
        return x


def create_model():
    torch.manual_seed(opt.seed)
    n_nodes = [64, 96]
    model = TCN_ED(n_nodes, opt.hist, opt.seq_predict, opt.ksize).to(opt.device)
    model.apply(init_weights)

    # loss = nn.MSELoss(reduction='sum')
    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(list(model.net.parameters()),
                                 # list(model.fc.parameters()),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer


def test(dataloader, model, out=False):
    model.eval()
    model.to(opt.device)
    dir_check(os.path.join(opt.save_out, opt.seq_model))
    dir_check(os.path.join(opt.save_out, opt.seq_model, opt.suffix))
    time_log = [0., 0]
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            #if i % 5:
            #    continue
            data = data.float().squeeze()
            target = target.float().numpy().squeeze()
            if opt.device == 'cuda':
                data = data.cuda(non_blocking=True)

            start = time.time()
            output = model(data).to('cpu').numpy().squeeze()
            end = time.time()
            #logger.debug('time: %s' % str(end - start))
            if opt.reproduce == 'time':
                time_log[0] += (end - start)
                time_log[1] += 1
                if time_log[1] == 100:
                    #logger.debug('time: %s' % str(time_log[0] / time_log[1]))
                    return
                continue
            img = None
            color = 0.5
            if len(output.shape) == 2:
                output = output[np.newaxis, :]
                target = target[np.newaxis, :]
            horizontal_line = np.ones((3, output[0].shape[1])) * color
            vertical_line = np.ones((2 * output[0].shape[0] + 3, 3)) * color
            sweaty_out, out23 = model.test(data, ret_out23=True)
            sweaty_out = sweaty_out.cpu().numpy()[-1]
            sweaty_out = sweaty_out / np.max(sweaty_out)
            out23 = out23.cpu().numpy()[-1]
            out23 = out23 / np.max(out23)
            if not out:
                for idx in range(opt.seq_predict):
                    tmp_img = np.concatenate((output[idx], horizontal_line, target[idx]), axis=0)
                    if img is None:
                        img = tmp_img
                    else:
                        img = np.concatenate((img, vertical_line, tmp_img), axis=1)

                tmp_img = np.concatenate((sweaty_out, horizontal_line, out23), axis=0)
                img = np.concatenate((img, vertical_line, tmp_img), axis=1)
                img = plt.imshow(img)
                plt.savefig(os.path.join(opt.save_out, opt.seq_model, opt.suffix, 'out%d.png' % i))
            else:
                data = data.to('cpu').numpy().squeeze()[-1:]
                d_v_l = np.ones((3, data.shape[2], 3)) * color
                # data = np.concatenate((data[0], d_v_l, data[1], d_v_l, data[2]), axis=2)
                d_v_l = np.ones((out23.shape[0], 3)) * color
                target[-1] = target[-1] / np.max(target[-1])
                output[-1] = output[-1] / np.max(output[-1])
                # tmp_img = np.concatenate((sweaty_out, d_v_l, out23, d_v_l, target[-1], d_v_l, output[-1]), axis=1)
                tmp_img = np.concatenate((out23, d_v_l, target[-1], d_v_l, output[-1]), axis=1)
                # horizontal_line = np.ones((5, data.shape[0])) * color
                # img = np.concatenate((data, horizontal_line, tmp_img), axis=0)
                data = data.squeeze().transpose(1, 2, 0)
                plt.axis('off')
                data = plt.imshow(data)
                plt.savefig(os.path.join(opt.save_out, opt.seq_model, opt.suffix, 'hist%d.png' % i))
                plt.axis('off')
                tmp_img = plt.imshow(tmp_img)
                plt.savefig(os.path.join(opt.save_out, opt.seq_model, opt.suffix, 'real.gt.pr.%d.png' % i))




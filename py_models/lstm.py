import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from py_models.convolution_lstm import ConvLSTM
from py_utils.logging_setup import logger
from arguments import opt
from py_utils.util_functions import dir_check


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.clstm = ConvLSTM(input_channels=opt.hist,
                              hidden_channels=[32, 64, 32, opt.seq_predict],
                              kernel_size=5,
                              step=5,
                              effective_step=[4])
        self.time_log = [0.0, 0]

    def forward(self, x):
        start = time.time()
        x = self.clstm(x)
        end = time.time()
        self.time_log[0] += (end - start)
        self.time_log[1] += 2
        #logger.debug('time %s' % (time.time() - start))
        return x


def create_model():
    torch.manual_seed(opt.seed)
    model = LSTM()
    loss = nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer


def test(dataloader, model, out=False):
    idx = 0
    model.eval()
    model.to(opt.device)
    dir_check(os.path.join(opt.save_out, opt.seq_model))
    dir_check(os.path.join(opt.save_out, opt.seq_model, opt.suffix))
    saved = 0
    time_log = [0., 0]
    with torch.no_grad():
        for i, data_item in enumerate(dataloader):
            if 'new' in opt.dataset:
                data, target, _, _ = data_item
                data = data.float()
            else:
                #if i % 10:
                #    continue
                saved += 1
                # if saved == 5:
                #     break
                data, target = data_item
                data = data.float().squeeze()

            target = target.float().numpy().squeeze()
            if opt.device == 'cuda':
                data = data.cuda(non_blocking=True)

            start = time.time()
            output, (h, cc) = model(data)
            end = time.time()
            if opt.reproduce == 'time':
                time_log[0] += (end - start)
                time_log[1] += 1
                if time_log[1] == 100:
                    #model.time_out()

                    #logger.debug('time: %s' % str(time_log[0] / time_log[1]))
                    return
                continue
            output = output[0].cpu().numpy().squeeze()

            img = None
            color = np.max(output)
            if len(output.shape) == 2:
                output = output[np.newaxis, :]
                target = target[np.newaxis, :]
            horizontal_line = np.ones((5, output[0].shape[1])) * color
            vertical_line = np.ones((2 * output[0].shape[0] + 5, 5)) * color
            sweaty_out, out23 = model.test(data, ret_out23=True)
            sweaty_out = sweaty_out.cpu().numpy()[-1]
            sweaty_out = sweaty_out / np.max(sweaty_out)
            out23 = out23.cpu().numpy()[-1]
            out23 = out23 / np.max(out23)
            for idx in range(opt.seq_predict):
                tmp_img = np.concatenate((output[idx], horizontal_line, target[idx]), axis=0)
                if img is None:
                    img = tmp_img
                else:
                    img = np.concatenate((img, vertical_line, tmp_img), axis=1)
            tmp_img = np.concatenate((sweaty_out, horizontal_line, out23), axis=0)
            img = np.concatenate((img, vertical_line, tmp_img), axis=1)
            img_ = plt.imshow(img)
            plt.text(1, -5, 'first row left: lstm prediction frame t \nfirst row middle: lstm prediction frame t+1 \nfirst row right: sweatynet output for t-1',
                     # verticalalignment='bottom', horizontalalignment='right',
                     color='green', fontsize=15)

            plt.text(1, img.shape[0] + 80, 'second row left: target t\nsecond row middle: target t+1\nsecond row right: residual information',
                     # verticalalignment='bottom', horizontalalignment='right',
                     color='green', fontsize=15)
            plt.axis('off')
            plt.savefig(os.path.join(opt.save_out, opt.seq_model, opt.suffix, '%d_lstm_output.png' % i))
            plt.clf()




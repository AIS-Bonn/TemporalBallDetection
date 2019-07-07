#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'March 2019'

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os
import random

from arguments import opt
from py_utils.logging_setup import path_logger, logger

from py_train.test_finetuned import data1, data2, eval_data1, eval_data2, data3, eval_data3, data_multi
from py_dataset.seq_dataset import NewDataset

from py_utils.util_functions import dir_check

path_logger()
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

vars_iter = list(vars(opt))
for arg in sorted(vars_iter):
    logger.debug('%s: %s' % (arg, getattr(opt, arg)))

if opt.dataset == 'provided':
    opt.data_root_seq = 'SoccerDataSeq'
    sweaty_model_template = 'model/sweaty/Model_lr_0.001_opt_adam_epoch_100_net_net%d_drop_0.%d'

    testloader_data1 = data1()
    if opt.reproduce == 'all':

        for sweaty_dp in [0, 3, 5]:
            for sweaty_net in [1, 2, 3]:
                sweaty_model = sweaty_model_template % (sweaty_net, sweaty_dp)
                opt.drop_p = sweaty_dp * 0.1
                opt.net = 'net%d' % sweaty_net

                opt.seq_both_resume = False
                opt.sweaty_resume_str = sweaty_model
                opt.seq_resume = False
                eval_data1(testloader_data1)

        sequential_models = [
            'lstm.real.scr.30',
            'lstm.real.ft.20',
            'tcn.real.scr.30',
            'tcn.real.ft.20',
            'gru.real.ft.70',
            'gru.real.scr.30'
        ]

        opt.seq_both_resume = True
        opt.net = 'net1'
        for model_name in sequential_models:
            if 'lstm' in model_name:
                opt.seq_model = 'lstm'
                opt.seq_predict = 2
            if 'tcn' in model_name:
                opt.seq_model = 'tcn'
                opt.seq_predict = 1
            if 'gru' in model_name:
                opt.seq_model = 'gru'
                opt.seq_predict = 1
            opt.seq_both_resume_str = 'model/both/' + model_name
            eval_data1(testloader_data1)

    if opt.reproduce == 'best':
        sweaty_model = 'model/sweaty/Model_lr_0.001_opt_adam_epoch_100_net_net1_drop_0.5'
        opt.drop_p = 0.5
        opt.seq_both_resume = False
        opt.sweaty_resume_str = sweaty_model
        opt.seq_resume = False
        eval_data1(testloader_data1)

        opt.seq_both_resume = True
        opt.seq_both_resume_str = 'model/both/lstm.real.ft.20'
        if 'lstm' in opt.seq_both_resume_str:
            opt.seq_model = 'lstm'
            opt.seq_predict = 2
        if 'tcn' in opt.seq_both_resume_str:
            opt.seq_model = 'tcn'
            opt.seq_predict = 1
        if 'gru' in opt.seq_both_resume_str:
            opt.seq_model = 'gru'
            opt.seq_predict = 1
        eval_data1(testloader_data1)

    if opt.reproduce == 'time':
        sweaty_model_template = 'model/sweaty/Model_lr_0.001_opt_adam_epoch_100_net_net%d_drop_0.5'
        opt.drop_p = 0.5
        opt.seq_both_resume = True
        sequential_models = [
            'lstm.real.ft.20',
            'tcn.real.ft.20',
            'gru.real.scr.30'
        ]
        sweaty_models = [1,2,3]
        opt.net = 'net1'
        testloader_data2 = data2()
        for sweaty_n in sweaty_models:
            sweaty_model = sweaty_model_template % sweaty_n
            opt.net = 'net%d' % sweaty_n
            opt.sweaty_resume_str = sweaty_model
            opt.seq_both_resume = False
            opt.seq_resume = False
            eval_data2(testloader_data2)


        '''
        for model_name in sequential_models:
            if 'lstm' in model_name:
                opt.seq_model = 'lstm'
                opt.seq_predict = 2
            if 'tcn' in model_name:
                opt.seq_model = 'tcn'
                opt.seq_predict = 1
            if 'gru' in model_name:
                opt.seq_model = 'gru'
                opt.seq_predict = 1
            opt.seq_both_resume_str = 'model/both/' + model_name
            eval_data2(testloader_data2)
        '''



        #logger.debug('saving some visualization in seq_output folder...')
        #testloader_data2 = data2()
        #eval_data2(testloader_data2)

if 'new' in opt.dataset:
    opt.suffix = 'new_dataset'
    newdataset = NewDataset(transform=transforms.Compose([
                                transforms.ColorJitter(brightness=0.3,
                                                       contrast=0.4, saturation=0.4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ]))

    opt.batch_size = 1
    opt.workers = 1
    testloader = torch.utils.data.DataLoader(newdataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.workers,
                                             drop_last=False)

    opt.seq_both_resume = True
    opt.seq_both_resume_str = 'model/both/lstm.real.ft.20'
    opt.seq_model = 'lstm'
    opt.seq_predict = 2
    dir_check(opt.save_out)
    dir_check(os.path.join(opt.save_out, opt.seq_model))
    dir_check(os.path.join(opt.save_out, opt.seq_model, opt.suffix))
    eval_data1(testloader)
    if opt.dataset == 'new_seq':
        testloader = torch.utils.data.DataLoader(newdataset,
                                                 batch_size=20,
                                                 shuffle=False,
                                                 num_workers=opt.workers,
                                                 drop_last=False)
        eval_data2(testloader)

if opt.dataset == 'multi':
    sweaty_model = 'model/sweaty/Model_lr_0.001_opt_adam_epoch_100_net_net1_drop_0.5'
    opt.data_root_seq = 'SoccerDataMulti'
    opt.drop_p = 0.5
    opt.seq_both_resume = False
    opt.sweaty_resume_str = sweaty_model
    opt.seq_resume = False
    # testloader_data_multi = data_multi()
    # eval_data3(testloader_data_multi)

    opt.seq_both_resume = True
    opt.seq_both_resume_str = 'model/both/lstm.real.ft.20'
    opt.seq_model = 'lstm'
    opt.seq_predict = 2
    test_data3 = data3()
    # eval_data2(test_data3)

    sequential_models = [
        # 'lstm.real.scr.30',
        # 'lstm.real.ft.20',
        'tcn.real.scr.30',
        'tcn.real.ft.20',
        # 'gru.real.ft.70',
        # 'gru.real.scr.30'
    ]

    opt.seq_both_resume = True
    opt.net = 'net1'
    for model_name in sequential_models:
        opt.suffix = 'multi.%s' % model_name
        if 'lstm' in model_name:
            opt.seq_model = 'lstm'
            opt.seq_predict = 2
        if 'tcn' in model_name:
            opt.seq_model = 'tcn'
            opt.seq_predict = 1
        if 'gru' in model_name:
            opt.seq_model = 'gru'
            opt.seq_predict = 1
        opt.seq_both_resume_str = 'model/both/' + model_name
        eval_data2(test_data3)




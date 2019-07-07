#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'


import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import random

from arguments import opt
from py_utils.logging_setup import path_logger, logger
from py_dataset.seq_dataset import RealBallDataset, RealBallDatasetMulti
from py_utils.util_functions import dir_check
from py_models import joined_model, lstm, tcn_ed, gru
from py_train.evaluator import ModelEvaluator
from py_train.evaluator_muli import ModelEvaluator as ModelEvalMulti
from py_dataset.dataset import SoccerDataSet

#
# path_logger()
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True
#
# vars_iter = list(vars(opt))
# for arg in sorted(vars_iter):
#     logger.debug('%s: %s' % (arg, getattr(opt, arg)))


def data1():
    testset = SoccerDataSet(data_path='SoccerData1/test_cnn', map_file='test_maps',
                            transform=transforms.Compose([
                                transforms.ColorJitter(brightness=0.3,
                                                       contrast=0.4, saturation=0.4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ]))


    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=3,
                                             shuffle=False,
                                             num_workers=opt.workers,
                                             drop_last=True)
    return testloader

def data_multi():
    testset = SoccerDataSet(data_path='SoccerDataMulti/test_cnn', map_file='test_maps',
                            transform=transforms.Compose([
                                transforms.ColorJitter(brightness=0.3,
                                                       contrast=0.4, saturation=0.4),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ]))


    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=3,
                                             shuffle=False,
                                             num_workers=opt.workers,
                                             drop_last=True)
    return testloader


def eval_data1(testloader):
    opt.batch_size = opt.hist

    model, loss, optim_seq, optim_both = joined_model.create_model()
    model_name = ''
    if opt.seq_both_resume:
        model.resume_both(os.path.abspath(opt.seq_both_resume_str))
        model_name = opt.seq_both_resume_str
    else:
        model.resume_sweaty(os.path.abspath(opt.sweaty_resume_str))
        model_name = opt.sweaty_resume_str
        if opt.seq_resume:
            model_name += '_' + opt.seq_resume_str
            model.resume_seq(os.path.abspath(opt.seq_resume_str))

    opt.lr = 1e-5
    modeleval = ModelEvaluator(model, threshold=5.0535, min_radius=2.625,
                               optim_seq=optim_seq, optim_both=optim_both,
                               loss=loss)
    logger.debug('start')
    modeleval.test(model_name, testloader)
# exit(0)


# test sequences
def data2():
    testset = RealBallDataset(data_path='SoccerDataSeq',
                               transform=transforms.Compose([
                                   transforms.ColorJitter(brightness=0.3,
                                                          contrast=0.4, saturation=0.4),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               ]),
                               small=True)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=opt.workers)

    return testloader


def eval_data2(testloader):
    opt.batch_size = opt.hist

    model, loss, optim_seq, optim_both = joined_model.create_model()
    model_name = ''
    if opt.seq_both_resume:
        model.resume_both(os.path.abspath(opt.seq_both_resume_str))
        model_name = opt.seq_both_resume_str
    else:
        model.resume_sweaty(os.path.abspath(opt.sweaty_resume_str))
        model_name = opt.sweaty_resume_str
        if opt.seq_resume:
            model_name += '_' + opt.seq_resume_str
            model.resume_seq(os.path.abspath(opt.seq_resume_str))

    logger.debug('model name: %s' % model_name)
    dir_check(opt.save_out)
    dir_check(os.path.join(opt.save_out, opt.seq_model))
    model.eval()
    model = model.cuda()
    if opt.seq_model == 'lstm':
        lstm.test(testloader, model, out=True)
    if opt.seq_model == 'tcn':
        # tcn.test(testloader, model)
        tcn_ed.test(testloader, model, out=True)
    if opt.seq_model == 'gru':
        gru.test(testloader, model, out=True)


# test sequences two balls
def data3():
    testset = RealBallDatasetMulti(data_path='SoccerDataMulti',
                               transform=transforms.Compose([
                                   transforms.ColorJitter(brightness=0.3,
                                                          contrast=0.4, saturation=0.4),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               ]),
                               small=True)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=opt.workers)

    return testloader


def eval_data3(testloader):
    opt.batch_size = opt.hist

    model, loss, optim_seq, optim_both = joined_model.create_model()
    model_name = ''
    if opt.seq_both_resume:
        model.resume_both(os.path.abspath(opt.seq_both_resume_str))
        model_name = opt.seq_both_resume_str
    else:
        model.resume_sweaty(os.path.abspath(opt.sweaty_resume_str))
        model_name = opt.sweaty_resume_str
        if opt.seq_resume:
            model_name += '_' + opt.seq_resume_str
            model.resume_seq(os.path.abspath(opt.seq_resume_str))

    opt.lr = 1e-5
    modeleval = ModelEvalMulti(model, threshold=2.0535, min_radius=2.625,
                               optim_seq=optim_seq, optim_both=optim_both,
                               loss=loss)
    logger.debug('start')
    modeleval.test(model_name, testloader)


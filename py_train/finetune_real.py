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
from py_dataset.seq_dataset import RealBallDataset
from py_models import joined_model
from py_train.evaluator import ModelEvaluator
from py_dataset.dataset import SoccerDataSet


path_logger()
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

vars_iter = list(vars(opt))
for arg in sorted(vars_iter):
    logger.debug('%s: %s' % (arg, getattr(opt, arg)))


# trainset = BallDataset(opt.seq_dataset)
trainset = RealBallDataset(data_path=opt.data_root_seq,
                           transform=transforms.Compose([
                               # transforms.RandomResizedCrop(opt.input_size[1]),
                               # transforms.RandomHorizontalFlip(),
                               # transforms.RandomRotation(opt.rot_degree),
                               transforms.ColorJitter(brightness=0.3,
                                                      contrast=0.4, saturation=0.4),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                           ]),
                           small=False)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=opt.workers)



testset = SoccerDataSet(data_path=opt.data_root + '/test_cnn', map_file='test_maps',
                        transform=transforms.Compose([
                            transforms.ColorJitter(brightness=0.3,
                                                   contrast=0.4, saturation=0.4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ]))

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=5,
                                         shuffle=False,
                                         num_workers=opt.workers,
                                         drop_last=True)

opt.batch_size = opt.hist

model, loss, optimizer_seq, optimizer_both = joined_model.create_model()
model.resume_sweaty(os.path.abspath('../' + opt.sweaty_resume_str))
if opt.seq_resume:
    model.resume_seq(os.path.abspath('../' + opt.seq_resume_str))

modeleval = ModelEvaluator(model, threshold=5.0535, min_radius=2.625,
                           optim_seq=optimizer_seq,
                           optim_both=optimizer_both,
                           loss=loss)
modeleval.evaluator(trainloader, testloader, both=0)
modeleval.plot_loss()


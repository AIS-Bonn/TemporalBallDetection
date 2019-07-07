#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'March 2019'

import os

from arguments import opt
from py_utils.util_functions import dir_check
from py_dataset.prob_map import ProbMap
import py_dataset.toy_seq as toy_seq
from py_dataset import prep_real_ball_seq_ds


if opt.dataset == 'provided':
    # if opt.dataset == 'soccer':
    data_root = 'SoccerData1'
    if not os.path.exists(data_root):
        raise FileNotFoundError('Specify correctly path to the data')
    if not os.path.exists(os.path.join(data_root, 'train_cnn/train_maps')):
        prob = ProbMap(os.path.join(data_root, 'train_cnn'))
        prob.create_prob_map()
        prob.save_prob_map(data_file='train_maps')
    if not os.path.exists(os.path.join(data_root, 'test_cnn/test_maps')):
        prob = ProbMap(os.path.join(data_root, 'test_cnn'))
        prob.create_prob_map()
        prob.save_prob_map(data_file='test_maps')

    print('SoccerData1 is ready')

    # if opt.dataset == 'balls':
    # data_root = 'toy.seq'
    # if not os.path.exists(data_root):
    #     opt.balls_folder = data_root
    #     opt.n_balls = 100
    #     toy_seq.create()
    #     opt.balls_folder = 'test.' + data_root
    #     opt.n_balls = 100
    #     toy_seq.create()
    # print('Artificial balls are ready')

    # if opt.dataset == 'soccer_seq':
    data_root = 'SoccerDataSeq'
    prep_real_ball_seq_ds.create()

    print("Real ball sequences are ready")

if opt.dataset == 'new':
    try:
        assert opt.data_root != ''
    except AssertionError:
        print('Provide a root to your data')
        exit(-1)



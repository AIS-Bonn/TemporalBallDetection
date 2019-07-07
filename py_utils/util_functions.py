#!/usr/bin/env python

"""Utilities for the project"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import time
import os

from py_utils.logging_setup import logger


class Averaging(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def join_data(data1, data2, f, axis=None):
    """Simple use of numpy functions vstack and hstack even if data not a tuple

    Args:
        data1 (arr): array or None to be in front of
        data2 (arr): tuple of arrays to join to data1
        f: vstack or hstack from numpy

    Returns:
        Joined data with provided method.
    """

    if isinstance(data2, tuple):
        data2 = f(data2) if axis is None else f(data2, axis=axis)
    if data1 is not None:
        data2 = f((data1, data2)) if axis is None else f((data1, data2), axis=axis)
    return data2


def adjust_lr(optimizer, lr):
    """Decrease learning rate by 0.1 during training"""
    lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def timing(f):
    """Wrapper for functions to measure time"""
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.debug('%s took %0.3f ms ~ %0.3f min ~ %0.3f sec'
                     % (f, (time2-time1)*1000.0,
                        (time2-time1)/60.0,
                        (time2-time1)))
        return ret
    return wrap


def dir_check(path):
    """If folder given path does not exist it is created"""
    if not os.path.exists(path):
        os.mkdir(path)


def dir_clean(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
    else:
        # os.mkdir(path)
        os.makedirs(path)


class AverageLength(object):
    """Class helper for calculating average length during segmentation"""
    def __init__(self) -> None:
        self._total_length = 0
        self._nmb_of_sgmts = 0

    def add_segments(self, new_segments) -> None:
        if new_segments is None:
            return
        assert len(new_segments) > 1
        for idx, start in enumerate(new_segments[:-1]):
            end = new_segments[idx + 1]
            length = end - start
            self._total_length += length
            self._nmb_of_sgmts += 1

    def __call__(self, *args, **kwargs):
        return int(self._total_length / self._nmb_of_sgmts)


def f1(line):
    line  = line.split(' ')
    tp = int(line[1])
    fp = int(line[3])
    tn = int(line[5])
    fn = int(line[7])
    pr = tp / (tp + fp)
    rc = tp / (tp + fn)
    f1 = 2 * (pr * rc) / (pr + rc)
    print('precision: %f\nrecall: %f\nf1: %f' % (pr, rc, f1))
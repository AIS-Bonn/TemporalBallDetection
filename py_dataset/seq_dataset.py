#!/usr/bin/env python

__author__ = 'Anna Kukleva'
__date__ = 'December 2018'


from torch.utils.data import Dataset
import torch
import numpy as np
from os.path import join
import os
import re
from scipy import signal
from PIL import Image

from py_utils.util_functions import join_data
from py_utils.logging_setup import logger
from arguments import opt


def keyf(filename):
    if filename.endswith('txt'):
        return (-1, -1)
    search = re.search(r'ball(\d*)_(\d*)', filename)
    n_ball = int(search.group(1))
    n_frame = int(search.group(2))
    return (n_ball, n_frame)


class BallDataset(Dataset):
    def __init__(self, path, maxlen=20, prediction=10):
        logger.debug('create ball dataset')
        self.balls = {}
        self.balls_coord = {}
        self.balls_frames = []
        self.len = 0
        self.balls_maxframe = {}
        self.maxlen = maxlen
        if opt.seq_model == 'lstm':
            self.prediction = prediction
        if opt.seq_model == 'tcn':
            self.prediction = 1
        files = sorted(list(os.listdir(path)), key=keyf)
        for filename in files:
            if not filename.endswith('.npy'):
                continue
            search = re.search(r'ball(\d*)_(\d*).npy', filename)
            n_ball = int(search.group(1))
            n_frame = int(search.group(2))
            feature_map = np.load(join(path, filename))[..., None]
            features = self.balls.get(n_ball, None)
            features = join_data(features, feature_map, np.concatenate, axis=2)
            self.balls[n_ball] = features
        self.h = features.shape[0]
        self.w = features.shape[1]
        for ball_idx, data in self.balls.items():
            for n_frame in range(1, data.shape[-1] - self.prediction):
                self.balls_frames.append([ball_idx, n_frame])
            self.balls_coord[ball_idx] = np.loadtxt(os.path.join(path, 'ball%d.txt' % ball_idx))

    def __len__(self):
        return len(self.balls_frames)

    def __getitem__(self, idx):
        ball_idx, frame = self.balls_frames[idx]
        if frame > self.maxlen:
            seq = self.balls[ball_idx][..., frame - self.maxlen: frame]
        else:
            seq = np.zeros((self.h, self.w, self.maxlen))
            seq[..., -frame:] = self.balls[ball_idx][..., :frame]

        seq = seq.transpose(2, 0, 1)
        if opt.seq_model == 'lstm':
            next_steps = self.balls[ball_idx][..., frame: frame + self.prediction].transpose(2, 0, 1)
            return np.asarray(seq, dtype=float), np.asarray(next_steps, dtype=float)
        if opt.seq_model == 'tcn':
            seq = seq.reshape((-1, opt.hist))
            coords = self.balls_coord[ball_idx][frame]
            return np.asarray(seq, dtype=float), np.asarray(coords, dtype=float)


class RealBallDataset(Dataset):
    def __init__(self, data_path, transform=None, prediction=20, small=False):
        self.dataroot = data_path
        self.transform = transform
        self._small = small
        window = signal.gaussian(opt.window_size, std=3).reshape((-1, 1))
        self.window = np.dot(window, window.T)

        self.threshold = 0.7
        self._read_seq()

    def _read_seq(self):
        self.balls = {}
        self.ball_frames = []
        self.filenames = []
        # if self._small:
        #     folder_name = 'balls_wo_zeros'
        # else:
        folder_name = 'balls'
        for filename in os.listdir(os.path.join(opt.data_root_seq, folder_name)):
            if 'ball' in filename:
                # balls_files.append(os.path.join(opt.seq_real_balls, filename))
                # ball_filename = os.path.join(opt.seq_real_balls, filename)
                ball_idx = len(self.balls)
                with open(os.path.join(opt.data_root_seq, folder_name, filename), 'r') as f:
                    ball = []
                    for line in f:
                        line = list(map(lambda x: int(x), line.split()))
                        # ball_filename = 'imageset_%d/frame%04d.jpg' % (line[0], line[1])
                        ball_filename = 'SoccerData1/%s/' + 'frame%04d_imageset_%d.jpg' % (line[1], line[0])
                        existance = False
                        for subfolder in ['train_cnn', 'test_cnn']:
                            if os.path.exists(ball_filename % subfolder):
                                existance = True
                                break
                        ball_filename = ball_filename % subfolder
                        if not existance:
                            continue
                        ball_center = line[-2:]
                        ball.append([ball_filename, ball_center])
                        self.filenames.append(ball_filename)
                    self.balls[ball_idx] = ball
                    for i in range(opt.hist, len(ball) - opt.seq_predict):
                        self.ball_frames.append([ball_idx, i])

    def __len__(self):
        return len(self.ball_frames)

    def __getitem__(self, idx):
        ball_idx, iframe = self.ball_frames[idx]
        filenames = []
        # predict next coordinate of the ball
        frame = iframe
        _, gt_center = self.balls[ball_idx][frame]
        # collect sequence of the data back from the past
        while frame and len(filenames) != opt.hist:
            frame -= 1
            fn, _ = self.balls[ball_idx][frame]
            filenames.append(fn)

        seq = None
        for img_name in filenames:
            # img_path = os.path.join(self.dataroot, img_name)
            img_path = img_name
            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img).view(1, 3, 480, 640)
            if seq is None:
                seq = img
            else:
                seq = torch.cat((img, seq))
        if len(filenames) < opt.hist:
            raise IndexError
        if opt.seq_predict > 1:
            heatmap = np.zeros((opt.seq_predict, opt.map_size_x, opt.map_size_y), dtype=np.float32)
            for idx in range(opt.seq_predict):
                _, (x,y) = self.balls[ball_idx][frame]
                x_r = opt.window_size if x + opt.window_size < opt.map_size_x else abs(opt.map_size_x - x)
                y_r = opt.window_size if y + opt.window_size < opt.map_size_y else abs(opt.map_size_y - y)
                heatmap[idx, x:x + x_r, y:y + y_r] = self.window[:x_r, :y_r]
                frame += 1
        else:
            heatmap = np.zeros((opt.map_size_x, opt.map_size_y), dtype=np.float32)
            x,y = gt_center
            x_r = opt.window_size if x + opt.window_size < opt.map_size_x else opt.map_size_x - x
            y_r = opt.window_size if y + opt.window_size < opt.map_size_y else opt.map_size_y - y
            heatmap[x:x + x_r, y:y + y_r] = self.window[:x_r, :y_r]
        return seq, heatmap


class RealBallDatasetMulti(Dataset):
    def __init__(self, data_path, transform=None, prediction=20, small=False):
        self.dataroot = data_path
        self.transform = transform
        self._small = small
        window = signal.gaussian(opt.window_size, std=3).reshape((-1, 1))
        self.window = np.dot(window, window.T)

        self.threshold = 0.7
        if opt.dataset == 'multi':
            self._read_seq_multi()
        else:
            self._read_seq()


    def _read_seq_multi(self):
        self.balls = {}
        self.ball_frames = []
        self.filenames = []
        # if self._small:
        #     folder_name = 'balls_wo_zeros'
        # else:
        folder_name = 'balls'
        for filename in os.listdir(os.path.join(opt.data_root_seq, folder_name)):
            if 'ball' in filename:
                # balls_files.append(os.path.join(opt.seq_real_balls, filename))
                # ball_filename = os.path.join(opt.seq_real_balls, filename)
                ball_idx = len(self.balls)
                with open(os.path.join(opt.data_root_seq, folder_name, filename), 'r') as f:
                    ball = []
                    for line in f:
                        line = list(map(lambda x: int(x), line.split()))
                        # ball_filename = 'imageset_%d/frame%04d.jpg' % (line[0], line[1])
                        ball_filename = 'SoccerDataMulti/1508/frame%04d.jpg' % line[1]
                        ball_center = line[-4:]
                        ball.append([ball_filename, ball_center])
                        self.filenames.append(ball_filename)
                    self.balls[ball_idx] = ball
                    for i in range(opt.hist, len(ball) - opt.seq_predict):
                        self.ball_frames.append([ball_idx, i])

    def __len__(self):
        return len(self.ball_frames)

    def __getitem__(self, idx):
        ball_idx, iframe = self.ball_frames[idx]
        filenames = []
        # predict next coordinate of the ball
        frame = iframe
        _, gt_centers = self.balls[ball_idx][frame]
        # collect sequence of the data back from the past
        while frame and len(filenames) != opt.hist:
            frame -= 1
            fn, _ = self.balls[ball_idx][frame]
            filenames.append(fn)

        seq = None
        for img_name in filenames:
            # img_path = os.path.join(self.dataroot, img_name)
            img_path = img_name
            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img).view(1, 3, 480, 640)
            if seq is None:
                seq = img
            else:
                seq = torch.cat((img, seq))
        if len(filenames) < opt.hist:
            raise IndexError
        if opt.seq_predict > 1:
            heatmap = np.zeros((opt.seq_predict, opt.map_size_x, opt.map_size_y), dtype=np.float32)
            for idx in range(opt.seq_predict):
                _, (x,y, x2, y2) = self.balls[ball_idx][frame]
                x_r = opt.window_size if x + opt.window_size < opt.map_size_x else abs(opt.map_size_x - x)
                y_r = opt.window_size if y + opt.window_size < opt.map_size_y else abs(opt.map_size_y - y)
                heatmap[idx, x:x + x_r, y:y + y_r] = self.window[:x_r, :y_r]

                if x2 != -1:
                    x_r2 = opt.window_size if x2 + opt.window_size < opt.map_size_x else abs(opt.map_size_x - x2)
                    y_r2 = opt.window_size if y2 + opt.window_size < opt.map_size_y else abs(opt.map_size_y - y2)
                    heatmap[idx, x2:x2 + x_r2, y2:y2 + y_r2] = self.window[:x_r2, :y_r2]
                frame += 1
        else:
            heatmap = np.zeros((opt.map_size_x, opt.map_size_y), dtype=np.float32)
            x,y, x2, y2 = gt_centers
            x_r = opt.window_size if x + opt.window_size < opt.map_size_x else opt.map_size_x - x
            y_r = opt.window_size if y + opt.window_size < opt.map_size_y else opt.map_size_y - y
            heatmap[x:x + x_r, y:y + y_r] = self.window[:x_r, :y_r]

            if x2 != -1:
                x_r2 = opt.window_size if x2 + opt.window_size < opt.map_size_x else abs(opt.map_size_x - x2)
                y_r2 = opt.window_size if y2 + opt.window_size < opt.map_size_y else abs(opt.map_size_y - y2)
                heatmap[x2:x2 + x_r2, y2:y2 + y_r2] = self.window[:x_r2, :y_r2]
        return seq, heatmap


class NewDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.balls = []
        for filename in os.listdir(opt.data_root):
            if 'txt' not in filename:
                continue
            with open(os.path.join(opt.data_root, filename), 'r') as f:
                for line in f:
                    line = line.split()
                    ball_filename = line[0]
                    ball_numbers = list(map(lambda x: int(x), line[1:]))
                    ball_center = ball_numbers[:2]
                    ball_resolution = ball_numbers[2:]
                    ball_center[0] = int(ball_center[0] * 120 / ball_resolution[0])
                    ball_center[1] = int(ball_center[1] * 160 / ball_resolution[1])
                    self.balls.append([ball_filename, ball_center])

    def __len__(self):
        return len(self.balls)

    def __getitem__(self, idx):
        filename, center = self.balls[idx]
        img = Image.open(os.path.join(opt.data_root, filename))
        # gt_map = np.zeros((120, 160), dtype=float)

        map_size = (120, 160)
        if self.transform:
            img = self.transform(img)

        return img, np.zeros(map_size, dtype=float),  center, os.path.join(opt.data_root, filename)


if __name__ == '__main__':
    folder = 'toy.seq'
    dataset = BallDataset(folder)
    features, gt = dataset[38]
    logger.debug('done')



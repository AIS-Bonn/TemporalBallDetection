#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'

import os
import re
import numpy as np

from arguments import opt

save_path = os.path.join(opt.data_root_seq, 'balls')
# save_path = os.path.join(os.path.abspath(dataset_root), 'npy')
if not os.path.exists(save_path):
    os.mkdir(save_path)


class Ball:
    def __init__(self, filename):
        self.filename = filename
        self.x = []
        self.y = []
        self.frames = []
        self.valid = True

    def next_frame(self, x, y, frame):
        if len(self.x) == 0 or abs(self.frames[-1] - frame) < 6:
            self.x.append(x)
            self.y.append(y)
            self.frames.append(frame)
            return True
        if len(self.x) < 20:
            self.valid = False
        return False

    def centers(self):
        coord = []
        self.frames = list(reversed(self.frames))
        self.x = list(reversed(self.x))
        self.y = list(reversed(self.y))
        for fidx, frame in enumerate(self.frames[:-1]):
            coord.append([frame, self.x[fidx], self.y[fidx]])
            for fidx_fake in range(self.frames[fidx + 1] - frame - 1):
                coord.append([fidx_fake + frame, self.x[fidx], self.y[fidx]])
            # coord.append([[-1, -1]] * (self.frames[fidx + 1] - frame))
        coord.append([self.frames[-1], self.y[-1], self.x[-1]])
        coord = np.array(coord).reshape((-1, 3))
        coord[:, 1] = coord[:, 1] * opt.map_size_x / 480
        coord[:, 2] = coord[:, 2] * opt.map_size_y / 640
        coord = np.hstack((np.asarray([self.filename] * coord.shape[0]).reshape(-1, 1), coord))
        return coord


def create():
    balls = []
    path = 'SoccerDataSeq'
    # path = '../SoccerDataMulti'

    def keyf(line):
        try:
            n = int(re.search(r'frame(\d*).jpg', line).group(1))
        except AttributeError:
            return 0
        return n

    for filename in os.listdir(path):
        if not filename.endswith('txt'):
            continue
        with open(os.path.join(path, filename), 'r') as f:
            filename = int(filename.split('.')[0][-3:])
            ball = Ball(filename)
            lines = f.read().split('\n')
            lines = sorted(lines, key=keyf)
            # for line_idx, line in enumerate(f):
            for line_idx, line in enumerate(lines):
                if not line.startswith('label::ball'):
                    continue
                line = line.split('|')
                frame = int(re.search(r'frame(\d*).jpg', line[1]).group(1))
                # y,x = list(map(lambda x: int(x.split('.')[0]), line[-4:-2]))
                y,x = list(map(lambda x: int(x), line[-8:-6]))
                if not ball.next_frame(x, y, frame) or line_idx == len(lines) - 1:
                    if ball.valid:
                        balls.append(ball)
                    ball = Ball(filename)

    for ball_idx, ball in enumerate(balls):
        np.savetxt(os.path.join(save_path, 'ball%d.txt' % ball_idx), ball.centers(), fmt='%d')


def reparse_multi():
    balls = []
    # path = 'SoccerDataSeq'
    path = '../SoccerDataMulti'

    for filename in os.listdir(os.path.join(path, 'balls')):
        new_file = []
        with open(os.path.join(path, 'balls', filename), 'r') as f:
            lines = f.read().split('\n')[:-1]
            pr_frame = 0
            for line in lines:
                line = list(map(lambda x: int(x), line.split()))
                frame = line[1]
                if frame == pr_frame:
                    newline = new_file[-1] + [line[-2], line[-1]]
                    new_file[-1] = newline
                else:
                    if pr_frame and len(new_file[-1]) == 4:
                        new_file[-1] = new_file[-1] + [-1, -1]
                    new_file.append(line)
                    pr_frame = frame
        if len(new_file[-1]) == 4:
            new_file[-1] = new_file[-1] + [-1, -1]
        new_file = np.array(new_file, dtype=int)
        np.savetxt(os.path.join(save_path, filename), new_file, fmt='%d')



if __name__ == '__main__':
    create()
    reparse_multi()

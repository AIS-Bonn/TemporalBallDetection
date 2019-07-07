#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

from py_utils.util_functions import dir_clean
from arguments import opt


class Ball:
    def __init__(self, x, y, map_size, window_size):
        # left top corner
        self.x = x
        self.y = y
        self.map_size_x = map_size[0]
        self.map_size_y = map_size[1]
        self.window_size = window_size
        self.sign_x = 1
        self.sign_y = 1
        self.sigma = 0
        self.window = np.zeros((window_size, window_size))
        self.coord = [[x, y]]

    def init_sigma(self, sigma):
        # self.sigma = sigma
        self.sigma = 4
        window = signal.gaussian(self.window_size, std=sigma).reshape((-1, 1))
        self.window = np.dot(window, window.T)

    def set_direction(self, sign_x, sign_y):
        self.sign_x = sign_x
        self.sign_y = sign_y

    def update(self, dx, dy):
        dx, dy = int(dx), int(dy)
        if self.x + self.sign_x * dx < 0 or \
            self.x + self.sign_x * dx > self.map_size_x - self.window_size:
            self.sign_x = self.sign_x * -1
        if self.y + self.sign_y * dy < 0 or \
            self.y + self.sign_y * dy > self.map_size_y - self.window_size:
            self.sign_y = self.sign_y * -1

        self.x += self.sign_x * dx
        self.y += self.sign_y * dy
        self.coord.append([self.x, self.y])


def create():
    # empty folder to save the sequence
    folder = opt.balls_folder
    npy_folder = os.path.join(folder, 'npy')
    dir_clean(npy_folder)
    png_folder = os.path.join(folder, 'png')
    dir_clean(png_folder)


    map_size_x = opt.map_size_x
    map_size_y = opt.map_size_y
    window_size = 15
    n_balls = 20
    min_sigma = 2
    max_sigma = 4
    max_shift = 500
    min_shift = 50
    max_move_steps = 60
    min_move_steps = 30

    # create heat map for the imaginary ball
    # window = signal.gaussian(window_size, std=sigma).reshape((-1, 1))
    # gauss_window = np.dot(window, window.T)

    balls = []

    # init balls
    for ball_idx in range(n_balls):
        x = np.random.randint(0, map_size_x - window_size, 1)[0]
        y = np.random.randint(0, map_size_y - window_size, 1)[0]
        ball = Ball(x, y, [map_size_x, map_size_y], window_size)
        ball.init_sigma(np.random.randint(min_sigma, max_sigma, 1)[0])
        balls.append(ball)

    # mimic movements
    # seq_len = 1
    # for seq_idx in range(seq_len):
    #     heatmap = np.zeros((map_size, map_size), dtype=np.float32)
    for ball_idx, ball in enumerate(balls):
        dx, dy = np.random.randint(min_shift, max_shift, 2)
        move_steps = np.random.randint(min_move_steps, max_move_steps, 1)[0]
        ball.set_direction(sign_x=np.random.choice([-1, 1]),
                           sign_y=np.random.choice([-1, 1]))
        ddx = max(int(dx / move_steps), 1)
        ddy = max(int(dy / move_steps), 1)
        for dd_idx in range(move_steps):
            heatmap = np.zeros((map_size_x, map_size_y), dtype=np.float32)
            ball.update(ddx, ddy)
            plt.clf()
            heatmap[ball.x:ball.x + window_size, ball.y:ball.y + window_size] = ball.window
            pltimg = plt.imshow(heatmap)
            plt.savefig(os.path.join(png_folder, 'ball%d_%d.png' %
                                     (ball_idx, dd_idx)))
            np.save(os.path.join(npy_folder, 'ball%d_%d.npy' %
                                 (ball_idx, dd_idx)), heatmap)

        for ball_idx, ball in enumerate(balls):
            np.savetxt(os.path.join(npy_folder, 'ball%d.txt' % ball_idx),
                       np.array(ball.coord).reshape((-1, 2)), fmt='%d')


if '__name__' == '__main__':
    create()
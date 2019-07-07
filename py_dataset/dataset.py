from arguments import opt
import os
import pdb
import xmltodict as xd
import numpy as np
from torchvision import transforms
from scipy.stats import multivariate_normal
from PIL import Image
import torch
import h5py
from tqdm import tqdm

class SoccerDataSet:
    '''
    DataSet reader: Reader to get images and probability map from a folder
    '''
    def __init__(self, data_path, map_file, transform=None):
        self.dataroot = data_path
        self.map_file = map_file
        self.transform = transform

        with h5py.File(self.dataroot + '/' + self.map_file,'r') as hf:
            targets = hf['prob_maps'].value
            targets = np.array(targets).astype('float32')
            self.filenames = list(hf['filenames'].value)
            self.min_radius = hf['min_radius'].value
            centers = list(hf['centers'].value)

        if opt.dataset == 'multi':
            centers1 = {}
            centers2 = {}
            for c_idx, c in enumerate(centers):
                if c_idx % 2 == 1:
                    centers1[c_idx // 2] = c
                else:
                    centers2[c_idx // 2] = c

        self.threshold = 0.7*targets.max()
        self.images, self.targets, self.centers = [], [], []
        self.filenames = [filename.decode('utf-8') for filename in self.filenames]

        for filename in tqdm(os.listdir(self.dataroot)):
            name = filename[:-4]
            if filename.endswith('.jpg'):
                self.images.append(filename)
                if name in self.filenames:
                    idx = self.filenames.index(name)
                    self.targets.append(targets[idx])
                    if opt.dataset == 'multi':
                        centers = [centers1[idx].astype('float32')]
                        centers.append(centers2[idx].astype('float32'))
                        self.centers.append(centers)
                    else:
                        self.centers.append(centers[idx].astype('float32'))
                else:
                    self.targets.append(np.zeros([120, 160], dtype='float32'))
                    self.centers.append(np.array([-1, -1], dtype='float32'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.dataroot, img_name)
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        prob_ = self.targets[idx]
        coord_ = self.centers[idx]
        return img, prob_, coord_, img_name

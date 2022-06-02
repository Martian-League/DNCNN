# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020

@author: lx
"""

import torch.utils.data as data
import random
import torch
import numpy as np


class ImageDataset(data.Dataset):  # 继承

    def __init__(self):

        self.num = 1
        #root_dir = "/home/gwb/PPP/data/trainset/"
        #filename = "/home/gwb/Train_prior/Data/dataset2.npy"

        filename = "/home/gwb/PPP/data/others/test.npy"
        '''for i in range(self.num):
            filename = root_dir + 'testdata' + str(i) + '.npy'
            self.data.append(np.load(filename))'''
        self.datalabel=np.load(filename)
        print(self.datalabel.shape)
        print("read meta done")

    def __len__(self):  # 这儿应该是自定义函数，和系统自带的len不同
        return self.num

    def __getitem__(self, idx):

        #img = self.datalabel[idx,:,:]
        label = self.datalabel
        MAX = np.max(label)
        MIN = np.min(label)
        label = (label-MIN) / (MAX-MIN)
        label = label.reshape(1, 500, 600)
        label = torch.FloatTensor(label)
        return label,label


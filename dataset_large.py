# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020

@author: lx
"""

import torch.utils.data as data
import random
import os
import torch
import numpy as np
import random
def read_data(root_dir,Normlize):
    item_paration=[]
    for i in range(2):
        filename = root_dir + "testdata" + str(i) + '.npy'
        y = np.load(filename)  # (1000,128,128,1)
        y = y.astype(np.float32)
        if Normlize:
            y = y/np.max(y)
        item_paration.append(y)
    return item_paration

def Normlize(img,label):
    MAX = np.max(img)
    MIN = np.min(img)
    img = (img - MIN) / (MAX - MIN)
    label = label#(label - MIN) / (MAX - MIN)
    return img, label

class ImageDataset(data.Dataset):  # 继承

    def __init__(self):

        self.num = 2
        self.data = []
        self.label = []
        root_dir_data = "/home/gwb/PPP/data/trainset/Testdata/"
        #root_dir_data = "/home/gwb/Gittest/result/images/Noise_Result/"
        #root_dir_label='/home/gwb/Gittest/result/images/Denosing_Result/'
        #filename = "/home/gwb/Train_prior/Data/dataset2.npy"

        self.data = read_data(root_dir_data,True)
        self.label = self.data

        print("read meta done")

    def __len__(self):  # 这儿应该是自定义函数，和系统自带的len不同
        return self.num

    def __getitem__(self, idx):

        img = self.data[idx]
        label = self.label[idx]
        row = img.shape[0]
        col = img.shape[1]

        img, label = Normlize(img,label)

        img = img.reshape(1, row, col)
        label = label.reshape(1, row, col)
        img = torch.FloatTensor(img)
        label = torch.FloatTensor(label)
        #print(torch.max(label))
        return img, label


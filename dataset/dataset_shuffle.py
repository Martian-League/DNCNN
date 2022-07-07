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
    for i in range(10):
        filename = root_dir + "sesmic_" + str(i) + 'th_result_denosing.npy'
        y = np.load(filename)  # (1000,128,128,1)
        y = y.astype(np.float32)
        item_paration.append(y[2000:2500, 0:600])
        #数据切割
    E_shuffle = Randomlize(item_paration[0])
    np.save("/home/gwb/DNCNN/result/E_shuffle.npy", E_shuffle)
    item_paration = np.array(item_paration)
    paration_shuffle = np.dot(item_paration, E_shuffle)

    return paration_shuffle

def Randomlize(Sample):
    N = Sample.shape[1]
    E_martrix = np.eye(N)
    E_shuffle = np.eye(N)
    random_sequence = random.sample(range(0,N),N)
    i = 0
    for item in random_sequence:
        E_shuffle[:,i]=E_martrix[:,item]
        i=i+1
    #seismic_shuffle = np.dot(Sample,E_shuffle)
    return  E_shuffle

def Normlize(img,label):
    MAX = np.max(img)
    MIN = np.min(img)
    img = (img - MIN) / (MAX - MIN)
    label = label#(label - MIN) / (MAX - MIN)
    return img, label

class ImageDataset(data.Dataset):  # 继承

    def __init__(self):

        self.num = 205
        self.data = []
        self.label = []
        #root_dir_data = "/home/gwb/PPP/data/trainset/Testdata/"
        #root_dir_data = "/home/gwb/Gittest/result/images/Noise_Result/"
        root_dir_label='/home/gwb/Gittest/result/images/Denosing_Result/'
        #filename = "/home/gwb/Train_prior/Data/dataset2.npy"

        self.data = read_data(root_dir_label,True)
        self.label = self.data

        print("read meta done")

    def __len__(self):  # 这儿应该是自定义函数，和系统自带的len不同
        return self.num

    def __getitem__(self, idx):

        img = self.data[idx,:,:]
        label = self.label[idx,:,:]
        row = img.shape[0]
        col = img.shape[1]
        #并未对label做归一化，主要是利用label来恢复到原先地大小
        img, label = Normlize(img,label)

        img = img.reshape(1, row, col)
        label = label.reshape(1, row, col)
        img = torch.FloatTensor(img)
        label = torch.FloatTensor(label)
        #print(torch.max(label))
        return img, label


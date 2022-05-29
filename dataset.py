# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020
此文件中归一化是单个数据上进行的，而不是全局
@author: lx
"""

import torch.utils.data as data
import random
import os
import torch
import numpy as np

def read_data(root_dir,Normlize):

    test_index = [6, 51, 61, 41, 137, 18, 13, 188, 95, 73, 126, 203, 35, 26, 71]
    #test_index = [150,160,180,190,200]
    metas=[]
    data=[]
    i=0
    for filename in os.listdir(root_dir):  # 展开成一个新的列表
        metas.append(filename)
    for file  in metas:
        filename = root_dir + "/" + file
        if i  in test_index:
            i = i + 1
            continue
        A = np.load(filename)
        A = np.pad(A, ((0, 0), (1152-A.shape[1],0)), 'symmetric')

        data.append(A.astype(np.float32))  # ,此时读取来的数据com还是1个200*3001的一长串，不能二维显示，需要转换形状
        i = i + 1
    #数据切割
    Xlabel = []
    Nrow = 6
    Ncol = 2
    for k in range(len(data)):
        MAX = np.max(data[k])
        for i in range(Nrow):
            for j in range(Ncol):
            #Xlabel.append(data[k][250 + i * 700:250 + i * 700 + 1152, :])
                item = data[k][500 + i * 300:500 + i * 300 + 512, j*100:j*100+512]
                if (Normlize == True):
                    item = item/MAX
                Xlabel.append(item)
    return Xlabel

def Normlize(img,label):
    MAX = np.max(img)
    MIN = np.min(img)
    img = (img - MIN) / (MAX - MIN)
    label = (label - MIN) / (MAX - MIN)
    return img,label

class ImageDataset(data.Dataset):  # 继承

    def __init__(self):

        self.num = 50
        self.data = []
        self.label = []
        root_dir_data = "/home/gwb/PPP/data/trainset/Testdata/"
        #root_dir_data = "/home/gwb/Gittest/result/images/Noise_Result/"
        root_dir_label='/home/gwb/Gittest/result/images/Denosing_Result/'
        #filename = "/home/gwb/Train_prior/Data/dataset2.npy"

        self.data = read_data(root_dir_data,True)
        self.label = read_data(root_dir_label,False)

        print("read meta done")

    def __len__(self):  # 这儿应该是自定义函数，和系统自带的len不同
        return self.num

    def __getitem__(self, idx):

        img = self.data[idx]
        label = self.label[idx]
        row = img.shape[0]
        col = img.shape[1]
        #print(img.shape)
        #exit()
        img, label = Normlize(img,label)
        #label = Normlize(label)
        print(np.max(label))
        print(np.min(label))
        print(np.max(img))
        print(np.min(img))
        img = img.reshape(1, row, col)
        label = label.reshape(1, row, col)
        img = torch.FloatTensor(img)
        label = torch.FloatTensor(label)
        return img, label


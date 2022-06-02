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
def read_data(root_dir,Normlize,name):

    #test_index = [6, 51, 61, 41, 137, 18, 13, 188, 95, 73, 126, 203, 35, 26, 71]
    test_index = [105,146,155,137,39,116,111,85,186,166,29,99,131,123,164]
    metas=[]
    data=[]
    i=0
    for filename in os.listdir(root_dir):  # 展开成一个新的列表
        metas.append(filename)
    #print(metas)
    #for file in metas:
    for i in test_index:
        #filename = root_dir + "/" + file
        if name=='noisy':
            filename=root_dir+"testdata" + str(i)+'.npy'
        else:
            filename = root_dir + "sesmic_" + str(i) + 'th_result_denosing.npy'
        '''if i not in test_index:
            i = i + 1
            continue'''
        A = np.load(filename)
        A = np.pad(A, ((0, 3456-A.shape[0]), (0,1152-A.shape[1])), 'symmetric')
        data.append(A.astype(np.float32))  # ,此时读取来的数据com还是1个200*3001的一长串，不能二维显示，需要转换形状
        #i = i + 1
    #数据切割
    Xlabel = []
    for k in range(len(data)):
        MAX = np.max(data[k])
        item = data[k]
        if (Normlize == True):
            item = item/MAX
        Xlabel.append(item)
    return Xlabel

def Normlize(img,label):
    MAX = np.max(img)
    MIN = np.min(img)
    img = (img - MIN) / (MAX - MIN)
    label = (label - MIN) / (MAX - MIN)
    return img, label

class ImageDataset(data.Dataset):  # 继承

    def __init__(self):

        self.num = 2
        self.data = []
        self.label = []
        root_dir_data = "/home/gwb/PPP/data/trainset/Testdata/"
        #root_dir_data = "/home/gwb/Gittest/result/images/Noise_Result/"
        root_dir_label='/home/gwb/Gittest/result/images/Denosing_Result/'
        #filename = "/home/gwb/Train_prior/Data/dataset2.npy"

        self.data = read_data(root_dir_data,True,'noisy')
        self.label = read_data(root_dir_label,False,'label')

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
        img,label = Normlize(img,label)

        img = img.reshape(1, row, col)
        label = label.reshape(1, row, col)
        img = torch.FloatTensor(img)
        label = torch.FloatTensor(label)
        return img, label


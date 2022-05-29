# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020
这是自己测试是不是由于归一化的原因导致出现问题的程序
@author: lx
"""

import torch.utils.data as data
import random
import os
import torch
import numpy as np
import segyio
def read_data(root_dir,Normlize):

    metas=[]
    metaslabel=[]
    sample=[]
    label=[]
    for filename in os.listdir(root_dir + "sample"):  # 展开成一个新的列表
        metas.append(filename)
    for filename in os.listdir(root_dir + "label"):  # 展开成一个新的列表
        metaslabel.append(filename)
    print(len(metas))
    print(len(metaslabel))
    train_filename = random.sample(metas, 10)
    for file in train_filename:
        filename_sample = root_dir + "sample/" + file
        filename_label = root_dir + "label/clean_" + file
        sample_t = ReadSegyData(filename_sample)
        #print("训练集OK")
        label_t = ReadSegyData(filename_label)
        #sample_t = np.pad(sample_t, ((0, 0), (1152-sample_t.shape[1],0)), 'symmetric')
        #label_t = np.pad(label_t, ((0, 0), (1152 - label_t.shape[1], 0)), 'symmetric')
        if Normlize == True:
            #print(np.max(sample_t))
            #print(np.max(label_t))
            sample_t  = sample_t / np.max(sample_t)
            label_t = label_t / np.max(label_t)
        sample.append(sample_t.astype(np.float32))  # ,此时读取来的数据com还是1个200*3001的一长串，不能二维显示，需要转换形状
        label.append(label_t.astype(np.float32))
    #sample_torch = [torch.FloatTensor(item) for item in sample]
    #label_torch = [torch.FloatTensor(item) for item in label]
    return sample, label

def ReadSegyData(filename):
    with segyio.open(filename,'r',ignore_geometry=True) as f:
        f.mmap()
        data2D = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data2D

def Normlize(img,label):
    MAX = np.max(img)
    MIN = np.min(img)
    eps = 10e-8;
    img = (img - MIN) / (MAX - MIN + eps)
    label = (label - MIN) / (MAX - MIN +eps)
    return img,label

def Split_data( data ):
    Xlabel = []
    height, width = data.shape
    row_begin, win_size, step_row, step_col = 500, 512, 300, 200
    num_row = (height-row_begin-win_size)//step_row+1
    num_col = (width-win_size)//step_col+1

    for i in range(num_row):
        for j in range(num_col-1):
            item = data[500 + i * 300:500 + i * 300 + 512, j*200:j*200+512]
            Xlabel.append(item)
    Xlabel = np.array(Xlabel)
    #print(Xlabel.shape)
    return Xlabel

class SegyDataset(data.Dataset):  # 继承

    def __init__(self,path):

        self.num = 50
        self.data = []
        self.label = []
        root_dir = path

        data, label = read_data(root_dir, False)
        #self.label = read_data(root_dir_label, False)
        for i in range(len(data)):
            if (len(Split_data(data[i]))>0):
                self.data.append(Split_data(data[i]))
                self.label.append(Split_data(label[i]))
        self.data = np.concatenate(self.data, axis=0)
        self.label = np.concatenate(self.label, axis=0)

        print("read meta done")


    def __len__(self):  # 这儿应该是自定义函数，和系统自带的len不同
        return self.num

    def __getitem__(self, idx):

        img = self.data[idx]
        label = self.label[idx]
        row = img.shape[0]
        col = img.shape[1]
        img, label = Normlize(img, label)

        img = img.reshape(1, row, col)
        label = label.reshape(1, row, col)
        img = torch.FloatTensor(img)
        label = torch.FloatTensor(label)
        return img, label


# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020

@author: lx
"""
from math import floor, ceil

import torch.utils.data as data
import random
import os
import torch
import numpy as np
import segyio
import pywt

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
    train_filename = random.sample(metas, 25)
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
    sample_torch = [torch.FloatTensor(item) for item in sample]
    label_torch = [torch.FloatTensor(item) for item in label]
    return sample_torch, label_torch

def ReadSegyData(filename):
    with segyio.open(filename,'r',ignore_geometry=True) as f:
        f.mmap()
        data2D = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data2D


def Split_data( data ):
    Xlabel = []
    n_sample, height, width = data.shape
    #row_begin, win_size, step_row, step_col = 500, 60, 20, 20
    row_begin, win_size, step_row, step_col = 500, 512, 300, 200
    num_row = (height-row_begin-win_size)//step_row+1
    num_col = (width-win_size)//step_col+1
    for k in range(n_sample):
        for i in range(num_row):
            for j in range(num_col-1):
                item = data[k, 500 + i * 300:500 + i * 300 + 512, j*200:j*200+512]
                '''if np.max(item)-np.min(item) < 0.1:
                    print("成功避免")
                    continue'''
                Xlabel.append(item)
    Xlabel = np.array(Xlabel)
    #print(Xlabel.shape)
    return Xlabel

def Split_data_c( data ):
    Xlabel = []
    n_sample, height, width = data.shape
    row_begin, win_size, step_row, step_col = 500, 60, 20, 20
    #row_begin, win_size, step_row, step_col = 500, 512, 300, 200
    num_row = (height-row_begin-win_size)//step_row+1
    num_col = (width-win_size)//step_col+1
    #num_col = [i // 10 + 1 for i in range(30)]
    for k in range(n_sample):
        for j in range(num_col):
            num_row = num_row -1
            for i in range(num_row-1):
                item = data[k, row_begin + i * step_row:row_begin + i * step_row + win_size, j*step_col:j*step_col+win_size]
                '''if np.max(item) == np.mean(item):
                    print("成功避免")
                    continue'''
                Xlabel.append(item)
    Xlabel = np.array(Xlabel)
    #print(Xlabel.shape)
    return Xlabel

def Split_data_random( data,label ):
    Xlabel = []
    Xdata = []
    n_sample, rows, cols = data.shape

    if cols > 900: loop = 2000
    elif cols > 500: loop = 1000
    else: loop = 500

    p_size = 60

    left = floor(p_size / 2);
    right = ceil(p_size / 2);
    x_axis = np.random.randint(floor(p_size / 2), cols - right, loop);
    y_axis = np.random.randint(floor(p_size / 2), rows - right, loop);
    #temp = zeros(length_p, loop);
    for k in range(n_sample):
        for i in range(loop):
            temp1 = data[k, y_axis[i] - left:y_axis[i] + right-1, x_axis[i] - left:x_axis[i] + right-1]
            temp2 = label[k, y_axis[i] - left:y_axis[i] + right - 1, x_axis[i] - left:x_axis[i] + right - 1]
            if np.max(temp1) == np.min(temp1):
                continue
            Xdata.append(temp1)
            Xlabel.append(temp2)
    Xlabel = np.array(Xlabel)
    Xdata = np.array(Xdata)
    return Xdata, Xlabel

def Normlize(img,label):
    B, H, W = img.shape
    img = img.contiguous().view(B, -1)
    label = label.contiguous().view(B, -1)

    label -= img.min(1, keepdim=True)[0]
    img -= img.min(1, keepdim=True)[0]

    label /= img.max(1, keepdim=True)[0]
    img /= img.max(1, keepdim=True)[0]

    img = img.view(B,  H, W)
    label = label.view(B,  H, W)
    return img, label

def Normlize_c(img,label):
    #排除野值干扰
    B, H, W = img.shape
    img = img.contiguous().view(B, -1)
    label = label.contiguous().view(B, -1)

    img_sort, idx = torch.sort(img)
    Min = torch.quantile(img_sort, 0.0001, dim=1, keepdim=True) * 1.2
    Max = torch.quantile(img_sort, 0.9999, dim=1, keepdim=True) * 1.2

    label -= Min
    img -= Min

    label /= (Max-Min)
    img /= (Max-Min)

    img = img.view(B,  H, W)
    label = label.view(B,  H, W)
    return img, label

def Wavelet(img):
    coeffs2 = pywt.dwt2(img, 'db4')
    LL, (LH, HL, HH) = coeffs2
    #print("经过小波变换")
    combin = np.array([LL, LH, HL, HH])

    return combin

class SegyDataset(data.Dataset):  # 继承

    def __init__(self,path):

        self.num = 20
        self.data = []
        self.label = []
        root_dir = path

        data, label = read_data(root_dir, False)
        #self.label = read_data(root_dir_label, False)
        for i in range(len(data)):
            data[i], label[i] = Normlize_c(data[i].unsqueeze(0), label[i].unsqueeze(0))
            #label[i] = (label[i] / torch.max(label[i])).unsqueeze(0)
            #data[i] = (data[i]/torch.max(data[i])).unsqueeze(0)

            #print(self.data[i].shape)
            if (len(Split_data(data[i].numpy()))>0):
                d_numpy = data[i].numpy()
                lb_numpy = label[i].numpy()
                #print(d_numpy.shape)
                temp1, temp2 = Split_data_random(d_numpy[:, : , :], lb_numpy[: ,: ,:])
                self.data.append(temp1)
                self.label.append(temp2)
        self.data = np.concatenate(self.data, axis=0)
        self.label = np.concatenate(self.label, axis=0)
        self.num = self.data.shape[0]
        #self.num = 10;
        print("read meta done")

    def __len__(self):  # 这儿应该是自定义函数，和系统自带的len不同
        return self.num

    def __getitem__(self, idx):

        #print(self.data[0].shape)
        random_idx = idx
        while np.max(self.data[random_idx, :, :]) == np.min(self.data[random_idx, :, :]):
            random_idx = np.random.randint(0, high=self.num)
            print("漏网之鱼")
            #将数据的最大值和最小值输出偶然发现，空白时两者一样大
        '''img = Wavelet(self.data[random_idx, :, :])
        label = Wavelet(self.label[random_idx, :, :])
        row = img.shape[1]
        col = img.shape[2]

        img = img.reshape(4, row, col)
        label = label.reshape(4, row, col)'''

        img = self.data[random_idx, :, :]
        label = self.label[random_idx, :, :]
        row = img.shape[0]
        col = img.shape[1]

        img = img.reshape(1, row, col)
        label = label.reshape(1, row, col)
        img = torch.FloatTensor(img)
        label = torch.FloatTensor(label)
        #data, label = Normlize(img, label)

        return img, label


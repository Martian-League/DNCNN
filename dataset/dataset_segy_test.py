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
import segyio
import pywt

def read_data(root_dir,Normlize):

    metasdata=[]
    metaslabel=[]
    sample=[]
    label=[]
    for filename in os.listdir(root_dir + "sample"):  # 展开成一个新的列表
        metasdata.append(filename)
    for filename in os.listdir(root_dir + "label"):  # 展开成一个新的列表
        metaslabel.append(filename)
    #train_filename = random.sample(metas, 5)

    metaslabel.sort(key=lambda x:int(x.split('_')[-2]))
    metasdata.sort(key=lambda x:int(x.split('_')[-2]))

    metas = list(zip(metasdata,metaslabel))

    for filename_sample, filename_label in metas:
        #filename_sample = root_dir + "sample/" + file
        #filename_label = root_dir + "label/clean_" + file
        sample_t = ReadSegyData(root_dir + "sample/" + filename_sample)
        #print("训练集OK")
        label_t = ReadSegyData(root_dir + "label/" + filename_label)
        #print(file)
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
    return sample_torch, label_torch, metasdata

def ReadSegyData(filename):
    with segyio.open(filename,'r',ignore_geometry=True) as f:
        f.mmap()
        data2D = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    '''
    dst_root = "/home/gwb/DNCNN/result/segy_dncnn/sgy_root/"
    dstpath = dst_root + os.path.basename(filename)
    segyio.tool.from_array2D(dstpath, data2D.T)
    '''
    return data2D


def Split_data( data ):
    Xlabel = []
    n_sample, height, width = data.shape
    row_begin, win_size, step_row, step_col = 500, 512, 300, 200
    num_row = (height-row_begin-win_size)//step_row+1
    num_col = (width-win_size)//step_col+1
    for k in range(n_sample):
        for i in range(num_row):
            for j in range(num_col-1):
                item = data[k, 500 + i * 300:500 + i * 300 + 512, j*200:j*200+512]
                Xlabel.append(item)
    Xlabel = np.array(Xlabel)
    #print(Xlabel.shape)
    return Xlabel

def Normlize(img,label):
    B, H, W = img.shape
    img = img.contiguous().view(B, -1)#contiguous转换为内存中的连续存储，不然不能实验view
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


class SegyDataset(data.Dataset):  # 继承

    def __init__(self,path):

        self.num = 5
        self.data = []
        self.label = []
        self.filename= []
        self.rawdata = []
        root_dir = path

        data, label ,self.filename = read_data(root_dir, False)

        for i in range(len(data)):
            self.rawdata.append(data[i].unsqueeze(0))
            data[i], label[i] = Normlize(data[i].unsqueeze(0), label[i].unsqueeze(0))
            self.data.append(data[i])
            self.label.append(label[i])

        #self.data = np.concatenate(self.data, axis=0)
        #self.label = np.concatenate(self.label, axis=0)
        self.num = len(self.data)
        print("read meta done")

    def __len__(self):  # 这儿应该是自定义函数，和系统自带的len不同
        return self.num

    def __getitem__(self, idx):

        img = self.data[idx]#[500:1012, 0:512]
        label = self.label[idx]#[500:1012, 0:512]
        raw_data = self.rawdata[idx]
        #img = img[:,500:1012, 0:512]
        #label = label[:,500:1012, 0:512]
        #row = img.shape[1]
        #col = img.shape[2]
        #print(img.shape)
        img_name = self.filename[idx]
        #img = img.reshape(1, row, col)
        #label = label.reshape(1, row, col)
        #img = torch.FloatTensor(img)
        #label = torch.FloatTensor(label)
        return img, label, raw_data, img_name


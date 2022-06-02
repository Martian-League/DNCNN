# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020
这是完成自己矩阵方式处理overlap的程序
@author: lx
"""

import torch.utils.data as data
import random
import os
import torch
import numpy as np
def read_data(root_dir,Normlize,name):

    test_index = [105,146,155,137,39,116,111,85,186,166,29,99,131,123,164]
    testdata_set=[]
    for i in test_index:
        if name == 'noisy':
            filename = root_dir+"testdata" + str(i)+'.npy'
        else:
            filename = root_dir + "sesmic_" + str(i) + 'th_result_denosing.npy'
        testdata = np.load(filename)
        if (Normlize == True):
            MAX = np.max(testdata)
            testdata = testdata / MAX
        data.append(testdata.astype(np.float32))  # ,此时读取来的数据com还是1个200*3001的一长串，不能二维显示，需要转换形状

    return testdata_set

def Normlize(img,label):
    MAX = np.max(img)
    MIN = np.min(img)
    img = (img - MIN) / (MAX - MIN)
    label = (label - MIN) / (MAX - MIN)
    return img, label


def Patch(img, Windows_size, Overlap_size):

    Step_size = Windows_size - Overlap_size
    Height, Width = img.shape
    Width_new = ((Width-Windows_size)//Step_size+1)*Step_size+Windows_size
    Height_new = ((Height-Windows_size)// Step_size + 1) * Step_size+Windows_size
    img = np.pad(img, ((0, Height_new - img.shape[0]), (0, Width_new - img.shape[1])), 'symmetric')
    img_lapcol = Overlap(img,Windows_size,Overlap_size)
    img_laprow = Overlap(img_lapcol.T,Windows_size,Overlap_size)
    return img_laprow

def Overlap(img,Windows_size,Overlap_size):
    Step_size = Windows_size - Overlap_size
    Height, Width = img.shape
    #print(img.shape)
    num_windows = ((Width - Windows_size) // Step_size + 1)
    N_col = Windows_size * num_windows
    Matrix_extend = np.zeros((Width,N_col))
    Matrix = np.eye(Width)
    for i in range(num_windows-1,0,-1):
        Matrix_extend[:,i*Windows_size:(i+1)*Windows_size] = Matrix[:,i*Step_size:i*Step_size+Windows_size]
    img_lap = np.dot(img, Matrix_extend)
    return img_lap
    '''这是从前往后做的方法
    for i in range(1, num_windows):
    B = Matrix[:, i*Step_size+(i-1)*Overlap_size-1:i*Step_size+(i)*Overlap_size-1]
    for j in range(Overlap_size):
        Matrix=np.insert(Matrix, i*Windows_size-1, B[:,Overlap_size-1-j], axis=1)'''
class ImageDataset(data.Dataset):  # 继承

    def __init__(self):

        self.num = 1
        self.data = []
        self.label = []
        windows_size = windows_size
        overlap_size = overlap_size
        root_dir_data = "/home/gwb/PPP/data/trainset/Testdata/"
        #root_dir_data = "/home/gwb/Gittest/result/images/Noise_Result/"
        root_dir_label = '/home/gwb/Gittest/result/images/Denosing_Result/'
        #filename = "/home/gwb/Train_prior/Data/dataset2.npy"

        self.data = read_data(root_dir_data, True, 'noisy')
        #self.data = Patch(self.data, windows_size, overlap_size)
        self.label = read_data(root_dir_label, False, 'label')
        #self.data = Patch(self.label, windows_size, overlap_size)

        print("read meta done")

    def __len__(self):  # 这儿应该是自定义函数，和系统自带的len不同
        return self.num

    def __getitem__(self, idx):

        img = self.data[idx, :, :]
        label = self.label[idx, :, :]
        row = img.shape[0]
        col = img.shape[1]

        img, label = Normlize(img,label)
        img = img.reshape(1, row, col)
        label = label.reshape(1, row, col)
        img = torch.FloatTensor(img)
        label = torch.FloatTensor(label)
        return img, label


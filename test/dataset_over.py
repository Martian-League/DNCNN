# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020
这是完成自己矩阵方式处理overlap的程序
@author: lx
"""

import torch.utils.data as data
import random
import os
import copy
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
    print(img.max())
    num_windows = ((Width - Windows_size) // Step_size + 1)
    N_col = Windows_size * num_windows
    Matrix_extend = np.zeros((Width,N_col))
    Matrix = np.eye(Width)
    for i in range(num_windows-1,-1,-1):
        Matrix_extend[:,i*Windows_size:(i+1)*Windows_size] = Matrix[:,i*Step_size:i*Step_size+Windows_size]
    img_lap = np.dot(img, Matrix_extend)
    print(img_lap.max())
    return img_lap

def combine(patches, patch_size, lap):
    patches_row = combine_son(patches, patch_size, lap)
    patches_col = combine_son(patches_row, patch_size, lap)
    return patches_col

def combine_son(patches, patch_size, lap):
    n_w = patches.shape[1]
    num_lap = int(np.ceil(n_w // patch_size - 1))

    sigmoid = lambda x: 1 / (1 + np.exp(-x + 1e-15))
    weight1 = np.diag(sigmoid(np.linspace(-3, 3, lap)))

    weight2 = np.eye(np.shape(weight1)[0]) - weight1
    weight = np.concatenate([weight1, weight2], axis=0)
    bound_loc = [patch_size * (item + 1) for item in list(range(num_lap))]

    result = copy.deepcopy(patches)
    print(bound_loc)
    for idx in bound_loc:
        result[:, idx - lap:idx] = np.dot(patches[:, idx - lap:idx + lap], weight)
    count = 0
    for idx in reversed(bound_loc):
        count = count + 1
        result = np.delete(result, list(range(idx, idx + lap)), 1)
    return result


class ImageDataset(data.Dataset):  # 继承

    def __init__(self):

        self.num = 1
        self.data = []
        self.label = []
        windows_size = 200
        overlap_size = 40
        root_dir_data = "/home/gwb/PPP/data/trainset/Testdata/"
        #root_dir_data = "/home/gwb/Gittest/result/images/Noise_Result/"
        root_dir_label = '/home/gwb/Gittest/result/images/Denosing_Result/'
        #filename = "/home/gwb/Train_prior/Data/dataset2.npy"

        self.data = read_data(root_dir_data, True, 'noisy')
        #self.data = Patch((self.data, windows_size, overlap_size))
        self.label = read_data(root_dir_label, False, 'label')
        #self.data = Patch((self.data, windows_size, overlap_size))

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


# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020
这个文档是当时为了用矩阵来实现进行的
@author: lx
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import copy
from dataset_test import ImageDataset
from models import DnCNN
from dataset_origin import prepare_data, Dataset
from utils import *
from core.windows import window_partition,window_reverse
from tool import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = arg_set()
opt = parser.parse_args()

def Normlize(img,label):
    B, C, H, W = img.shape
    img = img.view(B, -1)
    label = label.view(B, -1)

    label -= img.min(1, keepdim=True)[0]
    img -= img.min(1, keepdim=True)[0]

    label /= img.max(1, keepdim=True)[0]
    img /= img.max(1, keepdim=True)[0]

    img = img.view(B, C, H, W)
    label = label.view(B, C, H, W)
    return img, label


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_test = ImageDataset()
    #dataset_val = ImageDataset()
    loader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_test)))
    # Build model

    for i, (data, label) in enumerate(loader_test, 0):
        # testing step
        #一次只处理一个数据，但是切分成多个也可以很好的利用并行化计算
        torch.cuda.empty_cache()
        data_lap = Patch(data, opt.windows_size, opt.Overlap)
        label_lap = Patch(data, opt.windows_size, opt.Overlap)

        B, C, H, W = data_lap.shape
        window_size = opt.windows_size

        data_split_norm = window_partition(data_lap, window_size)
        label_split_norm = window_partition(label_lap, window_size)

        item = test(data_split_norm, label_split_norm)
        result = item.cpu()
        #解归一化
        #result = Anti_Normlize(data_split,result)

        result = window_reverse(result, window_size, H, W)
        result = result.numpy().reshape(B,C,H,W)

        result = combine(result, opt.windows_size, opt.overlap_size)

        noise = data.numpy()-result
        img_name = 'the_' + str(i) + '_seismic'
        np.save('%s/test/images_result/%s_%s.npy' %(opt.exp_path, img_name, opt.mode),result)
        np.save('%s/test/images_noise/%s_%s.npy' % (opt.exp_path, img_name, opt.mode), noise)

def test(data,label):
    with torch.no_grad():
        net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
        criterion = nn.MSELoss(size_average=False)
        # Move to GPU
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_17_globalnorm.pth')))
        criterion.cuda()
        # training
        writer = SummaryWriter(opt.outf)
        model.eval()

        img_test = label
        imgn_test = data
        noise = imgn_test - img_test

        save_imgs = img_test.cpu().clone()
        save_noise = noise.cpu().clone()

        noise = Variable(noise.cuda())
        out_test = model(imgn_test)
        loss = criterion(out_test, noise) / (imgn_test.size()[0] * 2)

        #img_test, imgn_test = img_test.cuda(), imgn_test.cuda()
        # results
        out_result = torch.clamp(imgn_test - out_test.cpu(), -2, 2.)
        #out_result = combine(out_result,wins,overlap)
        #img_test = combine(img_test,wins,overlap)

        psnr_train = batch_PSNR(out_result, img_test, 1.)
        print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
              (1, 1, 100, loss.item(), psnr_train))
        #output = window_reverse(out_test, window_size, H, W)
        output = out_result#.reshape(H, W)

        img_name = 'the_' + str(1) + '_seismic'
        # np.save('%s/images_result/%s_%s.npy' %(opt.exp_path, img_name, opt.mode),output.detach().cpu().numpy())
        save_imgs = torch.cat((save_imgs, out_test.cpu()), dim=0)
        # save_noise = torch.cat((save_noise, out_train.cpu()), dim=0)
        utils.save_image(
            save_imgs.float(),
            '%s/images_sheet/%s_%s.jpg' %
            (opt.exp_path, img_name, opt.mode),
            nrow=int(save_imgs.size(0) ** 1),
            normalize=True)
        utils.save_image(
            imgn_test.float(),
            '%s/images_clean/%s_%s.jpg' %
            (opt.exp_path, img_name, opt.mode),
            nrow=int(save_imgs.size(0) ** 1),
            normalize=True)

        utils.save_image(
            save_noise.float(),
            '%s/images_noise/%s_%s.jpg' %
            (opt.exp_path, img_name, opt.mode),
            nrow=int(save_imgs.size(0) ** 1),
            normalize=True)
        # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
        ## the end of each epoch
        torch.cuda.empty_cache()
        return output#.squeeze(0)#之前的时候需要增加维度


if __name__ == "__main__":
    '''if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)'''
    main()

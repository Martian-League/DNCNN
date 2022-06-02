# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:17:19 2020
这个文档是多次波压制的程序
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

from dataset_shuffle import ImageDataset
from models import DnCNN
from dataset_origin import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=8, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=5, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=5, help='noise level used on validation set')
parser.add_argument("--exp_path", type=str, default="/home/gwb/DNCNN/result", help='with known noise level (S) or blind training (B)')
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = ImageDataset()
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    criterion = nn.MSELoss(size_average=False)
    criterion.cuda()
    #net.apply(weights_init_kaiming)
    # Move to GPU
    device_ids = [0,2]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_17_globalnorm.pth')))
    model.eval()
    for i, (data, label) in enumerate(loader_train, 0):
        # training step
        img_train = label.cuda()
        imgn_train = data.cuda()
        #label = label.cpu().numpy()

        save_imgs = img_train.cpu().clone()
        img_denoise = model(imgn_train)
        out_train = torch.clamp(imgn_train-img_denoise, 0., 1.)

        loss = criterion(out_train, imgn_train) / (imgn_train.size()[0] * 2)
        psnr_train = batch_PSNR(out_train, imgn_train, 1.)

        ##解归一化,即使打乱也不影响，#并未对label做归一化，主要是利用label来恢复到原先地大小
        out_train = Anti_Normlize(img_train, out_train)
        #img_denoise = Anti_Normlize(img_train, img_denoise)

        ##解随机打乱
        result = Anti_Shuffle(out_train)
        #noise = Anti_Shuffle(img_denoise)
        print(torch.max(label[0,:,:,:]))
        label_save = Anti_Shuffle(label)
        print(np.max(label_save[0,:,:]))
        noise = label_save-result
        print("[%d/%d] loss: %.4f PSNR_train: %.4f" %
            ( i+1, len(loader_train), loss.item(), psnr_train))
        img_name = 'the_'+str(i)+'_seismic'
        img_noise = 'the_'+str(i)+'_noise'
        img_origin = 'the_' + str(i) + '_origin'

        np.save('%s/shuffle_result/images_result/%s_%s.npy' % (opt.exp_path, img_name, opt.mode), result)
        np.save('%s/shuffle_result/images_noise/%s_%s.npy' % (opt.exp_path, img_noise, opt.mode), noise)
        np.save('%s/shuffle_result/images_origin/%s_%s.npy' % (opt.exp_path, img_origin, opt.mode), label_save)
        save_imgs = torch.cat((save_imgs, out_train.cpu()), dim=0)
        utils.save_image(
            save_imgs.float(),
            '%s/shuffle_result/images_sheet/%s_%s.jpg' %
            (opt.exp_path, img_name, opt.mode),
            nrow=int(save_imgs.size(0) ** 1),
            normalize=True)
        utils.save_image(
            imgn_train.float(),
            '%s/shuffle_result/images_clean/%s_%s.jpg' %
            (opt.exp_path, img_name, opt.mode),
            nrow=int(save_imgs.size(0) ** 1),
            normalize=True)

    #stack_noise()

def stack_noise():
    root_dir="/home/gwb/DNCNN/result/shuffle_result/images_noise/"
    record1=[]
    for i in range(205//8+1):
        filename = root_dir + 'the_'+str(i)+'_noise_B.npy'
        y = np.load(filename)
        y = y.astype(np.float32)
        #print(y.shape)
        result = np.sum(y, axis=2)
        record1.append(result)
    record1 = np.concatenate(record1, axis=0)
    stack_name = 'stack_205'
    np.save('%s/shuffle_result/%s_%s.npy' % (opt.exp_path, stack_name, opt.mode), record1)

def stack_signal():
    root_dir="/home/gwb/DNCNN/result/shuffle_result/images_result/"
    record1=[]
    for i in range(205//8+1):
        filename = root_dir + 'the_'+str(i)+'_seismic_B.npy'
        y = np.load(filename)  # (1000,128,128,1)
        y = y.astype(np.float32)
        #print(y.shape)
        result = np.sum(y, axis=2)
        record1.append(result)
    record1 = np.concatenate(record1, axis=0)
    print(record1.shape)
    stack_name = 'stack_205_signal'
    np.save('%s/shuffle_result/%s_%s.npy' % (opt.exp_path, stack_name, opt.mode), record1)

def stack_origin():
    root_dir="/home/gwb/DNCNN/result/shuffle_result/images_origin/"
    record1=[]
    for i in range(205//8+1):
        filename = root_dir + 'the_'+str(i)+'_origin_B.npy'
        y = np.load(filename)
        y = y.astype(np.float32)
        #print(y.shape)
        result = np.sum(y, axis=2)
        record1.append(result)
    record1 = np.concatenate(record1, axis=0)
    stack_name = 'stack_205_origin'
    np.save('%s/shuffle_result/%s_%s.npy' % (opt.exp_path, stack_name, opt.mode), record1)


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

def Anti_Normlize(img,result):
    B, C, H, W = img.shape
    #img = img.view(img.size(0), -1)
    result = result.view(result.size(0), -1)
    img = img.view(img.size(0), -1)
    result *= (img.max(1, keepdim=True)[0]-img.min(1, keepdim=True)[0])
    result += img.min(1, keepdim=True)[0]

    result = result.view(B, C, H, W)
    return result
    # 解归一化
    '''batch_size = label.shape[0]
    MAX = np.max(label,axis=(2,3))
    MIN = np.min(label,axis=(2,3))
    MAX = np.expand_dims(MAX,axis=2)
    MIN = np.expand_dims(MIN,axis=2)
    result = result * (MAX - MIN) + MIN
    noise = noise * (MAX - MIN) + MIN'''

def Anti_Shuffle(img):
    B,C,H,W = img.shape
    #print(img.shape)
    img = img.detach().cpu().numpy()
    #noise = img_denoise.squeeze().detach().cpu().numpy()
    #label = label.squeeze().detach().cpu().numpy()

    E_shuffle = np.load("/home/gwb/DNCNN/result/E_shuffle.npy")
    #result = np.dot(result, E_shuffle.T)
    #noise = np.dot(noise, E_shuffle.T)
    img_save = np.dot(img, E_shuffle.T)
    img_save = img_save.reshape(B,H,W)
    return img_save


if __name__ == "__main__":
    '''if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)'''
    #main()
    #stack_noise()
    #stack_signal()
    stack_origin()

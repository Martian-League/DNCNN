import math
import torch
import torch.nn as nn
import numpy as np
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        #print(np.shape(Iclean[i,:,:,:]))
        SSIM += compare_ssim(Iclean[i,0,:,:], Img[i,0,:,:], channel_axis=False)
    return (SSIM/Img.shape[0])

def SSIM_loss(img,imclean,level):
    #设置各层的所占的权重
    weight_layer = [0.2,0.2,0.2,0.2,0.2]
    ssim_loss = 0
    img_iter = img.clone()
    label_iter = imclean.clone()
    downsample = nn.AvgPool2d(2, stride=2)
    for i in range(level):
        loss = weight_layer[i]*batch_SSIM(img_iter, label_iter)
        ssim_loss = ssim_loss+loss
        img_iter = downsample(img_iter)
        label_iter = downsample(label_iter)
    return ssim_loss

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def arg_set():
    parser = argparse.ArgumentParser(description="DnCNN")
    parser.add_argument("--batchSize", type=int, default=300, help="Training batch size")
    parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--milestone", type=int, default=8,
                        help="When to decay learning rate; should be less than epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--outf", type=str, default="logs", help='path of log files')

    parser.add_argument("--exp_path", type=str, default="/home/gwb/DNCNN/result",
                        help='with known noise level (S) or blind training (B)')
    parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
    parser.add_argument("--path_train", type=str, default="/home/gwb/Dataset/train/", help='path of log files')
    parser.add_argument("--path_val", type=str, default="/home/gwb/Dataset/train/", help='path of log files')

    parser.add_argument("--Windows_size", type=int, default=600, help="Number of total layers")
    parser.add_argument("--Overlap_size", type=int, default=100, help="Number of total layers")
    parser.add_argument("--seismic_path", type=str, default="/home/gwb/PPP/data/trainset/Testdata/",
                        help='with known noise level (S) or blind training (B)')
    parser.add_argument("--seismic_label", type=str, default="/home/gwb/Gittest/result/images/Denosing_Result/",
                        help='with known noise level (S) or blind training (B)')
    return parser

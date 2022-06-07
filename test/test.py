import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset_test import ImageDataset
from models import DnCNN
from utils import *
from core.windows import window_partition,window_reverse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=5, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=5, help='noise level used on validation set')
parser.add_argument("--exp_path", type=str, default="/home/gwb/DNCNN/result",
                    help='with known noise level (S) or blind training (B)')
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--Windows_size", type=int, default=600, help="Number of total layers")
parser.add_argument("--Overlap_size", type=int, default=100, help="Number of total layers")
parser.add_argument("--seismic_path", type=str, default="/home/gwb/PPP/data/trainset/Testdata/",
                    help='with known noise level (S) or blind training (B)')
parser.add_argument("--seismic_label", type=str, default="/home/gwb/Gittest/result/images/Denosing_Result/",
                    help='with known noise level (S) or blind training (B)')
opt = parser.parse_args()

def test(data,label):
    with torch.no_grad():
        net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
        criterion = nn.MSELoss(size_average=False)
        # Move to GPU
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        #25个样本大小
        model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_17_segy_25num.pth')))
        #50个样本大小
        #model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_17_globalnorm.pth')))
        criterion.cuda()
        # training
        writer = SummaryWriter(opt.outf)
        model.eval()
        window_size = 576
        H, W = data.shape[2:]
        #data = window_partition(data, window_size)
        #label = window_partition(label, window_size)
        #data = data.unsqueeze(0)
        #label = label.unsqueeze(0)
        #print(label.shape)
        img_test = label
        imgn_test = data
        noise = imgn_test - img_test

        save_imgs = img_test.cpu().clone()
        save_noise = noise.cpu().clone()

        noise = Variable(noise.cuda())
        out_test = model(imgn_test)
        loss = criterion(out_test, noise) / (imgn_test.size()[0] * 2)

        img_test, imgn_test = img_test.cuda(), imgn_test.cuda()
        # results
        out_test = torch.clamp(imgn_test - out_test, -1., 1.)
        psnr_train = batch_PSNR(out_test, img_test, 1.)
        print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
              (1, 1, 100, loss.item(), psnr_train))
        #output = window_reverse(out_test, window_size, H, W)
        output = out_test#.reshape(H, W)

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
        return output#.squeeze(0)

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
    img = img.view(img.size(0), -1)
    result = result.view(result.size(0), -1)

    result *= (img.max(1, keepdim=True)[0]-img.min(1, keepdim=True)[0])
    result += img.min(1, keepdim=True)[0]

    result = result.view(B, C, H, W)
    return result
def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = ImageDataset()
    #dataset_val = ImageDataset()
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    result=[]
    for i, (data, label) in enumerate(loader_train, 0):
        # testing step
        torch.cuda.empty_cache()
        B, C, H, W = data.shape
        window_size = W//2
        data_split_norm = window_partition(data,window_size)
        label_split_norm = window_partition(label, window_size)

        torch.cuda.empty_cache()

        #将原先的for循环的方式改为矩阵方式，这样如果有多个GPU，就能够提高处理的效率，
        #data_split_norm的第一个通道为B，也就是从单个数据切出来的数据个数
        '''   之前的做法    
        for j in range(data_split_norm.shape[0]):
            torch.cuda.empty_cache()
            #print(torch.max(label_split_norm[j, :, :, :]))
            #print(torch.min(label_split_norm[j, :, :, :]))
            item = test(data_split_norm[j, :, :, :].unsqueeze(0), label_split_norm[j, :, :, :].unsqueeze(0))
            result.append(item.cpu())
        result = torch.tensor([item.detach().numpy() for item in result])'''

        item = test(data_split_norm,label_split_norm)
        result = item.cpu()
        #解归一化
        #result = Anti_Normlize(data_split,result)
        result = window_reverse(result, window_size, H, W)

        result = result.numpy().reshape(H,W)
        noise = data.numpy()-result

        img_name = 'the_' + str(i) + '_seismic'
        np.save('%s/test/images_result/%s_%s.npy' %(opt.exp_path, img_name, opt.mode),result)
        np.save('%s/test/images_noise/%s_%s.npy' % (opt.exp_path, img_name, opt.mode), noise)
if __name__ == "__main__":
    '''if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)'''
    main()

import os
import sys
sys.path.append("/tmp/pycharm_project_577/tool")
from tool.tool import Anti_Normlize

import argparse
import segyio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset.dataset_segy_test import SegyDataset
from core.models import DnCNN

from core.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=8, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=5, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=5, help='noise level used on validation set')
parser.add_argument("--exp_path", type=str, default="/home/gwb/DNCNN/result", help='with known noise level (S) or blind training (B)')
parser.add_argument("--logdir", type=str, default="/home/gwb/DNCNN/logs", help='path of log files')
parser.add_argument("--path_val", type=str, default="/home/gwb/Dataset/train/", help='path of log files')
parser.add_argument("--path_train", type=str, default="/home/gwb/Dataset/train/", help='path of log files')

opt = parser.parse_args()

def main():
    # Load dataset
    with torch.no_grad():
        print('Loading dataset ...\n')
        dataset_train = SegyDataset(opt.path_val)
        #dataset_train = ImageDataset()
        loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=False)
        print("# of training samples: %d\n" % int(len(dataset_train)))
        # Build model
        net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
        criterion = nn.MSELoss(size_average=False)
        criterion.cuda()
        #net.apply(weights_init_kaiming)
        # Move to GPU
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_17_segy_60.pth')))
        model.eval()
        for i, (data, label,raw_data,img_name) in enumerate(loader_train, 0):
            # training step
            torch.cuda.empty_cache()
            img_train = label.cuda()
            raw_data = raw_data.cuda()
            imgn_train = data.cuda()
            #label = label.cpu().numpy()

            save_imgs = img_train.cpu().clone()
            img_denoise = model(imgn_train)
            out_train = torch.clamp(imgn_train-img_denoise, 0, 1.)

            loss = criterion(out_train, img_train) / (img_train.size()[0] * 2)
            psnr_train = batch_PSNR(out_train, img_train, 1.)

            result, noise = Anti_Normlize(raw_data,out_train)
            result = result.cpu().squeeze().numpy()
            noise = noise.cpu().squeeze().numpy()
            #img_denoise = Anti_Normlize(img_train, img_denoise)#这种做法是错误的
            print(result.shape)
            #result = out_train
            #noise = img_denoise.cpu().squeeze().numpy()
            #暂时先放这，没有进行复原
            #print(torch.max(label[0,:,:,:]))
            label_save = label.cpu().numpy()

            print("[%d/%d] loss: %.4f PSNR_train: %.4f" %
                ( i+1, len(loader_train), loss.item(), psnr_train))
            #print(img_name)
            img_name = img_name[0][:-4]#img_name是个元组
            print(img_name)
            #img_noise = 'the_'+str(i)+'_noise'
            #img_origin = 'the_' + str(i) + '_origin'

            '''
            np.save('%s/segy_dncnn/images_result/%s_%s.npy' % (opt.exp_path, 'result', img_name), result)
            np.save('%s/segy_dncnn/images_noise/%s_%s.npy' % (opt.exp_path, 'noise', img_name), noise)
            np.save('%s/segy_dncnn/images_origin/%s_%s.npy' % (opt.exp_path, 'origin', img_name), label_save)

            result.T.tofile('%s/segy_dncnn/images_result/%s_%s.bin' % (opt.exp_path, 'result', img_name))
            noise.T.tofile('%s/segy_dncnn/images_noise/%s_%s.bin' % (opt.exp_path, 'noise', img_name))
            label_save.T.tofile('%s/segy_dncnn/images_origin/%s_%s.bin' % (opt.exp_path, 'origin', img_name))
            '''
            segyio.tools.from_array2D('%s/segy_dncnn/images_result_norm/%s_%s.sgy' % (opt.exp_path, 'result', img_name), result.T, dt=2000)
            segyio.tools.from_array2D('%s/segy_dncnn/images_noise_norm/%s_%s.sgy' % (opt.exp_path, 'noise', img_name), noise.T, dt=2000)
            #segyio.tool.from_array2D('%s/segy_dncnn/images_result/%s_%s.bin' % (opt.exp_path, 'result', img_name), label_save.T)

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


if __name__ == "__main__":
    '''if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)'''
    main()


import argparse

import torch
import numpy as np
from torch.autograd import Variable
import torchvision.utils as utils
import torch.optim as optim
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset_segy import SegyDataset
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description="DnCNN_Seismic")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=10, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=8, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=5, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=5, help='noise level used on validation set')
parser.add_argument("--exp_path", type=str, default="/home/gwb/DNCNN/result", help='with known noise level (S) or blind training (B)')
parser.add_argument("--path_val", type=str, default="/home/gwb/Dataset/train/", help='path of log files')
parser.add_argument("--path_train", type=str, default="/home/gwb/Dataset/train/", help='path of log files')
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

def main():
    #Load dataset
    print('Loading dataset ...\n')
    dataset_train = SegyDataset(opt.path_train)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    dataset_val = SegyDataset(opt.path_val)
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    print("Finished！")
    #Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_id = [0]
    model = nn.DataParallel(net,device_ids=device_id).cuda()
    #model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_17_DIP.pth')))
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    writer = SummaryWriter(opt.outf)
    step = 0

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr/10
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print('learning rate %f' % current_lr)
        for i,(data,label) in enumerate(loader_train):
            torch.cuda.empty_cache()
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            img_train = label
            imgn_train = data
            noise = imgn_train-img_train

            save_imgs = img_train.cpu().clone()
            save_noise = noise.cpu().clone()

            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_noise = model(imgn_train)
            loss = criterion(out_noise, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()

            out_train = torch.clamp(imgn_train - model(imgn_train), 0, 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)

            print("[epoch %d][%d/%d] loss:%.4f PSNR: %.4f"%
                  (epoch+1,i+1,len(loader_train),loss.item(),psnr_train))
            img_name = 'the_' + str(i) + '_seismic'
            save_imgs = torch.cat((save_imgs, out_train.cpu()), dim=0)
            save_noise = torch.cat((save_noise, out_noise.cpu()), dim=0)
            utils.save_image(
                save_imgs.float(),
                '%s/images_sheet/%s_%s.jpg' %
                (opt.exp_path, img_name, opt.mode),
                nrow=int(save_imgs.size(0) ** 1),
                normalize=True)
            utils.save_image(
                imgn_train.float(),
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
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            #step += 1

                model.eval()
                psnr_val = 0

                for k,(data_val,label_val) in enumerate(loader_val):
                    out_val = torch.clamp(data_val-model(data_val).cpu(),0.,1.)
                    psnr_val += batch_PSNR(out_val, label_val, 1.)
                psnr_val /= len(dataset_val)
                print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
            step += 1

            Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
            Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
            Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
            writer.add_image('clean image', Img, epoch)
            writer.add_image('noisy image', Imgn, epoch)
            writer.add_image('reconstructed image', Irecon, epoch)
            # save model
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net_17_segy.pth'))

if __name__=="__main__":
    main()
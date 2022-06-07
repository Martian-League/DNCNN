import os
import sys
#print(sys.path)
sys.path.append("/home/gwb/DNCNN/")
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.models import DnCNN
from core.utils import *
from dataset.dataset_segy import SegyDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=400, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=8, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=5, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=5, help='noise level used on validation set')
parser.add_argument("--exp_path", type=str, default="/home/gwb/DNCNN/result", help='with known noise level (S) or blind training (B)')
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--path_train", type=str, default="/home/gwb/Dataset/train/", help='path of log files')
parser.add_argument("--path_val", type=str, default="/home/gwb/Dataset/train/", help='path of log files')
opt = parser.parse_args()

def inv_wavelet(coeffs):
    LL = coeffs[:, 0:, :, :].cpu().numpy()
    LH = coeffs[:, 1:, :, :].cpu().numpy()
    HL = coeffs[:, 2:, :, :].cpu().numpy()
    HH = coeffs[:, 3:, :, :].cpu().numpy()
    coeff = ((LL),( LH, HL, HH))
    inv_wave = torch.tensor(pywt.idwt2(coeff, 'db4'))
    return inv_wave

def main():
    # Load dataset
    print('Loading dataset ...\n')
    #dataset_train = ImageDataset()
    dataset_train = SegyDataset(opt.path_train)
    loader_train = DataLoader(dataset=dataset_train, num_workers=2, batch_size=opt.batchSize, shuffle=True)
    dataset_val = SegyDataset(opt.path_val)
    loader_val = DataLoader(dataset=dataset_val, num_workers=2, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)

    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0,1]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    seed = 456
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    #model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_17_segy.pth')))
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    #noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(0,opt.epochs):
        '''if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr'''
        #print('learning rate %f' % current_lr)
        # train
        for i, (data, label)in enumerate(loader_train, 0):
            # training step
            if torch.max(data) == torch.min(label):
                print("暂时没有成功")
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
            loss_mse = criterion(out_noise, noise) / (imgn_train.size()[1]*2)

            out_train = torch.clamp(imgn_train - model(imgn_train),-2,2)
            TV_loss = TVLoss()
            tv_loss  = TV_loss(out_train)
            ssim_loss = SSIM_loss(out_train, img_train, 3)
            beta = 0
            alpha = 0
            loss = loss_mse + beta*tv_loss - alpha*ssim_loss

            loss.backward()
            optimizer.step()
            # results
            #model.eval()
            out_train = torch.clamp(imgn_train - model(imgn_train), -2, 2.)


            psnr_train = batch_PSNR(out_train, img_train, 1.)
            #print(np.shape(out_train))
            ssim_train = batch_SSIM(out_train, img_train)
            print("[epoch %d][%d/%d] loss: %.4f ,tv_loss: %.4f, PSNR_train: %.4f SSIM_train: %.4f SSIM_loss: %.4f" %
                (epoch+1, i+1, len(loader_train), loss_mse.item(), tv_loss.item(), psnr_train, ssim_train,ssim_loss))
            img_name = 'the_'+str(i)+'_seismic'
            np.save('%s/test/images_result/%s_%s.npy' % (opt.exp_path, img_name, opt.mode), out_train.detach().cpu().numpy())
            save_imgs = torch.cat((save_imgs[[0]], out_train.cpu()[[0]]), dim=0)
            #当批次较大时使用上面的，不然一下子保存的图像太多就会出错
            #save_imgs = torch.cat((save_imgs, out_train.cpu()), dim=0)
            #save_noise = torch.cat((save_noise, out_train.cpu()), dim=0)

            '''utils.save_image(
                imgn_train.float(),
                '%s/images_clean/%s_%s.jpg' %
                (opt.exp_path, img_name, opt.mode),
                nrow=int(save_imgs.size(0) ** 1),
                normalize=True)'''

            utils.save_image(
                save_imgs.float(),
                '%s/images_sheet/%s_%s.jpg' %
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
            if step % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            if step % 200 == 0:
                # Log the scalar values
                with torch.no_grad():
                    model.eval()
                    psnr_val = 0

                    for k,(data_val,label_val) in enumerate(loader_val):
                        #out_val = torch.clamp(data_val-model(data_val).cpu())
                        out_val = torch.clamp(data_val - model(data_val).cpu(),-2,2)
                        psnr_val += batch_PSNR(out_val, label_val, 1.)
                        count = k+1
                        if count == 50:
                            break
                    psnr_val /= count
                    print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
                    writer.add_scalar('PSNR on validation data', psnr_val, epoch)
            step += 1
        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train),-2,2)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net_17_segy_25num.pth'))

if __name__ == "__main__":
    '''if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)'''
    main()

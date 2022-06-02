import torch
import torch.nn as nn
from core.models import DnCNN

class Parallel_DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(Parallel_DnCNN, self).__init__()

        self.dncnn1 = DnCNN(channels, num_of_layers)
        self.dncnn2 = DnCNN(channels, num_of_layers)
        self.dncnn3 = DnCNN(channels, num_of_layers)
        self.dncnn4 = DnCNN(channels, num_of_layers)
    def forward(self, x):
        out1 = self.dncnn1(x[:,[0],:,:])
        out2 = self.dncnn2(x[:,[1],:,:])
        out3 = self.dncnn3(x[:,[2],:,:])
        out4 = self.dncnn4(x[:,[3],:,:])
        #print(out4.shape)
        out = torch.cat([out1,out2,out3,out4],dim = 1)
        #print(out.shape)
        return out

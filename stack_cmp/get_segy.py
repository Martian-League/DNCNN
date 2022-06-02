import os
import numpy as np
import segyio
import torch

def read_data(file_path):

    metas=[]
    sample=[]

    for filename in os.listdir(file_path):  # 展开成一个新的列表
        metas.append(filename)
    metas.sort(key=lambda x:int(x.split('_')[-2]))

    for filename in metas:

        sample_t = ReadSegyData(os.path.join(file_path,filename))
        sample.append(sample_t.astype(np.float32))  # ,此时读取来的数据com还是1个200*3001的一长串，不能二维显示，需要转换形状

    #sample_torch = [torch.FloatTensor(item) for item in sample]

    return sample, metas

def ReadSegyData(filename):
    with segyio.open(filename,'r',ignore_geometry=True) as f:
        f.mmap()
        data2D = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()

    return data2D
# 导入os库
import os
import numpy as np
import segyio
def main():
    # 图片存放的路径
    path = r"/home/gwb/DNCNN/result/segy_dncnn/images_noise_norm_25_newcadowz/"
    metaslabel=[]
    metasdata=[]

    for filename in os.listdir(path):  # 展开成一个新的列表
        metaslabel.append(filename)
    for filename in os.listdir("/home/gwb/DNCNN/result/segy_dncnn/images_noise_norm_25/"):  # 展开成一个新的列表
        metasdata.append(filename)
    metaslabel.sort(key=lambda x:int(x.split('_')[-2]))
    metasdata.sort(key=lambda x:int(x.split('_')[-2]))
    metas = list(zip(metaslabel,metasdata))
    # 遍历更改文件名
    for i in range(len(metaslabel)):
        filename1 = metaslabel[i]
        os.rename(os.path.join(path,filename1),os.path.join(path,metasdata[i]))
    print("finish!")
def printt():

    """
    metasdata=[]
    for filename in os.listdir("/home/gwb/DNCNN/result/segy_dncnn/images_noise_norm_10/"):  # 展开成一个新的列表
        metasdata.append(filename)
    metasdata.sort(key=lambda x: int(x.split('_')[-2]))
    print(metasdata[1800])

    """
    x = "noise_gather_397933_398919.sgy"
    print(x.split('_'))
def in_out():
    result = np.fromfile("/home/gwb/DNCNN/result/shuffle_result/images_result/stack_205_origin_B.bin",np.float32)
    result = result.reshape(205,500);
    segyio.tools.from_array2D(
        "/home/gwb/DNCNN/result/shuffle_result/images_result/stack_205_origin_B.sgy" ,
        result, dt=2000)
    print('finish')

if __name__=="__main__":
    # main()
    #printt()
    in_out()
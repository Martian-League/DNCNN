import os
import numpy as np
import segyio

def ReadSegyData(filename):
    with segyio.open(filename,'r',ignore_geometry=True) as f:
        f.mmap()
        data2D = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data2D

def main():

    source1_path = os.path.abspath(r'/home/gwb/DNCNN/result/segy_dncnn/images_result_norm/')
    source2_path = os.path.abspath(r'/home/gwb/DNCNN/result/segy_dncnn/images_result_norm_test/')

    target_path = os.path.abspath(r'/home/gwb/DNCNN/result/segy_dncnn/download/compare')
    file_com = [382214,866556,932193,1203627,1280711]
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(source1_path):
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
        #for root1, dirs, files in os.walk(source1_path):
        for file in os.listdir(source1_path):
            if int(file.split('_')[-2]) in file_com:
                src1_file = os.path.join(source1_path, file)
                src2_file = os.path.join(source2_path, file)

                data1 = ReadSegyData(src1_file)
                data2 = ReadSegyData(src2_file)
                minus = data1-data2
                print(np.max(minus))
                segyio.tools.from_array2D(os.path.join(target_path,file), minus.T, dt=2000)

    print('copy files finished!')
if __name__ == '__main__':
    main()
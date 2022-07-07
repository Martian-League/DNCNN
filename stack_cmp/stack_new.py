import sys
sys.path.append("..")
import numpy as np
import segyio
from get_segy import read_data
from core import arg_set

parser = arg_set()
opt = parser.parse_args()

'''
read_data(opt.path_train,False)
'''

def stack_segy(src_path, dst_path):

    record_data = []
    record_label = []
    data, img_name = read_data(src_path)
    #label, label_name = read_data(label_path)
    num_gather = len(data)

    for i in range(num_gather):

        #item_data = data[i]-label[i]
        item_data = data[i]
        #print(item_data.shape)
        record_data.append(np.sum(item_data, axis=1,keepdims = True))
    #print(record_data[0].shape)
    record_data = np.concatenate(record_data, axis=1)
    print(record_data.shape)

        #dst_root = "/home/gwb/DNCNN/result/segy_dncnn/stack_2050/"
        #dst_data = os.path.join(dst_root,"stack_cmp")

    segyio.tools.from_array2D(dst_path, record_data.T, dt=2000)
def main():
    '''
    src_path = "/home/gwb/Dataset/train/sample/"
    dst_path = "/home/gwb/DNCNN/result/segy_dncnn/stack_2050/origin_gather_stack.sgy"
    :return:
    '''
    '''
    src_path = "/home/gwb/Dataset/train/label/"
    dst_path = "/home/gwb/DNCNN/result/segy_dncnn/stack_2050/clean_DIP_gather_stack.sgy"
    '''
    src_path = "/home/gwb/DNCNN/result/segy_dncnn/images_noise_norm_25_newcadowz/"
    #label_path = "/home/gwb/Dataset/train/label/"
    dst_path = "/home/gwb/DNCNN/result/segy_dncnn/stack_2050/noise_norm_DNCNN_gather_stack_cadows_num25.sgy"
    stack_segy(src_path, dst_path)

def antinorm():
    src_path1 = "/home/gwb/DNCNN/result/segy_dncnn/images_noise_norm_25/"
    src_path2 = "/home/gwb/Dataset/train/sample/"
    data, img_name = read_data(src_path2)
    noise, noise_name = read_data(src_path1)
    num_gather = len(data)
    print(num_gather)
    print(len(noise))
    for i in range(num_gather):
        MAX = np.max(data[i])
        noise[i] = noise[i]*MAX
        if i%500==0:
            print(noise_name[i])
            print(img_name[i])
        segyio.tools.from_array2D('%s/segy_dncnn/images_noise_norm_25/%s_%s' % (opt.exp_path, 'noise', noise_name[i]),
                              noise[i].T, dt=2000)
if __name__=="__main__":
    main()
    #antinorm()
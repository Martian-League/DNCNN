import os
import random
def main():
    root_dir = "/home/gwb/Dataset/train/"
    metas=[]
    metaslabel=[]

    for filename in os.listdir(root_dir + "sample"):  # 展开成一个新的列表
        metas.append(filename)
    for filename in os.listdir(root_dir + "label"):  # 展开成一个新的列表
        metaslabel.append(filename)
    print(len(metas))
    print(len(metaslabel))

    for file in metas:
        temp = "clean_"+file
        if temp not in metaslabel:
            f = open("/home/gwb/Dataset/label_list.txt",'w+')
            #f.seek(0)
            f.truncate(0)
            f.write(temp+ os.linesep)
            f.close()

if __name__=="__main__":
    main()
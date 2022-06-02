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
def test():
    import os
    import sys
    # 获取当前代码文件绝对路径
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    # 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到sys.path中
    sys.path.append("/home/gwb/DNCNN/core/")
    import core
if __name__=="__main__":
    #main()
    test()
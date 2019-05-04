import os
import glob
import matplotlib.image as mpimg
import pickle # 一种存储格式的库
def give_me_filename(dirname, suffixes, prefix=""):
    """
        函数的作用是根据dirname和suffixes生成对应的文件，如果shuffixes是个list，则生成同一前缀文件的多个后缀文件名 
    """
    if not isinstance(suffixes, list):
        suffixes=[suffixes]
    
    suffixes=[p if p[0]=='.' else '.'+p for p in suffixes]
    #如果suffixes前面不带.就加上点，否则不变
          
    while True:
        bname="%09d"%random.randint(0,999999999)#字段宽9，随机生成一个9位的数字
        fnames=[]
        for suffix in suffixes:
            fname=os.path.join(dirname,prefix+bname+suffix)
            if not os.path.isfile(fname):#判断是否这个文件已经存在了
                fnames.append(fname)#不存在就表示生成的可以加入结果了
                
        if len(fnames) == len(suffixes): break
    
    if len(fnames)==1:
        return fnames[0]
    else:
        return fnames
def loadDTD(DTD_path,out_put_path,filename):
    bg_images = []
    for subdir in glob.glob(DTD_path + '/*'):
        for f in glob.glob(subdir + '/*.jpg'):
            bg_images.append(mpimg.imread(f))
    print("背景图片总数:{}".format(len(bg_images)))
    pickle.dump(bg_images,open(out_put_path + "/" + filename,'wb',encoding='utf-8'))

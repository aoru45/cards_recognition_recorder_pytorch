import glob
import os
import shutil
def fz(path,target_path):
    images = glob.glob(path + "/*.jpg")
    labels = glob.glob(path + "/*.xml")
    i=0
    for image in images:
        name = image.split("/")[-1]
        shutil.move(image,target_path[0]+"/"+name)
    for label in labels:
        name = label.split("/")[-1]
        shutil.move(label,target_path[1]+"/"+name)
def set_txt(xml_path,train_txt_path):
    xmls = glob.glob(xml_path + "/*.xml")
    for xml in xmls:
        file_data = ""
        with open(xml,'r') as f:
            for line in f:
                
    with open(train_txt_path,"w") as f:
        
        
        for xml in xmls:
            name = image.split("/")[-1][:-1] + ".jpg"
            
            f.write(name + '\n')
        
#fz("/home/xueaoru/视频/val",["/home/xueaoru/视频/VOC/VOC2018/JPEGImages","/home/xueaoru/视频/VOC/VOC2018/Annotations"])
#set_txt()
def upper2lower(xml_path,old_words_dict = ['_A','_J','_K','_Q']):
    labels = glob.glob(xml_path + "/*.xml")
    for label in labels:
        file_data = ""
        with open(label,'r') as f:
            for line in f:
                for old_word in old_words_dict:
                    line = line.replace(old_word,old_word.lower())
                file_data = file_data + line
        with open(label,'w') as f:
            f.write(file_data)
set_txt("")
if __name__ == "__main__":
    upper2lower("/media/xueaoru/DATA/ubuntu/cards/test_dataset")
    train_txt_path = "/home/xueaoru/视频/VOC/VOC2018/ImageSets/Main/train.txt"
    xml_path = "/home/xueaoru/视频/VOC/VOC2018/Annotations"
    val_path = "/home/xueaoru/视频/val"
    jpeg_path = "/home/xueaoru/视频/VOC/VOC2018/JPEGImages"
    annotations_path = "/home/xueaoru/视频/VOC/VOC2018/Annotations"
    #fz(val_path,jpeg_path,annotations_path)
    #set_txt(xml_path)
    #upper2lower()

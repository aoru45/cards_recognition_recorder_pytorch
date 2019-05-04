import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon
import random
xml_body_1="""<annotation>
        <folder>FOLDER</folder>
        <filename>{FILENAME}</filename>
        <path>{PATH}</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>{WIDTH}</width>
                <height>{HEIGHT}</height>
                <depth>3</depth>
        </size>
"""
xml_object=""" <object>
                <name>{CLASS}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{XMIN}</xmin>
                        <ymin>{YMIN}</ymin>
                        <xmax>{XMAX}</xmax>
                        <ymax>{YMAX}</ymax>
                </bndbox>
        </object>
"""
xml_body_2="""</annotation>        
"""
        

class Backgrounds():
    def __init__(self,backgrounds_pck_fn):
        self._images=pickle.load(open(backgrounds_pck_fn,'rb'))
        self._nb_images=len(self._images)
        print("加载图片总数:", self._nb_images)
    def get_random(self, display=False):
        bg=self._images[random.randint(0,self._nb_images-1)]
        if display: 
            plt.imshow(bg)
        return bg


def create_voc_xml(xml_file, img_file,listbba,imgSize,display=False):
    imgW,imgH = imgSize
    with open(xml_file,"w") as f:
        f.write(xml_body_1.format(**{'FILENAME':os.path.basename(img_file), 'PATH':img_file,'WIDTH':imgW,'HEIGHT':imgH}))
        for bba in listbba:            
            f.write(xml_object.format(**{'CLASS':bba.classname,'XMIN':bba.x1,'YMIN':bba.y1,'XMAX':bba.x2,'YMAX':bba.y2}))
        f.write(xml_body_2)
        if display: print("New xml",xml_file)

class Cards():
    def __init__(self,cards_pck_fn=cards_pck_fn):
        self._cards=pickle.load(open(cards_pck_fn,'rb'))
        # 这个cards是一个dic，上面写好的。
        #(img,hullHL,hullLR) 
        self._nb_cards_by_value={k:len(self._cards[k]) for k in self._cards}
        print("Nb of cards loaded per name :", self._nb_cards_by_value)
        
    def get_random(self, card_name=None, display=False):#随机选取一张图片
        if card_name is None:
            card_name= random.choice(list(self._cards.keys()))# 随机选择一个key
        card,hull1,hull2=self._cards[card_name][random.randint(0,self._nb_cards_by_value[card_name]-1)]
        if display:
            if display: display_img(card,[hull1,hull2],"rgb")
        return card,card_name,hull1,hull2
def kps_to_polygon(kps):
    """
    把imgaug格式的kps（特征点）变成shapely的polygon
    """
    pts=[(kp.x,kp.y) for kp in kps]
    return Polygon(pts)

def hull_to_kps(hull, decalX, decalY,imgSize):
    imgW,imgH = imgSize
    """
        将hull变成imgaug的keypoints
    """
    # hull的shape = (n,1,2)
    kps=[ia.Keypoint(x=p[0]+decalX,y=p[1]+decalY) for p in hull.reshape(-1,2)]
    kps=ia.KeypointsOnImage(kps, shape=(imgH,imgW,3))
    return kps

def kps_to_BB(kps):
    """
        通过keypoints确定图像中的Bbox
    """
    extend=3 # 
    kpsx=[kp.x for kp in kps.keypoints] #取每个keypoints的kp的x值
    minx=max(0,int(min(kpsx)-extend))# 左x
    maxx=min(imgW,int(max(kpsx)+extend))#右x
    kpsy=[kp.y for kp in kps.keypoints]#取每个keypoints的kp的y值
    miny=max(0,int(min(kpsy)-extend))# 左y
    maxy=min(imgH,int(max(kpsy)+extend)) # 右y
    if minx==maxx or miny==maxy:# 面积为0的情况--返回
        return None
    else:
        return ia.BoundingBox(x1=minx,y1=miny,x2=maxx,y2=maxy)# 否则返回一个Bbox


# imgaug keypoints of the bounding box of a whole card
#定义一个整个zoom后的图像的keypoints的bbox，一张图的情况
cardKP = ia.KeypointsOnImage([
    ia.Keypoint(x=decalX,y=decalY),#左上角的点
    ia.Keypoint(x=decalX+cardW,y=decalY),   #右上角的点
    ia.Keypoint(x=decalX+cardW,y=decalY+cardH),#右下角的点
    ia.Keypoint(x=decalX,y=decalY+cardH)#左下角的点
    ], shape=(imgH,imgW,3))

# imgaug transformation for one card in scenario with 2 cards
# 利用imgaug将一张画布一张图变换为一张画布两张图
transform_1card = iaa.Sequential([# 定义一个贯序的变换
    iaa.Affine(scale=[0.65,1]),#缩放范围
    iaa.Affine(rotate=(-180,180)),#旋转范围
    iaa.Affine(translate_percent={"x":(-0.25,0.25),"y":(-0.25,0.25)}),#平移范围百分比
])

# For the 3 cards scenario, we use 3 imgaug transforms, the first 2 are for individual cards, 
# and the third one for the group of 3 cards
# 先分别对三张图片进行变换，然后统一进行一个变换
#对一张画布一张图
trans_rot1 = iaa.Sequential([#贯序模型（第一次变换）
    iaa.Affine(translate_px={"x": (10, 20)}),#平移的x范围
    iaa.Affine(rotate=(22,30))#旋转角度范围
])
trans_rot2 = iaa.Sequential([#贯序模型（第二次变换）
    iaa.Affine(translate_px={"x": (0, 5)}),#平移范围
    iaa.Affine(rotate=(10,15))#旋转角度范围
])
transform_3cards = iaa.Sequential([#定义一个三卡场景的变换
    iaa.Affine(translate_px={"x":decalX-decalX3,"y":decalY-decalY3}),#左移cardW/2 上移cardH/2
    iaa.Affine(scale=[0.65,1]),#缩放范围
    iaa.Affine(rotate=(-180,180)),#旋转范围
    iaa.Affine(translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)})#平移范围百分比   
])

# imgaug transformation for the background
scaleBg=iaa.Scale({"height": imgH, "width": imgW})#对背景进行缩放变换到画布的高宽

def augment(img, list_kps, seq, restart=True):# 变换函数（参数img是一张图像） 参数seq是变换方法，也就是上面定义的一种
    """
    将seq作用在img和kps的list上
        Apply augmentation 'seq' to image 'img' and keypoints 'list_kps'
        If restart is False, the augmentation has been made deterministic outside the function (used for 3 cards scenario)
    """ 
    while True:
        if restart:
            myseq=seq.to_deterministic()
        else:
            myseq=seq
        img_aug = myseq.augment_images([img])[0]#取图像变换后的img
        list_kps_aug = [myseq.augment_keypoints([kp])[0] for kp in list_kps]# 对list kps进行变换
        list_bbs = [kps_to_BB(list_kps_aug[1]),kps_to_BB(list_kps_aug[2])]# 将kps转化为bbox对象
        valid=True
        # Check the card bounding box stays inside the image
        for bb in list_bbs:
            if bb is None or int(round(bb.x2)) >= imgW or int(round(bb.y2)) >= imgH or int(bb.x1)<=0 or int(bb.y1)<=0:
                valid=False
                break
        if valid: break
        elif not restart:
            img_aug=None
            break
                
    return img_aug,list_kps_aug,list_bbs

class BBA:  # Bounding box + annotations
    def __init__(self,bb,classname):      
        self.x1=int(round(bb.x1))
        self.y1=int(round(bb.y1))
        self.x2=int(round(bb.x2))
        self.y2=int(round(bb.y2))
        self.classname=classname

class Scene:
    def __init__(self,bg,img1, class1, hulla1,hullb1,img2, class2,hulla2,hullb2,img3=None, class3=None,hulla3=None,hullb3=None):
        if img3 is not None:
            self.create3CardsScene(bg,img1, class1, hulla1,hullb1,img2, class2,hulla2,hullb2,img3, class3,hulla3,hullb3)
        else:
            self.create2CardsScene(bg,img1, class1, hulla1,hullb1,img2, class2,hulla2,hullb2)

    def create2CardsScene(self,bg,img1, class1, hulla1,hullb1,img2, class2,hulla2,hullb2):#这里的img就是card
        #kpsa1是img1的左上角的kps，其他同理
        kpsa1=hull_to_kps(hulla1)
        kpsb1=hull_to_kps(hullb1)
        kpsa2=hull_to_kps(hulla2)
        kpsb2=hull_to_kps(hullb2)
        
        # 对传入的img1进行随机变换
        self.img1=np.zeros((imgH,imgW,4),dtype=np.uint8)
        self.img1[decalY:decalY+cardH,decalX:decalX+cardW,:]=img1
        self.img1,self.lkps1,self.bbs1=augment(self.img1,[cardKP,kpsa1,kpsb1],transform_1card)

        # 对img2进行随机变换.
        # 如果img2经过变换后遮住的img1的一角，则重新变换
        while True:
            self.listbba=[]
            #对img2进行随机变换
            self.img2=np.zeros((imgH,imgW,4),dtype=np.uint8)
            self.img2[decalY:decalY+cardH,decalX:decalX+cardW,:]=img2
            self.img2,self.lkps2,self.bbs2=augment(self.img2,[cardKP,kpsa2,kpsb2],transform_1card)
            
            #将img2变换之后的整个card区域的kps转换为polygon
            mainPoly2=kps_to_polygon(self.lkps2[0].keypoints[0:4])
            invalid=False
            intersect_ratio=0.1#交叉比率阈值
            for i in range(1,3):
                # 将img1的左上角框定区域变换后转化为polygon
                smallPoly1=kps_to_polygon(self.lkps1[i].keypoints[:])# 这里不取0是因为lkps[0]是cardKp变换后的结果
                a=smallPoly1.area # 计算该区域面积
                # 计算img2的polygon与img1的框定polygon的交集
                intersect=mainPoly2.intersection(smallPoly1)
                ai=intersect.area # 计算交集的面积，表示的是被覆盖掉的面积
                if (a-ai)/a > 1-intersect_ratio: # 面积达到一定值，则break重新进行变换
                    self.listbba.append(BBA(self.bbs1[i-1],class1))
                # If intersectio area is not small, but also not big enough, we want apply new transform to card 2
                elif (a-ai)/a>intersect_ratio:
                    invalid=True
                    break
                    
            if not invalid: break
            
        self.class1=class1# 第一张牌的class
        self.class2=class2# 第二张牌的class
        for bb in self.bbs2: # 对第二张牌的四个关键点遍历
            self.listbba.append(BBA(bb,class2))
        
        self.bg=scaleBg.augment_image(bg)# 执行背景变换
        mask1=self.img1[:,:,3] #alpha层 img1 除了有牌的部分其他的全为0，包括alpha层
        self.mask1=np.stack([mask1]*3,-1)# 堆三个alpha层出来w*h*channel(3)
        self.final=np.where(self.mask1,self.img1[:,:,0:3],self.bg)# np.where 在mask1中非0位置从前景图中获取，0位置从bg中获取
        mask2=self.img2[:,:,3]# 下面同理
        self.mask2=np.stack([mask2]*3,-1)
        self.final=np.where(self.mask2,self.img2[:,:,0:3],self.final)
        
        
        
    def create3CardsScene(self,bg,img1, class1, hulla1,hullb1,img2, class2,hulla2,hullb2,img3, class3,hulla3,hullb3):
        # 创建三张牌的场景，同理
        kpsa1=hull_to_kps(hulla1,decalX3,decalY3)# 第一张牌左上角的keypoints
        kpsb1=hull_to_kps(hullb1,decalX3,decalY3)# 第一张牌右上角的keypoints
        kpsa2=hull_to_kps(hulla2,decalX3,decalY3)# 以此类推
        kpsb2=hull_to_kps(hullb2,decalX3,decalY3)
        kpsa3=hull_to_kps(hulla3,decalX3,decalY3)
        kpsb3=hull_to_kps(hullb3,decalX3,decalY3)
        self.img3=np.zeros((imgH,imgW,4),dtype=np.uint8)
        self.img3[decalY3:decalY3+cardH,decalX3:decalX3+cardW,:]=img3
        self.img3,self.lkps3,self.bbs3=augment(self.img3,[cardKP,kpsa3,kpsb3],trans_rot1)
        self.img2=np.zeros((imgH,imgW,4),dtype=np.uint8)
        self.img2[decalY3:decalY3+cardH,decalX3:decalX3+cardW,:]=img2
        self.img2,self.lkps2,self.bbs2=augment(self.img2,[cardKP,kpsa2,kpsb2],trans_rot2)
        self.img1=np.zeros((imgH,imgW,4),dtype=np.uint8)
        self.img1[decalY3:decalY3+cardH,decalX3:decalX3+cardW,:]=img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img3,_lkps3,self.bbs3=augment(self.img3,self.lkps3,det_transform_3cards, False)
            if _img3 is None: continue
            _img2,_lkps2,self.bbs2=augment(self.img2,self.lkps2,det_transform_3cards, False)
            if _img2 is None: continue
            _img1,self.lkps1,self.bbs1=augment(self.img1,[cardKP,kpsa1,kpsb1],det_transform_3cards, False)
            if _img1 is None: continue
            break
        self.img3=_img3
        self.lkps3=_lkps3
        self.img2=_img2
        self.lkps2=_lkps2
        self.img1=_img1
        
        self.class1=class1
        self.class2=class2
        self.class3=class3
        self.listbba=[BBA(self.bbs1[0],class1),BBA(self.bbs2[0],class2),BBA(self.bbs3[0],class3),BBA(self.bbs3[1],class3)]
        
        # bg, img1, img2 and img3
        self.bg=scaleBg.augment_image(bg)# 背景变换
        mask1=self.img1[:,:,3]# img1的alpha层，除了牌部分其他全为0
        self.mask1=np.stack([mask1]*3,-1)# mask1*3堆叠，RGB全为mask
        self.final=np.where(self.mask1,self.img1[:,:,0:3],self.bg)#final图片mask1有值就是img1，否则就用bg
        mask2=self.img2[:,:,3]
        self.mask2=np.stack([mask2]*3,-1)
        self.final=np.where(self.mask2,self.img2[:,:,0:3],self.final)
        mask3=self.img3[:,:,3]
        self.mask3=np.stack([mask3]*3,-1)
        self.final=np.where(self.mask3,self.img3[:,:,0:3],self.final)

    def display(self):
        fig,ax=plt.subplots(1,figsize=(8,8))
        ax.imshow(self.final)# 显示变换后的图片
        for bb in self.listbba:# 每个bbox
            rect=patches.Rectangle((bb.x1,bb.y1),bb.x2-bb.x1,bb.y2-bb.y1,linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)# 在ax里面画出来
    def res(self):
        return self.final# 返回变换后的图片
    def write_files(self,save_dir,display=False):
        jpg_fn, xml_fn=give_me_filename(save_dir, ["jpg","xml"])# 保存
        plt.imsave(jpg_fn,self.final)
        if display: print("New image saved in",jpg_fn)
        create_voc_xml(xml_fn,jpg_fn, self.listbba,display=display)




if __name__ == '__init__':
    backgrounds = Backgrounds("/home/xueaoru/视频/backgrounds.pck")
    _=backgrounds.get_random(display=True)
    cards = Cards("/home/xueaoru/视频/cards.pck")
    nb_cards_to_generate=200
    save_dir="/media/xueaoru/DATA/ubuntu/cards/test_dataset"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for i in tqdm(range(nb_cards_to_generate)):
        bg=backgrounds.get_random()
        img1,card_val1,hulla1,hullb1=cards.get_random()
        img2,card_val2,hulla2,hullb2=cards.get_random() 
        newimg=Scene(bg,img1,card_val1,hulla1,hullb1,img2,card_val2,hulla2,hullb2)
        newimg.write_files(save_dir)

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def display_img(img,polygons=[],channels="bgr",size=9):
    """
        函数的作用是显示图像并且可选画出bounding box或者convex
        使用参数channels来控制是不是opencv中得到的图像
    """
    if not isinstance(polygons,list):
        polygons=[polygons]    #
    if channels=="bgr": # bgr (cv2 image)
        nb_channels=img.shape[2]
        if nb_channels==4:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA)#转为opencv中的图像格式
        else:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   #转为opencv中的图像格式 
    fig,ax=plt.subplots(figsize=(size,size))
    #给定参数size，单位是cm，建立一个fig对象，建立一个axis对象,一个fig对象包含若干个axis小图像
    ax.set_facecolor((0,0,0))#设置背景颜色，颜色一般用字符串表示如 '#00FFCC'
    ax.imshow(img)# 显示img图像
    for polygon in polygons:#遍历需要展示的polygon
        #(n,2), 
        #或者(n,1,2) 这是cv2的结构.
        #reshape成(n,2)
        if len(polygon.shape)==3:
            polygon=polygon.reshape(-1,2)
        patch=patches.Polygon(polygon,linewidth=1,edgecolor='g',facecolor='none')
        #调用patches画多边形得到子图
        ax.add_patch(patch)
        #在ax图中添加patch子图

def getAlphamask(bord_size=2,card_size,show = False): # 线宽
    cardH,cardW = card_size
    alphamask=np.ones((cardH,cardW),dtype=np.uint8)*255 #全白的图像mask
    cv2.rectangle(alphamask,(0,0),(cardW-1,cardH-1),0,bord_size)#黑色举行画满mask
    cv2.line(alphamask,(bord_size*3,0),(0,bord_size*3),0,bord_size)# 画一条斜线，就是四个角的斜线
    cv2.line(alphamask,(cardW-bord_size*3,0),(cardW,bord_size*3),0,bord_size)
    cv2.line(alphamask,(0,cardH-bord_size*3),(bord_size*3,cardH),0,bord_size)
    cv2.line(alphamask,(cardW-bord_size*3,cardH),(cardW,cardH-bord_size*3),0,bord_size)
    if show:
        plt.figure(figsize=(10,10))#10cm高 10cm宽的 画板
        plt.imshow(alphamask) # 在这个画板上显示alphamask图像
    return alphamask
def varianceOfLaplacian(img):
    """
    判断图像模糊度
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()

def extract_card (img, output_fn=None, min_focus=120):
    """
    给定图片，将card从图片中提取出来，保存到output_fn
    """
    
    imgwarp=None
    
    # 检查图像的模糊度
    focus=varianceOfLaplacian(img)
    print(focus)
    if focus < min_focus: 
        print("太模糊:", focus)
        return False,None
    
    # 变成灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # 双边滤波降噪
    gray=cv2.bilateralFilter(gray,11,17,17)
    
    # Canny边缘检测
    edge=cv2.Canny(gray,30,200)
    
    # 在canny之后的图像中寻找Contour
    _,cnts, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #我们假定找到的轮廓具有最大确定卡片的轮廓，就是想找卡片的contour
    cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    rect=cv2.minAreaRect(cnt)#得到cnt的最小外接矩形
    box=cv2.boxPoints(rect)# 得到外界举行的四个点的坐标
    box=np.int0(box)#将坐标转换为int0
    areaCnt=cv2.contourArea(cnt)#计算cnt的面积
    areaBox=cv2.contourArea(box)#计算外接矩形的面积
    valid=areaCnt/areaBox>0.95#计算面积比
    
    if valid:#如果面积比达到阈值
        if wr>hr:
            Mp=cv2.getPerspectiveTransform(np.float32(box),refCard)
        else:
            Mp=cv2.getPerspectiveTransform(np.float32(box),refCardRot)
        # 利用透视变换将源图片里的card变换到实际图像zoom后的图像的位置对应的点的坐标（铺满整个画布）
        imgwarp=cv2.warpPerspective(img,Mp,(cardW,cardH))# 之所以需要cardW和cardH是因为需要变换后的图像的size
        # 添加一个Alpha层，A
        imgwarp=cv2.cvtColor(imgwarp,cv2.COLOR_BGR2BGRA)
        
        # cnt的Shape(n,1,2), type=int,n是cnt数量
        # cnt的shape是(n,1,2),type = int,我们需要将其reshape成(1,n,2),contour记录的只是坐标
        cnta=cnt.reshape(1,-1,2).astype(np.float32)# 变成(1,n,2)是因为opencv里面变换函数要求的shape是这样
        # cnta应用Mp变换
        cntwarp=cv2.perspectiveTransform(cnta,Mp)
        cntwarp=cntwarp.astype(np.int)
        
        # 先将alpha通道的图像清空（透明），单层channel
        alphachannel=np.zeros(imgwarp.shape[:2],dtype=np.uint8)
        # 在alpha通道的cntwarp区域画出它
        cv2.drawContours(alphachannel,cntwarp,0,255,-1) # thickness = -1，直接封闭画(填充)
        # 拿到只有一个channel的层，四角剪掉
        alphamask = getAlphamask(bord_size=2,card_size=(57*4,87*4),show = False)
        # and操作是0就不添加，非0就覆盖掉
        alphachannel=cv2.bitwise_and(alphachannel,alphamask)# alpha 只保留了contour
        # bitwise_and的结果只有0和1
        # 添加alpha层到imgwarp
        imgwarp[:,:,3]=alphachannel # 应用到alpha层
        
        # 保存图像
        if output_fn is not None:
            cv2.imwrite(output_fn,imgwarp)

    return valid,imgwarp
def extract_cards_from_video(video_fn, output_dir=None, keep_ratio=5, min_focus=120, debug=False):
    """
        给定video文件，提取card保存到output_dir文件夹里,keep_ratio表示每隔多少帧保存一次
    """
    if not os.path.isfile(video_fn):
        print(f"Video file {video_fn} does not exist !!!")
        return -1,[]
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap=cv2.VideoCapture(video_fn)
    
    frame_nb=0
    imgs_list=[]
    while True:
        ret,img=cap.read()
        if not ret: break
        if frame_nb%keep_ratio==0:
            if output_dir is not None:
                output_fn=give_me_filename(output_dir,"png")
            else:
                output_fn=None
            valid,card_img = extract_card(img,output_fn,min_focus=min_focus,debug=debug)
            if debug: 
                k=cv2.waitKey(1)
                if k==27: break
            if valid:
                imgs_list.append(card_img)
        frame_nb+=1
    
    if debug:
        cap.release()
        cv2.destroyAllWindows()
    
    return imgs_list

def findHull(img, corner, debug="no"):#corner指的是花色和牌值所在的位置的四个点的坐标
    """
        给定img得到凸包点集
    """
    
    kernel = np.ones((3,3),np.uint8)# 定义一个图像膨胀所需要的kernel大小（3,3）
    corner=corner.astype(np.int)# 将参数corner转化为int类型操作

    # 下面获取到corner的实际图像zoom后的坐标
    x1=int(corner[0][0])
    y1=int(corner[0][1])
    x2=int(corner[2][0])
    y2=int(corner[2][1])
    w=x2-x1# zoom后的实际宽度
    h=y2-y1# zoom后的实际高度
    zone=img[y1:y2,x1:x2].copy()#copy这个区域到zone
    strange_cnt=np.zeros_like(zone)#返回一个和zone shape一样的全0对象
    gray=cv2.cvtColor(zone,cv2.COLOR_BGR2GRAY)# 将zone转化为黑白图像
    thld=cv2.Canny(gray,30,200)# Canny找轮廓
    thld = cv2.dilate(thld,kernel,iterations=1)# 对轮廓进行膨胀操作，迭代次数为1，目的使得原contour图像线条更浓
    if debug!="no": cv2.imshow("thld",thld)
    
    # 寻找contours
    _,contours,_=cv2.findContours(thld.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    min_area=30 # 最小接收区域面积
    min_solidity=0.3 # 最小接收面积比
    
    concat_contour=None # 通过这个参数将所有的contour添加成一个contour，扩大范围
    
    ok = True
    for c in contours:
        area=cv2.contourArea(c)# 计算该contour的面积，值更小
        hull = cv2.convexHull(c)# 得到contour的凸包,凸包一般包含的比contour的区域要多一些
        hull_area = cv2.contourArea(hull)#计算凸包的面积（一般面积更大）
        solidity = float(area)/hull_area#计算一下面积比，应该是一个(0,1]的数
        M=cv2.moments(c)# 计算轮廓的矩
        cx=int(M['m10']/M['m00'])# 计算图像的重心x坐标
        cy=int(M['m01']/M['m00'])# 计算图像重心y坐标
        #  abs(w/2-cx)<w*0.3 and abs(h/2-cy)<h*0.4 : 
        if area >= min_area and abs(w/2-cx)<w*0.3 and abs(h/2-cy)<h*0.4 and solidity>min_solidity:# 满足实验测试的条件
            if debug != "no" :
                cv2.drawContours(zone,[c],0,(255,0,0),-1)
            if concat_contour is None:
                concat_contour=c# 如果是None就直接是contour本身
            else:
                concat_contour=np.concatenate((concat_contour,c))#经过上一步他已经不是None了，所以要添加上去
        if debug != "no" and solidity <= min_solidity :
            print("面积比",solidity)
            cv2.drawContours(strange_cnt,[c],0,255,2)
            cv2.imshow("contours",strange_cnt)
            
     
    if concat_contour is not None:# 如果说找到了整体的contour
        #通过合并后的contour把hull给确定出来
        hull=cv2.convexHull(concat_contour)
        hull_area=cv2.contourArea(hull)# 计算hull的面积
        # 如果hull的面积太小或者太大，那都是有问题的，因此设置好实验阈值
        min_hull_area=940 
        max_hull_area=2120 
        if hull_area < min_hull_area or hull_area > max_hull_area: 
            ok=False
            if debug!="no":
                print("Hull area=",hull_area,"too large or too small")
        #我们上面得到的hull的坐标是在小矩形框图片也就是ROI图像中的坐标
        #现在我们需要把这个坐标转换到实际图像zoom后的图像当中去
        hull_in_img=hull+corner[0] #也就是 [cornerXmin,cornerYmin]
        # 其实很简单，hull中每个x坐标加cornerXmin,每个y坐标加cornerYmin

    else:
        ok=False
    
    
    if debug != "no" :
        if concat_contour is not None:
            cv2.drawContours(zone,[hull],0,(0,255,0),1)
            cv2.drawContours(img,[hull_in_img],0,(0,255,0),1)
        cv2.imshow("Zone",zone)
        cv2.imshow("Image",img)
        if ok and debug!="pause_always":
            key=cv2.waitKey(1)
        else:
            key=cv2.waitKey(0)
        if key==27:
            return None
    if ok == False:
        
        return None
    
    return hull_in_img

        
if __name__=='__main__':
    extract_cards_from_video("/home/xueaoru/视频/f1.avi","/home/xueaoru/视频/output")

import numpy as np
import cv2

#Haar Detector to extract features
def compute_rect(img,rec):
    point1=img[rec[0],rec[1]]
    point2=img[rec[0],(rec[1]+rec[2])]
    point3=img[(rec[0]+rec[3]),rec[1]]
    point4=img[(rec[0]+rec[3]),(rec[1]+rec[2])]
    return (point4+point1-point2-point3)
       
def Haar_detector(img,N_features):
    
    rows,cols,s =img.shape
    #color to grey
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grey=grey.astype('int')
    #integral image
    integral_img=np.cumsum(np.cumsum(grey,axis=0),axis=1)
    #compute feature
    feature=np.zeros(N_features,dtype='int')
    r=int(rows/2)
    c=int(cols/2)
    cnt=0
    for h in range(rows):
        for w in range (c):
            for i in range(0,rows-h):
                for j in range(0,cols-2*w-1):
                    rec1=[i,j,w,h]
                    rec2=[i,j+w,w,h]
                    feature[cnt]= compute_rect(grey,rec2)-compute_rect(grey,rec1)
                    cnt = cnt+1
                    
    for h in range(r):
        for w in range(cols):
            for i in range(0,rows-2*h-1):
                for j in range(0,cols-w):
                    rec1=[i,j,w,h]
                    rec2=[i+h,j,w,h]
                    feature[cnt]= compute_rect(grey,rec1)-compute_rect(grey,rec2)
                    cnt = cnt+1
    return feature

#load Data
filepath='ECE661_2016_hw11_DB2/'
category1='train/'               
category2='test/'
sub_category=['positive/','negative/']

#set params
N_pos=710
N_neg=1758
N_features=166000

#load training data
features_train=np.zeros((N_features,(N_pos+N_neg)),dtype=int)
count=0

#compute feature matrix
for sub in sub_category:
    if sub=='positive/':
        N_image_start=0
        N_image_stop=710
    else:
        N_image_start=0
        N_image_stop=1758
    for i in range(N_image_start,N_image_stop):
        
        if i+1<10:
            str_tmp='000'+str(i+1)
        elif i+1>=10 and i+1<100:
            str_tmp='00'+str(i+1)
        elif i+1>=100 and i+1<1000:
            str_tmp='0'+str(i+1)
        else:
            str_tmp=str(i+1)
        img=cv2.imread(filepath+category1+sub+'00'+str_tmp+'.png')
        feature=Haar_detector(img,N_features)
        
        features_train[:,count]=feature
        
        count +=1

np.savetxt('train/features_train.txt', features_train,fmt='%i')


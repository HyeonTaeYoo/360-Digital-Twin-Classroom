import glob
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import hog
import math
import predict_mask

# name='test/images/jinho.jpg'

def model_predict(name):   # name은 이미지 파일
    img = cv2.imread(name)
    mask = predict_mask.predict(img)
    dst = predict_mask.transfer(img, mask)
    cv2.imwrite(name.replace('images', 'outs'), mask)

    hog_db_male=np.loadtxt('output_male.txt',delimiter=' ')
    hog_db_female=np.loadtxt('output_female.txt',delimiter=' ')

    
    print("hog rows, cols{}".format(hog_db_male.shape))
    hog_db_rows, hog_db_cols=hog_db_male.shape
    mask_db=glob.glob('Mask_DB/*')
    result=np.zeros(54)    
    masks = name
    print("img: {}".format(masks))
    hog_feat_des = np.zeros( 28 * 28 * 8)
    mask=plt.imread(masks)
    #print(mask.shape)
    bins = 8
    pixels_per_cell = (8, 8)
    hog_value,hog_image = hog.hog(mask, bins, pixels_per_cell)
    print(hog_value.shape, hog_image.shape)
    hog_feat_des[ :] = hog_value.ravel()
  
    print(hog_feat_des.shape)
    r_max=0
    r_min=0
    c=0
    c_max=0
        
        
    rows, cols=mask.shape
    for i in range(0,rows):
        if c>c_max:
            c_max=c
        c=0
        for j in range(cols):
            if mask[i][j]!=0:
                c+=1
        
        for i in range(0,cols):
            for j in range(0,rows):
                if mask[j][i]!=0:
                    if j>r_max:
                        r_max=j
                    if j<r_min:
                        r_min=j
        
    if float(r_max/c_max)>0 and float(r_max/c_max)<1.3:
        for i in range(0,hog_db_rows):
            d=0.0000000
            a=0
            for j in range(0,hog_db_cols):
                a+=hog_db_male[i][j]    
            #print("i: {} , a: {} ".format(i,a)) 
            if a!=0:
                for j in range(0,hog_db_cols):
                    #print("현재 이미지 벡터{}".format(hog_feat_des[j]))
                    #print("hog 배열 값{}".format(hog_db[i][j]))
                    d+=pow(hog_feat_des[j] - hog_db_male[i][j],2)
                    result[i]=math.sqrt(d)                
            else:
                result[i]=9999999
            #print(result[i])            

                
                    
    else:
        for i in range(0,hog_db_rows):
            d=0.0000000
            b=0
            for j in range(0,hog_db_cols):
                b+=hog_db_female[i][j]
            #print("b: {} ".format(b))    
            if b!=0:
                for j in range(0,hog_db_cols):
                    #print("현재 이미지 벡터{}".format(hog_feat_des[j]))
                    #print("hog 배열 값{}".format(hog_db[i][j]))
                    d+=pow(hog_feat_des[j] - hog_db_female[i][j],2)
                    result[i]=math.sqrt(d)
            else:
                result[i]=9999999
            
            #print(result[i])


    hair_index=np.argmin(result)
    hair_name=mask_db[hair_index]
    hair_name=hair_name.replace('Mask_DB\\', '').replace('.JPG','')
    print(hair_name)
    return hair_name

# name='test/images/jinho.jpg'
# model_predict(name)
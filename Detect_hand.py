import cv2 as cv
import numpy as np
import os
from mtcnn import MTCNN
from keras.models import Model, load_model
from PIL import Image
import math
from collections import deque

#from shapely.geometry import Polygon

detector = MTCNN()
targetX, targetY = 100, 100

def square_face(img, box_x, box_y, box_width, box_height):
    try:
        face_center_x = box_x + int(1/2 * box_width)
        face_center_y = box_y + int(1/2 * box_height)
        a = int(1/2 * box_width) if box_width > box_height else int(1/2 * box_height)
        # 최대한 얼굴 형태 유지하며 정방형으로 잘라냄
        face = img[face_center_y - a : face_center_y + a, face_center_x - a : face_center_x + a]
        # 얼굴 (100, 100)로 만듦
        face = cv.resize(face, dsize=(targetX, targetY))

        return face

    except:
        return None


box_x=0
box_y=0
box_width =0
box_heigth =0

def removeFace(img): # 한 img에서 얼굴과 그 얼굴의 신원을 파악하는 코드
    face_infos = detector.detect_faces(img)
    #print(len(face_infos)) # 발견된 얼굴의 수
    global box_x, box_y,box_width, box_heigth
    for face_info in face_infos:
        box = face_info['box']
        box_x, box_y = box[0], box[1]
        box_width, box_heigth = box[2], box[3]
        
        # 얼굴 중심을 기준으로 정방형으로 얼굴 추출
        face = square_face(img, box_x, box_y, box_width, box_heigth)

        if face is None:
            continue # 얼굴이 아닌 경우에는 그냥 지나침

        # # 얼굴에 바운더리 박스 그리기
        img = cv.rectangle(img, (box_x-10, box_y), (box_x + box_width+10, box_y + box_heigth+30), (0, 0, 0), -1) #얼굴 가리개 
            
        
        # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY) # RGB2GRAY로 해야할까 BGR2GRAY로 해야할까...
        # # 박스 위에 이름과 정확도 출력하기
        # img = cv2.putText(
        #                 img, name + ' '+str(acc), (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return img, box_x, box_y, box_width, box_heigth


def make_mask_image(img):

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #img_h,img_s,img_v = cv.split(img_hsv)

    low = (0, 30, 0)
    high = (15, 255, 255)

    img_mask = cv.inRange(img_hsv, low, high)
    return img_mask



def findMaxArea(contours,box_x,box_y,width,height):

    max_contour = None
    max_area = -1

    for contour in contours:
        
        x, y, w, h = cv.boundingRect(contour)
        area =0
        if box_x-(width*2) <= x and box_y <= y and (box_x)-50 >=x and (box_y +height) >=y :
            area = cv.contourArea(contour)


        if (w*h)*0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 10:
        max_area = -1

    return max_area, max_contour




def process(img_bgr):

    img_result = img_bgr.copy()

    # STEP 1
    img_bgr, box_x, box_y, w, h = removeFace(img_bgr)

    # STEP 2
    img_binary = make_mask_image(img_bgr)

    # STEP 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
    cv.imshow("Binary", img_binary)

    # STEP 4
    contours, hierarchy = cv.findContours(
        img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # for cnt in contours:
    #     cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)
            #cv.drawContours(img_binary, [cnt], 0, (255, 0, 0), 3)

    # STEP 5
    img_result = cv.rectangle(img_result, (box_x- 2*w, box_y), (box_x-50, box_y + box_heigth), (0, 255, 0), 2) #손들 공간 표시9
    max_area, max_contour = findMaxArea(contours,box_x,box_y,w,h)

    if max_area == -1:
        return img_result
    

    

    if max_area >=100 :
        cv.putText(img_result,"Hand up",(box_x- w,box_y),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA )
        print("")
    print(max_area)
    cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

    # # STEP 6
    # ret, points = getFingerPosition(max_contour, img_result, debug)

    # # STEP 7
    # if ret > 0 and len(points) > 0:
    #     for point in points:
    #         cv.circle(img_result, point, 20, [255, 0, 255], 5)

    return img_result


# cap = cv.VideoCapture('test.avi')

cap = cv.VideoCapture(0)

while True:

    ret, img_bgr = cap.read()

    if ret == False:
        break

    img_result = process(img_bgr)

    key = cv.waitKey(1)
    if key == 27:
        break

    cv.imshow("Result", img_result)


cap.release()
cv.destroyAllWindows()

import cv2 as cv
import numpy as np
import os
from mtcnn import MTCNN
from keras.models import Model, load_model
from PIL import Image
import math
from collections import deque
import datetime

detector = MTCNN()
targetX, targetY = 224, 224


def square_face(img, box_x, box_y, box_width, box_height):
    try:
        face_center_x = box_x + int(1/2 * box_width)
        face_center_y = box_y + int(1/2 * box_height)
        a = int(1/2 * box_width) if box_width > box_height else int(1/2 * box_height)
        # 최대한 얼굴 형태 유지하며 정방형으로 잘라냄
        face = img[face_center_y - a: face_center_y +
                   a, face_center_x - a: face_center_x + a]
        # 얼굴 (100, 100)로 만듦
        face = cv.resize(face, dsize=(targetX, targetY))

        return face

    except:
        return None


def cropFace(img):  # 한 img에서 얼굴과 그 얼굴의 신원을 파악하는 코드
    face_infos = detector.detect_faces(img)

    for face_info in face_infos:
        box = face_info['box']
        box_x, box_y = box[0], box[1]
        box_width, box_heigth = box[2], box[3]

        # 얼굴 중심을 기준으로 정방형으로 얼굴 추출
        face2 = square_face(img, box_x, box_y, box_width, box_heigth)
        # img = img[box_y:box_y+box_heigth ,box_x:box_x+box_width ]

        return face2
        # return None # 얼굴이 아닌 경우에는 그냥 지나침

        # # 얼굴에 바운더리 박스 그리기


nowDate = datetime.datetime.now()

cap = cv.VideoCapture(0)
cnt = 0
while True:

    ret, img = cap.read()

    if ret == False:
        break

    img2 = cropFace(img)

    if img2 is None:
        continue

    cv.imshow("Result", img2)

    key = cv.waitKey(1)
    if key == 97:
        cnt = cnt+1
        print(cnt)
        cv.imwrite("AIAS/yooseong/" +
                   nowDate.strftime('%Y-%m-%d %H %M')+"_"+str(cnt)+".png", img2)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()

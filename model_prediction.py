import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model, load_model
from functools import partial
from inception_resnet_v1_lcl import *

# 학습 시켜둔 모델
model = load_model('0430_100_5p_fromVideo.hdf5')
categories = ["hyeontae", "jaehyeon", "jinho", "joohyeong", "yoosung"]

# 학습 당시의 데이터(이미지) 사이즈
targetx = 100
targety = 100


def predict_by_model(img, categories):
    img = img.convert("RGB")  # 모델 학습 당시 PIL을 이용하였기 때문에 OPENCV 이미지의 채널 변경
    img = img.resize((targetx, targety))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float") / 256
    X = X.reshape(-1, targetx, targety, 3)
    pred = model.predict(X)  # 모델을 통한 예측
    result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환

    est = categories[result[0]]  # 예측 결과(사람 이름)
    #acc = round(max(pred[0][0], pred[0][1], pred[0][2]) * 100, 2)
    acc = round(max(pred[0][0], pred[0][1], pred[0]
                    [2], pred[0][3], pred[0][4]) * 100, 2)  # 예측 정확도? 퍼센테이지

    return est, acc


# if __name__ == '__main__':
#     categories = ["hyeontae", "jinho", "yoosung"]

#     for i in image_path:
#         img = Image.open(i)
#         est, acc = predict_by_model(img, categories)
#         print(est, acc)

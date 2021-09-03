import glob
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import hog
import math



def predict(image, height=224, width=224):
    model = keras.models.load_model('checkpoints/checkpoint.hdf5')
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    
    pred = model.predict(im)
    
    mask = pred.reshape((224, 224))

    return mask  # 넣은 사진의 머리 이진화


def transfer(image, mask):
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask

    alpha = 0.8
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return dst


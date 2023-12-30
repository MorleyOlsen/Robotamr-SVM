#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import read_MNIST
import numpy as np
from sklearn.externals import joblib
model = joblib.load('./model_data/md.model')

def imgBinaryzation(imgs):
    for i in range(len(imgs)):
        cv_img = imgs[i].astype(np.uint8)
        cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
        imgs[i] = cv_img
    return imgs


def model_acc():
    """
    判断模型准确率的函数
    :return:
    """
    test_imgs = read_MNIST.loadImageSet('test')
    test_labels = read_MNIST.loadLabelSet('test')
    test_imgs = imgBinaryzation(test_imgs)
    print("---------", test_imgs[0])
    test_labels = test_labels.reshape((-1))
    test_result = model.predict(test_imgs)
    precision = sum(test_result == test_labels)/test_labels.shape[0]
    print('The accuracy socre is: ', precision)


def predict(img):
    """
    预测函数
    :return: 识别结果
    """
    test_result = model.predict(img)
    # print(test_result)
    return test_result[0]


# 图像
def img_input(img):
    img = img.reshape(-1)
    recognize_result = predict([img])
    return recognize_result

# model_acc()

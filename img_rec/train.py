#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import read_MNIST
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

train_imgs = read_MNIST.loadImageSet('train')[:]
train_labels = read_MNIST.loadLabelSet('train')[:]


for i in range(len(train_imgs)):
    cv_img = train_imgs[i].astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    train_imgs[i] = cv_img

# Training SVM
print('------Training SVM------')
clf = svm.SVC(C=5, gamma=0.05, max_iter=10)
model = clf.fit(train_imgs, train_labels)
joblib.dump(model, './model_data/md.model')

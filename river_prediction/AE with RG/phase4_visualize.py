# 0518 ae_regressor test (not bad but not so good, too)

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import cv2
import random
import network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import multiprocessing as mp

def make_data_dict(train_data):
    data_dict = dict()
    for i in range(200):
        data_dict[i] = train_data[:,1150*i:1150*(i+1)]
    return data_dict

def main():

    y_test  = np.load('ys_data.npy')[-30:,:230000] # 30 x 230000

    total_pred_dict = np.load('total_pred_dict.npy', allow_pickle=True)
    test_pic = y_test[15]
    pred_pic = total_pred_dict.item().get(15)

    naive_error = abs(test_pic - pred_pic)
    print(naive_error.sum())

    test_pic = np.reshape(test_pic, (500, 460))
    pred_pic = np.reshape(pred_pic, (500, 460))

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.namedWindow("pred", cv2.WINDOW_NORMAL)
    cv2.imshow('label', test_pic)
    cv2.imshow('pred', pred_pic)
    cv2.waitKey(0)
    cv2.destroyWindow()

if __name__ == '__main__':
    main()

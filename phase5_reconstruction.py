import os
import sys
import time
import torch
import pickle
import numpy as np
import pandas as pd
import cv2

from PIL import Image
from copy import deepcopy
from sklearn.model_selection import train_test_split, ShuffleSplit
import matplotlib.pyplot as plt

def return_river_data(folder=None, file_name=None, skiprows=6):
    path = os.path.join(folder, file_name)
    grid = np.array(Image.open(path))   # tiff size : 4704 x 4448
    grid[np.where(grid < 0)]  = 0       # -9999 -> 0
    return grid

def gen_valid_dataset(low = 0, high = 690, ext='tif'):

    # Union valid index - 1번이라도 침수된 적이 있는 grid only selects
    valid_index = np.loadtxt('data/union_valid_index.txt').astype(int)  # 2 x 231432
    valid_index = (valid_index[0][:230000], valid_index[1][:230000])  # 2 x 231432 - (x, y)

    folder_path     = '690_tifs/TIFF/'
    total_pred_dict = np.load('total_pred_dict.npy', allow_pickle=True)
    for i in range(660, high): # test data set 660 ~ 690

        pic_path = 'Depth (C{}).DEM_E5186_3m_with_FM.'.format(i + 1) + ext
        data      = return_river_data(folder_path, pic_path)

        data_pred1              = deepcopy(data)*0
        data_pred2              = deepcopy(data_pred1) * 0
        data_pred2[valid_index] = total_pred_dict.item().get(i-660)

        cv2.namedWindow('pic1', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('pic2', cv2.WINDOW_NORMAL)
        cv2.namedWindow('pic3', cv2.WINDOW_NORMAL)
        cv2.imshow('pic1', data)
        #cv2.imshow('pic2', data_pred1)
        cv2.imshow('pic3', data_pred2)
        cv2.waitKey(0)
        cv2.destroyWindow()


if __name__ == "__main__":
    gen_valid_dataset()


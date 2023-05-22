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

import utils
import logging


def paint(label, pred, idx):
    fig, axs = plt.subplots(1, 2)
    cmap = plt.cm.plasma
    cmap.set_under(color='black')  # set the color out of the range as black
    im1 = axs[0].imshow(label, vmin=0.000001, vmax=3, cmap=cmap)
    im2 = axs[1].imshow(pred, vmin=0.000001, vmax=3, cmap=cmap)
    axs[0].set_title('label')
    axs[1].set_title('pred')
    fig.colorbar(im1, ax=axs, orientation='vertical')
    plt.axis('on')
    #plt.show()

    plt.savefig(str(idx+1) + "_interpolation.png")

def preprocessing(folder=None, file_name=None, skiprows=6):
    path = os.path.join(folder, file_name)
    grid = np.array(Image.open(path))   # tiff size : 4704 x 4448
    grid[np.where(grid < 0)]  = 0       # -9999 -> 0

    return grid

def reconstruction(low = 0, high = 690, ext='tif'):

    # Union valid index - 1번이라도 침수된 적이 있는 grid only selects
    v_idx = np.loadtxt('data/union_valid_index.txt').astype(int)  # 2 x 231432
    v_idx = (v_idx[0], v_idx[1])  # 2 x 231432 - (x, y)

    inter_range     = 50
    folder_path     = '690_tifs/TIFF/'

    total_pred_dict = np.load('total_pred_dict_0521.npy', allow_pickle=True)
    pic_path = 'Depth (C{}).DEM_E5186_3m_with_FM.'.format(681) + ext
    label = preprocessing(folder_path, pic_path)
    pred = deepcopy(label) * 0
    pred[v_idx] = total_pred_dict.item().get(20)
    paint(label, pred, -1)

    pic_path = 'Depth (C{}).DEM_E5186_3m_with_FM.'.format(690) + ext
    label = preprocessing(folder_path, pic_path)
    pred = deepcopy(label) * 0
    pred[v_idx] = total_pred_dict.item().get(29)
    paint(label, pred, 49)

    interpolation = np.load('interpolation_0521.npy', allow_pickle=True)
    for i in range(0, inter_range-1): # test data set 660 ~ 690
        pred[v_idx] = interpolation.item().get(i)
        pred[np.where(pred < 0)] = 0
        paint(label, pred, i)

if __name__ == "__main__":
    reconstruction()


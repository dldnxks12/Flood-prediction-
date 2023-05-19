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

def paint(label, pred):
    fig, axs = plt.subplots(1, 2)
    cmap = plt.cm.plasma
    cmap.set_under(color='black')  # set the color out of the range as black
    im1 = axs[0].imshow(label, vmin=0.000001, vmax=3, cmap=cmap)
    im2 = axs[1].imshow(pred, vmin=0.000001, vmax=3, cmap=cmap)
    axs[0].set_title('label')
    axs[1].set_title('pred')
    fig.colorbar(im1, ax=axs, orientation='vertical')
    plt.axis('on')
    plt.show()

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
        label      = return_river_data(folder_path, pic_path)

        pred              = deepcopy(label)*0
        pred[valid_index] = total_pred_dict.item().get(i-660)
        pred[np.where(pred < 0)] = 0

        paint(label, pred)


if __name__ == "__main__":
    gen_valid_dataset()


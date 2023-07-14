import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
# import tensorflow
import time
import pickle
import torch
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
from skimage import io
import cv2

def return_river_data(folder=None, file_name=None, crop=False):
    path = os.path.join(folder, file_name)
    grid = io.imread(path)
    #grid = np.array(Image.open(path))

    grid[np.where(grid < 0)] = 0  # 2313 x 1991

    if not crop:
        return grid
    else:
        return grid[:4096, 352:]

def gen_valid_dataset(low, high , crop=False, ext='tif'):
    valid_index = np.loadtxt('union_valid_index.txt').astype(int)
    valid_index = (valid_index[0], valid_index[1])

    ys = np.empty((0, len(valid_index[0])), float)
    folder_name = 'Depth1'
    for i in range(low, high+1):
        file_name = 'Depth (PF {}).DEM_5m.'.format(i) + ext
        data = return_river_data(folder_name, file_name, crop=crop)
        data = data[valid_index]

        ys = np.append(ys, np.expand_dims(data, axis=0), axis=0)

    return ys

"""
1. 각 파트 별로 예측된 값들 모아서 원본 복구
2. 복구된 예측 값 시각화 (set visualize flag as 'True' in main.py)
"""

import utils
import numpy as np
from copy import deepcopy
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
    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()

import sys
def reconstruction(pred_dict, pred_once, y_data = None, visualize = False):
    valid_x = np.load("data/x_index.npy").astype(int)
    valid_y = np.load("data/y_index.npy").astype(int)

    v_idx = (valid_x, valid_y)   # 2 x 73883 - (x, y)
    grid = np.zeros([782, 992])  # tiff size : 782 x 992

    if not pred_once:
        pred_list = []
        for idx in range(len(pred_dict)):
            pred = deepcopy(grid) * 0
            pred[v_idx] = pred_dict[idx]
            pred[np.where(pred < 0)] = 0
            pred_list.append(pred)
        return np.array(pred_list)

    else:
        pred  = deepcopy(grid)*0
        pred[v_idx] = pred_dict
        pred[np.where(pred < 0)] = 0

        # visualize
        if visualize == True:
            label = deepcopy(grid) * 0
            label[v_idx] = y_data
            label[np.where(label < 0)] = 0
            paint(label, pred)
        else:
            return pred


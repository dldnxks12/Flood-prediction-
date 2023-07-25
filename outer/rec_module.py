import sys
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
    plt.axis('on')
    plt.show()

def reconstruction(pred_dict, y_data):
    v_idx = np.loadtxt('./union_valid_index.txt').astype(int)  # 2 x 231432

    v_idx = (v_idx[0], v_idx[1])     # 2 x 231432 - (x, y)

    grid = np.zeros([4704, 4448])  # tiff size : 4704 x 4448

    # prediction
    pred  = deepcopy(grid)*0
    pred[v_idx] = pred_dict
    pred[np.where(pred < 0)] = 0

    # true label
    label = deepcopy(grid) * 0
    label[v_idx] = y_data
    label[np.where(label < 0)] = 0

    # visualize
    paint(label, pred)

    # plt.figure()
    # cmap = plt.cm.plasma
    # cmap.set_under(color='black')  # set the color out of the range as black
    # plt.imshow(pred, vmin=0.000001, vmax=3, cmap=cmap)
    # plt.colorbar(orientation = 'vertical')
    #
    # plt.figure()
    # cmap = plt.cm.plasma
    # cmap.set_under(color='black')  # set the color out of the range as black
    # plt.imshow(label, vmin=0.000001, vmax=3, cmap=cmap)
    # plt.colorbar(orientation = 'vertical')
    #
    # plt.show()

import sys
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import prd_module

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

def reconstruction(pred_dict, pred_once, y_data = None, visualize = False):
    valid_x = np.load("./data/x_index.npy").astype(int)
    valid_y = np.load("./data/y_index.npy").astype(int)

    v_idx = (valid_x, valid_y)   # 2 x 73883 - (x, y)
    grid = np.zeros([782, 992])  # tiff size : 782 x 992

    if not pred_once:
        pred_list = []
        pred  = deepcopy(grid)*0
        for idx in range(len(pred_dict)):
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

#
# if __name__ == "__main__":
#     rains_x = np.load('./data/x_data.npy')[-5, :]
#     rains_y = np.load('./data/y_data.npy')[-5, :]
#
#     pred, t = prd_module.prediction(rains_x, rains_y)
#     reconstruction(pred, rains_y)
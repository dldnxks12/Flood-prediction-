import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
#import rainfall_data

# input grid와 error를 plt.plot으로 그림
def make_figure(grid, pred, error, inds=None):
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection = '3d')
    xs = inds[0]
    ys = inds[1]
    zs = grid[inds]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Simulation')

    fig2 = plt.figure()
    ax = fig2.add_subplot(projection = '3d')
    zs = pred[inds]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Prediction')

    fig3 = plt.figure()
    ax = fig3.add_subplot(projection = '3d')
    zs = error
    ax.scatter(xs, ys, zs, s=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Errors')
    ax.set_zlim([0, 70])

    fig4 = plt.figure()
    ax = fig4.add_subplot(projection='3d')
    ax.scatter(xs, ys, grid[inds], color='g', marker='1', label='sim', s=0.2)
    ax.scatter(xs, ys, pred[inds], color='r', label='pred', s=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Comparison')
    plt.legend()
    plt.show()


import sys
def zero_filling(arr):
    # arr : rains (= 690 x 36)
    for i in range(len(arr)):

        # min, max, min index, max index 같은 값 -> np.min, np.argmin, ...
        # np.where는 조건을 만들어 그 값을 찾을 때 이용 !
        # 출력은 인덱스로 ...!

        #print("arr[i:,]", arr[i, :].shape, arr[i, :])
        index = np.where(arr[i, :] > 0.0001) # index return
        #print("index = np.where(arr[i, :] > 0.0001)", index)
        data = arr[i, index].flatten()
        #print("data_경안천 = arr[i, index].flatten()", data_경안천)
        temp = np.concatenate((arr[i], data))
        #print("temp = np.concatenate((arr[i], data_경안천))", temp)
        temp = np.delete(temp, index)
        #print("temp = np.delete(temp, index)", temp)
        arr[i] = temp
        #print("arr[i] = temp", arr[i])
        #sys.exit()

    return arr

class Scorer:
    def __init__(self):
        # value accuracy
        self.rmse_list = []
        self.mae_list = []
        self.error_std_list = []
        self.max_error_list = []
        self.min_error_list = []
        self.sim_mean_list = []
        self.pred_mean_list = []
        self.mre_list = []

        # spatial accuracy
        self.sensitivity_list = []
        self.precision_list = []
        self.f1_list = []
        self.accuracy_list = []
        self.iou_list = []

    def track(self, test_data, true_data, valid_index=None):
        # grid error
        p_index = np.where(test_data > 0)
        union = sum((test_data > 0) | (true_data > 0))
        tp = sum((test_data > 0) & (true_data > 0))
        tn = sum(((test_data == 0) * (true_data == 0)))
        fp = sum((test_data == 0) & (true_data < 0))
        fn = sum((test_data == 0) * (true_data > 0))
        sensitivity = tp / len(p_index[0])
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + fn + tn + fp)
        f1 = 2 / (1 / sensitivity + 1 / precision)

        print(sensitivity, precision, f1, accuracy, tp / union)
        self.sensitivity_list.append(sensitivity)
        self.precision_list.append(precision)
        self.f1_list.append(f1)
        self.accuracy_list.append(accuracy)
        self.iou_list.append(tp / union)

        # value error

        if valid_index is None:
            error = test_data - true_data
        else:
            error = test_data[valid_index] - true_data[valid_index]
        rmse = np.sqrt((error ** 2).mean())
        self.rmse_list.append(rmse)
        self.mae_list.append(np.mean(abs(error)))
        self.error_std_list.append(np.std(error))
        self.max_error_list.append(np.max(error))
        self.min_error_list.append(np.min(error))
        self.sim_mean_list.append(np.mean(true_data))
        self.pred_mean_list.append(np.mean(test_data))
        self.mre_list.append(np.mean(error / (true_data + 0.000001)))

    def save_data(self, file_name):
        if file_name is None:
            file_name = 'default'
        np.savetxt('results\\{}\\RMSE.txt'.format(file_name), self.rmse_list, fmt='%.2f')
        np.savetxt('results\\{}\\MAE.txt'.format(file_name), self.mae_list, fmt='%.2f')
        np.savetxt('results\\{}\\error_std.txt'.format(file_name), self.error_std_list, fmt='%.2f')
        np.savetxt('results\\{}\\max_error.txt'.format(file_name), self.max_error_list, fmt='%.2f')
        np.savetxt('results\\{}\\min_error.txt'.format(file_name), self.min_error_list, fmt='%.2f')
        np.savetxt('results\\{}\\sim_mean.txt'.format(file_name), self.sim_mean_list, fmt='%.2f')
        np.savetxt('results\\{}\\pred_mean.txt'.format(file_name), self.pred_mean_list, fmt='%.2f')
        np.savetxt('results\\{}\\MRE.txt'.format(file_name), self.mre_list, fmt='%.2f')
        np.savetxt('results\\{}\\sensitivity.txt'.format(file_name), self.sensitivity_list, fmt='%.2f')
        np.savetxt('results\\{}\\precision.txt'.format(file_name), self.precision_list, fmt='%.2f')
        np.savetxt('results\\{}\\F1.txt'.format(file_name), self.f1_list, fmt='%.2f')
        np.savetxt('results\\{}\\accuracy.txt'.format(file_name), self.accuracy_list, fmt='%.2f')
        np.savetxt('results\\{}\\IOU.txt'.format(file_name), self.iou_list, fmt='%.2f')

    def print_data(self):
        print('RMSE       - mean: {}, std: {}'.format(np.mean(self.rmse_list), np.std(self.rmse_list)))
        print('MAE        - mean: {}, std: {}'.format(np.mean(self.mae_list), np.std(self.mae_list)))
        print('Max Error  - mean: {}, std: {}'.format(np.mean(self.max_error_list), np.std(self.max_error_list)))
        print('Min Error  - mean: {}, std: {}'.format(np.mean(self.min_error_list), np.std(self.min_error_list)))
        print('Sim Mean   - mean: {}, std: {}'.format(np.mean(self.sim_mean_list), np.std(self.sim_mean_list)))
        print('Pred Mean  - mean: {}, std: {}'.format(np.mean(self.pred_mean_list), np.std(self.pred_mean_list)))
        print('MRE        - mean: {}, std: {}'.format(np.mean(self.mre_list), np.std(self.mre_list)))
        print('Sensitivity- mean: {}, std: {}'.format(np.mean(self.sensitivity_list), np.std(self.sensitivity_list)))
        print('Precision  - mean: {}, std: {}'.format(np.mean(self.precision_list), np.std(self.precision_list)))
        print('F1         - mean: {}, std: {}'.format(np.mean(self.f1_list), np.std(self.f1_list)))
        print('Accuracy   - mean: {}, std: {}'.format(np.mean(self.accuracy_list), np.std(self.accuracy_list)))
        print('IOU        - mean: {}, std: {}'.format(np.mean(self.iou_list), np.std(self.iou_list)))


def print_image(test_data, true_data, index, file_name, vmax=3, save=True, show=False):
    fig, axs = plt.subplots(1, 2)
    cmap = plt.cm.plasma
    cmap.set_under(color='black')
    im1 = axs[0].imshow(true_data, vmin=0.000001, vmax=vmax, cmap=cmap)
    im2 = axs[1].imshow(test_data, vmin=0.000001, vmax=vmax, cmap=cmap)
    fig.suptitle(f'C{index + 1}_comparison')
    axs[0].set_title('True data_경안천')
    axs[1].set_title('Test result')
    fig.colorbar(im1, ax=axs, orientation='horizontal')
    plt.axis('off')
    if show:
        plt.show()
    if save:
        if file_name is None:
            file_name = 'default'
        plt.savefig(f'results\\{file_name}\\test{index}_true_pred_comparison.png')
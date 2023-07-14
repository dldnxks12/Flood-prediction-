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


def flipping(arr):
    for i in range(len(arr)):
        index = np.where(arr[i, :] > 0.0001)
        data = arr[i, index].flatten()
        temp = np.concatenate((arr[i], data))
        temp = np.delete(temp, index)
        arr[i] = temp

    return arr

# 강우 데이터
def flipped_raindata(filename=None):
    if filename is None:
        filename = 'rainfall_1.xlsx'

    df = pd.read_excel(filename, engine='openpyxl').fillna(0).to_numpy()
    rains = df #[1:, 1:]
    rains = rains.astype(np.float32)
    rains = np.transpose(rains)

    print("# - Rains shape : ", rains.shape)
    flipped_rains = flipping(rains)
    print("# - Flipped Rains shape : ", rains.shape)

    return flipped_rains


def return_river_data(folder=None, file_name=None, crop=False):
    path = os.path.join(folder, file_name)
    grid = io.imread(path)
    #grid = np.array(Image.open(path))

    grid[np.where(grid < 0)] = 0  # 2313 x 1991

    if not crop:
        return grid
    else:
        return grid[:4096, 352:]

# x_data, y_data = 강우량, 침수결과
# 침수된 적이 있는 grid만 우선적으로 학습 -> 추후 모든 grid로 확장
def gen_valid_dataset(low, high , crop=False, ext='tif'):
    X1 = flipped_raindata(filename = 'rainfall_1.xlsx')
    print(X1.shape)

    X2 = flipped_raindata(filename = 'rainfall_2.xlsx')
    print(X2.shape)

    X = np.append(X1, X2, axis = 0)
    print(X.shape)

    np.save('x_data', X)

    sys.exit()
    valid_index = np.loadtxt('union_valid_index.txt').astype(int)
    valid_index = (valid_index[0], valid_index[1])

    ys = np.empty((0, len(valid_index[0])), float)
    folder_name = 'Depth1'
    for i in range(low, high+1):
        file_name = 'Depth (PF {}).DEM_5m.'.format(i) + ext
        data = return_river_data(folder_name, file_name, crop=crop)
        data = data[valid_index]

        ys = np.append(ys, np.expand_dims(data, axis=0), axis=0)

    np.save('x_data', X)
    return X, ys

# Training models for both 인천, 경안천
def train_total_model(model, capacity=690, random=False, file_name=None):
    # gain dataset
    xs, ys = gen_valid_dataset(0, capacity, crop=False)
    # xs, ys = incheon_dataset()
    # ys = np.reshape(ys, (ys.shape[0], -1))

    # deviding train / test set
    ss = ShuffleSplit(random_state=random, test_size=0.05)
    ss.get_n_splits(xs, ys)
    train_index, test_index = next(ss.split(xs, ys))
    x_train, x_test = xs[train_index], xs[test_index]
    y_train, y_test = ys[train_index], ys[test_index]
    np.savetxt('{}_test_index.txt'.format(file_name), test_index, fmt='%d')

    start = time.time()
    model.fit(x_train, y_train)
    print(f'{type(model).__name__} score: {model.score(x_test, y_test)}')
    print(f'{type(model).__name__} training time: {time.time() - start}')

    if file_name is None:
        pickle.dump(model, open(f'{type(model).__name__}_rivers.pkl', 'wb'), protocol=4)
    else:
        pickle.dump(model, open(f'{file_name}_rivers.pkl', 'wb'), protocol=4)

    return model



if __name__ == "__main__":
    gen_valid_dataset(1, 15000)
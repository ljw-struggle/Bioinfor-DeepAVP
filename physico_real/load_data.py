# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
from sklearn import model_selection

def load_train_data():
    filename = "./data/train.mat"
    file = sio.loadmat(filename)
    data = file['one_hot_data']
    sequence_length = file['sequence_length']
    label = file['label']

    return data, sequence_length, label

def load_test_data():
    filename = "./data/test.mat"
    file = sio.loadmat(filename)
    data = file['one_hot_data']
    sequence_length = file['sequence_length']
    label = file['label']

    return data, sequence_length, label


def K_Fold_Split(feature, label):
    skfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    temp = skfold.split(X=feature, y=label)

    train_kfold_indices = []
    valid_kfold_indices = []

    for i, j in temp:
        train_kfold_indices.append(i)
        valid_kfold_indices.append(j)

    return train_kfold_indices, valid_kfold_indices
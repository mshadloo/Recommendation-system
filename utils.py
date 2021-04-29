import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.activations import elu
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def load_data(file, r_cols):
    ratings = pd.read_csv(file, sep='\t', names=r_cols, encoding='latin-1')
    # cleaning
    ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())
    return ratings.iloc[:, 0:3].values
def train_test_val_split(data, test_size):
    train_data, test_val_data = train_test_split(data, test_size=test_size)
    test_data, val_data = train_test_split(test_val_data, test_size=0.5)
    return train_data, test_data, val_data

class DefaultDict(dict):
    def __missing__(self, key):
        self[key] = [[],[]]
        return self[key]
def getDataDict(ratings):
    #input:a ratings dataset. col1:user_id col2:item_id col3:data
    #output: a dictionary keys: item_id value: list of two lists: one for user_id and one for data
    ratings_dict=DefaultDict()
    for i in range(len(ratings)):
        ratings_dict[ratings[i,1]][0].append(ratings[i,0])
        ratings_dict[ratings[i, 1]][1].append(ratings[i, 2])
    return  ratings_dict

def MSE_observed_ratings(r,r_pred):
    tf_zero=tf.constant(0,dtype=tf.float32)
    non_zero_positions=tf.not_equal(r,tf_zero)
    r=tf.boolean_mask(r,non_zero_positions)
    r_pred=tf.boolean_mask(r_pred,non_zero_positions)
    return K.mean(K.square(r_pred - r), axis=-1)
def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)
def get_X(rating_dict, items, num_users):
    X= np.zeros((len(items),num_users))
    for i,item in enumerate(items):
        for j, user in enumerate(rating_dict[item][0]):
            X[i,user] = rating_dict[item][1][j]
    return X


def save_result(file_name, rmse_train, mae_train, rmse_test, mae_test):
  file_name += ".pkl"
  f ={}
  f['rmse_train'] = rmse_train
  f['mae_train'] = mae_train
  f['rmse_test'] = rmse_test
  f['mae_test'] = mae_test
  name = open(file_name,'wb')
  pickle.dump(f,name)
  name.close()


def load_result(file_name):
  file_name += ".pkl"
  pkl_file = open(file_name, 'rb')
  f = pickle.load(pkl_file)
  rmse_train = f['rmse_train']
  mae_train = f['mae_train']
  rmse_test = f['rmse_test']
  mae_test = f['mae_test']

  pkl_file.close()
  return rmse_train, mae_train, rmse_test, mae_test


def save_testCaces_res(file_name, rmse_train, mae_train, rmse_test, mae_test, options,h_param):

    file_name += ".pkl"
    f = {}
    f['rmse_train'] = rmse_train
    f['mae_train'] = mae_train
    f['rmse_test'] = rmse_test
    f['mae_test'] = mae_test
    f['options'] = options
    f['options'] = h_param
    name = open(file_name, 'wb')
    pickle.dump(f, name)
    name.close()


def load_testCaces_res(file_name):
  file_name += ".pkl"
  pkl_file = open(file_name, 'rb')
  f = pickle.load(pkl_file)
  rmse_train = f['rmse_train']
  mae_train = f['mae_train']
  rmse_test = f['rmse_test']
  mae_test = f['mae_test']
  options = f['options']
  h_param = f['h_param']
  pkl_file.close()
  return rmse_train, mae_train, rmse_test, mae_test, options, h_param

colors = list(mcolors.BASE_COLORS.keys())
def draw(x,plots,plots_labels, xlim ,ylim,xlabel,ylabel,fig_file,title =""):

    fig = plt.figure()
    ax = fig.gca()
    ax.clear()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for i,y in enumerate(plots):
        ax.plot(x, y, linewidth=1, color=colors[i], marker='o', markerfacecolor='b', label=plots_labels[i], linestyle='-',
                markersize=1)
    ax.legend()
    # plt.xticks(np.arange(0, args.epochs +1, step=5), xax)
    plt.xticks(list(range(1, len(x)+1, 4)))
    plt.show(block=False)
    plt.savefig(fig_file + '.png')

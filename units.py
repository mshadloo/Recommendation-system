import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Embedding, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import elu,softplus

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

def getDataDict(ratings):
    #input:a ratings dataset. col1:user_id col2:item_id col3:data
    #output: a dictionary keys: item_id value: list of two lists: one for user_id and one for data
    ratings_dict={}
    for i in range(ratings.shape[0]):
        if(ratings_dict.get(ratings[i,1]) is None):
            ratings_dict[ratings[i,1]]=[[ratings[i,0]],[ratings[i,2]]]
        else:
            ratings_dict[ratings[i,1]][0].append(ratings[i,0])
            ratings_dict[ratings[i,1]][1].append(ratings[i,2])
    return  ratings_dict

def MSE_observed_ratings(r,r_pred):
    tf_zero=tf.constant(0,dtype=tf.float32)
    non_zero_positions=tf.not_equal(r,tf_zero)
    r=tf.boolean_mask(r,non_zero_positions)
    r_pred=tf.boolean_mask(r_pred,non_zero_positions)
    return K.mean(K.square(r_pred - r), axis=-1)

def get_x_test_dict(x_train_dict,test_items):
    x_test_dict={}
    for item in test_items:
        if (item in  x_train_dict):
            x_test_dict[item]=x_train_dict[item]
        else:
            x_test_dict[item]=[[],[]]
    return  x_test_dict


def evaluate(model,x_test_dict,y_test_dict):
    mse = 0
    mae = 0
    n_ratings = 0
    batch_size=500
    num_items = len(list(y_test_dict.keys()))
    num_batches = int(math.ceil(num_items / batch_size))
    num_users=model.layers[0].input_shape[1]
    for b in range(num_batches):
        b_size = min(batch_size, num_items - b * batch_size)
        batch=np.zeros((b_size,num_users))
        for i in range(b_size):
            item_ind=list(x_test_dict.keys())[i+b*batch_size]
            batch[i, x_test_dict[item_ind][0]] = x_test_dict[item_ind][1]

        r_bacth_pred = model.predict(batch)

        for i in range(b_size):
            item_ind=list(y_test_dict.keys())[i+b*batch_size]
            r_item=np.array([y_test_dict[item_ind][1]])
            r_hat_item=[]
            for u in y_test_dict[item_ind][0]:
                if(u>=num_users):
                    print('find new user')
                    r_hat_item.append(0)
                else:
                    r_hat_item.append(r_bacth_pred[i,u])
            r_hat_item=np.array([r_hat_item])
            mse += np.sum((r_hat_item-r_item) ** 2)
            mae += np.sum(np.abs(r_hat_item-r_item))
            n_ratings += len(y_test_dict[item_ind][1])
    return np.sqrt(mse/n_ratings),mae/n_ratings




MOVIELENS_DIR = 'RatingsData'
ratings= np.load('ratingsSets100k.npz')
train_ratings=ratings['train_data']
val_ratings=ratings['val_data']
test_ratings=ratings['test_data']

num_users=np.max(train_ratings[:,0])+1


y_train_dict=getDataDict(train_ratings)
y_test_dict=getDataDict(test_ratings)
y_val_dict=getDataDict(val_ratings)

x_train_dict=y_train_dict
x_test_dict=get_x_test_dict(x_train_dict,list(y_test_dict.keys()))
x_val_dict=get_x_test_dict(x_train_dict,list(y_val_dict.keys()))



r_i_th=Input(shape=(num_users,))
h1_th=Dense(50,activation='tanh',kernel_regularizer=l2(0))(r_i_th)
r_hat_i_th=Dense(num_users,activation='linear')(h1_th)
model_th=Model(inputs=r_i_th, outputs=r_hat_i_th)
model_th.compile(optimizer='adam', loss=MSE_observed_ratings,  metrics=['mae'])


r_i_sel=Input(shape=(num_users,))
h1_sel=Dense(100,activation='tanh',kernel_regularizer=l2(0))(r_i_sel)
r_hat_i_sel=Dense(num_users,activation='linear')(h1_sel)
model_sel=Model(inputs=r_i_sel, outputs=r_hat_i_sel)
model_sel.compile(optimizer='adam', loss=MSE_observed_ratings,  metrics=['mae'])

r_i_sp=Input(shape=(num_users,))
h1_sp=Dense(200,activation='tanh',kernel_regularizer=l2(0))(r_i_sp)
r_hat_i_sp=Dense(num_users,activation='linear')(h1_sp)
model_sp=Model(inputs=r_i_sp, outputs=r_hat_i_sp)
model_sp.compile(optimizer='adam', loss=MSE_observed_ratings,  metrics=['mae'])

print('model created')



num_epochs=60
batch_size=100
train_mae = np.zeros((num_epochs,1))
train_mse = np.zeros((num_epochs,1))
train_score = np.zeros((num_epochs,1))
# test_mae = np.zeros((num_epochs,1))
# test_mse = np.zeros((num_epochs,1))
# test_score = np.zeros((num_epochs,1))
val_mae = np.zeros((num_epochs,1))
val_mse = np.zeros((num_epochs,1))
val_score = np.zeros((num_epochs,1))


train_mae_sel = np.zeros((num_epochs,1))
train_mse_sel = np.zeros((num_epochs,1))
train_score_sel = np.zeros((num_epochs,1))
# test_mae_sel = np.zeros((num_epochs,1))
# test_mse_sel = np.zeros((num_epochs,1))
# test_score_sel = np.zeros((num_epochs,1))
val_mae_sel = np.zeros((num_epochs,1))
val_mse_sel = np.zeros((num_epochs,1))
val_score_sel = np.zeros((num_epochs,1))

train_mae_sp = np.zeros((num_epochs,1))
train_mse_sp = np.zeros((num_epochs,1))
train_score_sp = np.zeros((num_epochs,1))
# test_mae_sel = np.zeros((num_epochs,1))
# test_mse_sel = np.zeros((num_epochs,1))
# test_score_sel = np.zeros((num_epochs,1))
val_mae_sp = np.zeros((num_epochs,1))
val_mse_sp = np.zeros((num_epochs,1))
val_score_sp = np.zeros((num_epochs,1))


num_items=len(x_train_dict.keys())
num_batches=int(math.ceil(num_items/batch_size))

xax = np.arange(0,num_epochs,1)

fig = plt.figure()
ax = fig.gca()






for epoch in range(num_epochs):
    #evaluate all ratings: train, test and val with current model on this epoch
  #  train_mse[epoch], train_mae[epoch]=evaluate(model_th,x_train_dict,y_train_dict)
    val_mse[epoch], val_mae[epoch] = evaluate(model_th,x_val_dict,y_val_dict)

   # train_mse_sel[epoch], train_mae_sel[epoch] = evaluate(model_sel, x_train_dict, y_train_dict)
    val_mse_sel[epoch], val_mae_sel[epoch] = evaluate(model_sel, x_val_dict, y_val_dict)

    #train_mse_sp[epoch], train_mae_sp[epoch] = evaluate(model_sp, x_train_dict, y_train_dict)
    val_mse_sp[epoch], val_mae_sp[epoch] = evaluate(model_sp, x_val_dict, y_val_dict)



    shuffle_items_inds = np.random.permutation(num_items)
    for b in range(num_batches):
        b_size=min(batch_size,num_items-b*batch_size)
        batch = np.zeros((b_size, num_users))
        for i in range(b_size):
            item_ind=list(x_train_dict.keys())[shuffle_items_inds[i+b*batch_size]]
            batch[i,x_train_dict[item_ind][0]]=x_train_dict[item_ind][1]
        #print("batch "+ str(b) +" is made")
        model_th.fit(batch, batch, epochs=1, verbose=0)#x_train=y_train
        model_sel.fit(batch, batch, epochs=1, verbose=0)
        model_sp.fit(batch, batch, epochs=1, verbose=0)




ax.clear()
plt.title('number of hidden units')
plt.xlim([-1,num_epochs])
plt.ylim([0,2])
plt.xlabel('number of epochs')
plt.ylabel('mean square error')
ax.plot(xax, val_mse_sel,  linewidth=1, color='r', marker='o',markerfacecolor='b', label='200', linestyle='-', markersize=1)
ax.plot(xax, val_mse,  linewidth=1, color='k', marker='o', markerfacecolor='b',
            label='50', linestyle='-',markersize=1)
ax.plot(xax, val_mse_sp,  linewidth=1, color='g', marker='o', markerfacecolor='b',
            label='100', linestyle='-',markersize=1)
ax.legend()
plt.show(block=False)
plt.savefig('units.png')
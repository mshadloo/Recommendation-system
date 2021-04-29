import math
from utils import  get_X, selu
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Embedding, Concatenate
from tensorflow.keras.models import Model
import random
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import tanh,softplus
def autorec_model(num_users, hidden_units, activation ):
  r_i=Input(shape=(num_users,))

  h1=Dense(hidden_units,activation=activation,kernel_regularizer=l2(0))(r_i)

  r_hat_i=Dense(num_users,activation='linear')(h1)

  model=Model(inputs=r_i, outputs=r_hat_i)
  return model
def evaluate(model,test_dict,rating_dict, batch_size,num_users):
    items = list(test_dict.keys())
    batch_num = math.ceil(len(items) / batch_size)
    se, me = 0, 0
    rating_num = 0
    for i in range(batch_num):
        test_items = items[i * batch_size:min((i + 1) * batch_size, len(items))]
        X_test = get_X(rating_dict, test_items, num_users)
        y_pred = model.predict(X_test)
        for i,item in enumerate(test_items):
            for j,user in enumerate(test_dict[item][0]):
                se += (y_pred[i,user] - test_dict[item][1][j])**2
                me += abs(y_pred[i,user] - test_dict[item][1][j])
                rating_num += 1
    return math.sqrt(se/rating_num), me/rating_num

def one_epoch_train(model, train_dict, batch_size,num_users):
    items = list(train_dict.keys())
    random.shuffle(items)
    batch_num = math.ceil(len(train_dict)/batch_size)
    for i in range(batch_num):
        X_train= get_X(train_dict,items[i*batch_size:min((i + 1 )*batch_size,len(items)) ],num_users)
        model.fit(X_train,X_train,epochs=1, verbose=0)
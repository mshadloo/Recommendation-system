import argparse
from main import activation_options, optimizer_options
import os
from autorec import autorec_model, one_epoch_train, evaluate
from utils import load_data, train_test_val_split, getDataDict, MSE_observed_ratings, save_result, selu, save_testCaces_res
import numpy as np
from default_config import activation, hidden, optimizer, data_dir, dataset, r_cols, epochs, batch_size

from tensorflow.keras.activations import tanh,softplus


h_options = ['activation', 'optimizer', 'units_num']
units_options =[20,100,200]
parser = argparse.ArgumentParser()
parser.add_argument('--h_param', metavar='hyperparameter', default='units_num',
                    help='the hyperparameter you want to tune '+ ' | '.join(h_options) +
                    ' (default: activation)')
parser.add_argument('--dataset', dest='dataset',
                    help='Name of data file ',
                    default=dataset, type=str)
parser.add_argument('--data_dir', dest='data_dir',
                    help='The directory containing dataset',
                    default=data_dir, type=str)

parser.add_argument('--result_dir', dest='result',
                    help='The directory used to save the results',
                    default='result', type=str)

parser.add_argument('--checkpoint', dest='checkpoint',
                    help='The directory used to save the trained models',
                    default='checkpoint', type=str)

args = parser.parse_args()
working_dir = './test_cases'
if not os.path.exists(working_dir):
      os.makedirs(working_dir)
data_dir = args.data_dir
checkpoint_dir = os.path.join(working_dir, args.checkpoint )
if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
result_dir = os.path.join(working_dir, args.result )
if not os.path.exists(result_dir):
      os.makedirs(result_dir)
if __name__ =="__main__":
    rating_data = load_data(os.path.join(data_dir, args.dataset), r_cols)
    train_ratings, test_ratings, val_ratings = train_test_val_split(rating_data, test_size=0.2)
    train_dict = getDataDict(train_ratings)
    # test_dict = getDataDict(test_ratings)
    val_dict = getDataDict(val_ratings)
    ratings_dict = train_dict
    num_users = np.max(train_ratings[:, 0]) + 1
    models = []
    models_name =[]
    rmse_train, mae_train = [], []
    rmse_val, mae_val = [], []
    options =None
    if args.h_param == "activation":
        options = list(activation_options.keys())
        for a in activation_options.keys():
            models.append(autorec_model(num_users,hidden, activation_options[a] ))
            models[-1].compile(optimizer=optimizer, loss=MSE_observed_ratings, metrics=['mae'])
            models_name.append(a+"_" +str(hidden)+"_"+optimizer)

    elif args.h_param== "optimizer":
        options= optimizer_options
        for o in optimizer_options:
            models.append(autorec_model(num_users, hidden, activation_options[activation]))
            models[-1].compile(optimizer=o, loss=MSE_observed_ratings, metrics=['mae'])
            models_name.append(activation + "_" + str(hidden) + "_" + o)

    else:
        for u in units_options:
            options = units_options
            models.append(autorec_model(num_users, u, activation_options[activation]))
            models[-1].compile(optimizer=optimizer, loss=MSE_observed_ratings, metrics=['mae'])
            models_name.append(activation + "_" + str(u) + "_" + optimizer)
    best_rmse = []
    for o in options:
        rmse_train.append([])
        mae_train.append([])
        rmse_val.append([])
        mae_val.append([])
        best_rmse.append(float('inf'))




    for epoch in range(epochs):
        for i in range(len(options)):

            one_epoch_train(models[i], train_dict, batch_size, num_users)
            mse_train, mae_train_ = evaluate(models[i], train_dict, ratings_dict, batch_size=batch_size,
                                             num_users=num_users)
            mse_val, mae_val_ = evaluate(models[i], val_dict, ratings_dict, batch_size=batch_size,
                                           num_users=num_users)
            print('Epoch {} model={} train_rmse {:.4f} train_mae {:.4f}'.format(epoch + 1,models_name[i], mse_train, mae_train_))
            print('Epoch {} model={} val_rmse {:.4f} val_mae {:.4f}'.format(epoch + 1, models_name[i],mse_val, mae_val_))
            rmse_train[i].append(mse_train)
            mae_train[i].append(mae_train_)
            rmse_val[i].append(mse_val)
            if mse_val < best_rmse[i]:
                best_rmse[i] = mse_val
                models[i].save(os.path.join(checkpoint_dir,
                                        'best_model_' + models_name[i] + '.h5'))
                save_result(os.path.join(result_dir,
                                         "best_result_" + models_name[i]),
                            mse_train, mae_train_, mse_val, mae_val_)
            mae_val[i].append(mae_val_)

    for i in range(len(options)):
        models[i].save(os.path.join(checkpoint_dir,
                                'model_' + models_name[i] + '.h5'))
        save_result(os.path.join(result_dir, "result_" +models_name[i]),
                    rmse_train[i], mae_train[i], rmse_val[i], mae_val[i])
    save_testCaces_res(os.path.join(result_dir, "test_case_" + args.h_param), rmse_train, mae_train, rmse_val, mae_val,
                       options, args.h_param)




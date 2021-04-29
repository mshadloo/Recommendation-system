from utils import load_data, train_test_val_split,getDataDict,MSE_observed_ratings, save_result, selu
import argparse
from autorec import autorec_model, one_epoch_train, evaluate
import numpy as np
from tensorflow.keras.activations import tanh,softplus
import os
from default_config import activation, hidden, optimizer, data_dir, dataset, epochs, batch_size
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser()
activation_options = {'tanh':tanh, 'selu':selu, 'softplus':softplus}
optimizer_options =['adam', 'sgd', 'RMSprop']
parser.add_argument('--activation', '-activation', metavar='activation', default=activation,
                    choices=list(activation_options.keys()),
                    help='Activation function of hidden layer: ' + ' | '.join(list(activation_options.keys())) +
                    ' (default: tanh)')
parser.add_argument('--optimizer', '-optimizer', metavar='Optimizer', default=optimizer,choices=optimizer_options,
                    help='Optimization method: ' +" | ".join(optimizer_options)+' (default: adam)')
parser.add_argument('--epochs', default=epochs, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--hidden', default=hidden, type=int, metavar='N',
                    help='number of hidden units of Recurrent Layer')

parser.add_argument('--batch_size', default=batch_size, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--checkpoint', dest='checkpoint',
                    help='The directory used to save the trained models',
                    default='checkpoint', type=str)
parser.add_argument('--dataset', dest='dataset',
                    help='Name of data file ',
                    default=dataset, type=str)
parser.add_argument('--data_dir', dest='data_dir',
                    help='The directory containing dataset',
                    default=data_dir, type=str)

parser.add_argument('--result_dir', dest='result',
                    help='The directory used to save the results',
                    default='result', type=str)


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
args = parser.parse_args()
working_dir = '.'

data_dir = args.data_dir
checkpoint_dir = os.path.join(working_dir, args.checkpoint )
if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
result_dir = os.path.join(working_dir, args.result )
if not os.path.exists(result_dir):
      os.makedirs(result_dir)


#input user ratings for ech item, item based


if __name__ == '__main__':

    rating_data = load_data(os.path.join(data_dir, args.dataset), r_cols)
    train_ratings, test_ratings, val_ratings = train_test_val_split(rating_data, test_size= 0.2)
    train_dict = getDataDict(train_ratings)
    test_dict = getDataDict(test_ratings)
    val_dict = getDataDict(val_ratings)
    ratings_dict = train_dict
    num_users = np.max(train_ratings[:, 0]) + 1
    model = autorec_model(num_users,args.hidden, activation_options[args.activation])
    model.compile(optimizer=args.optimizer, loss=MSE_observed_ratings, metrics=['mae'])
    rmse_train, mae_train = [], []
    rmse_test, mae_test = [], []
    best_rmse = float('inf')
    for epoch in range(args.epochs):
        one_epoch_train(model, train_dict, args.batch_size, num_users)
        mse_train,mae_train_ = evaluate(model, train_dict, ratings_dict, batch_size=args.batch_size, num_users= num_users)
        mse_test, mae_test_ = evaluate(model, test_dict, ratings_dict, batch_size=args.batch_size,
                                        num_users=num_users)
        print('Epoch {} train_mse {:.4f} train_mae {:.4f}'.format(epoch +1,  mse_train,mae_train_))
        print('Epoch {} test_mse {:.4f} test_mae {:.4f}'.format(epoch + 1, mse_test, mae_test_))
        rmse_train.append(mse_train)
        mae_train.append(mae_train_)
        rmse_test.append(mse_test)
        if mse_test < best_rmse:
            best_rmse = mse_test
            model.save(os.path.join(checkpoint_dir, 'best_model_'+args.activation+"_"+str(args.hidden)+"_"+args.optimizer+'.h5'))
            save_result(os.path.join(result_dir,"best_result_"+args.activation+"_"+str(args.hidden)+"_"+args.optimizer), mse_train, mae_train_, mse_test, mae_test_)
        mae_test.append(mae_test_)
    model.save(os.path.join(checkpoint_dir, 'model_'+args.activation+"_"+str(args.hidden)+"_"+args.optimizer+'.h5'))
    save_result(os.path.join(result_dir,"result_"+args.activation+"_"+str(args.hidden)+"_"+args.optimizer), rmse_train, mae_train,rmse_test, mae_test)









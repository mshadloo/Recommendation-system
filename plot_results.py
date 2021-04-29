import argparse
from default_config import activation, hidden, optimizer, data_dir, dataset, epochs, batch_size
import numpy as np
from utils import load_result, draw
import os

activation_options = ['tanh' , 'selu', 'softplus']
optimizer_options =['adam', 'sgd', 'RMSprop']
units_options =[20,100,200]
parser = argparse.ArgumentParser()
experiments= ['activation', 'optimizer', 'units_num','default']
parser.add_argument('--exp', '-exp', metavar='experiment', default='default',
                    choices=experiments,
                    help='show the results of experiments on : ' + ' | '.join(experiments) + ' setting'
                    ' (default: default setting)')
parser.add_argument('--activation', '-activation', metavar='activation', default=activation,
                    choices=activation_options,
                    help='Activation function of hidden layer: ' + ' | '.join(activation_options) +
                    ' (default: tanh)')
parser.add_argument('--optimizer', '-optimizer', metavar='Optimizer', default=optimizer,choices=optimizer_options,
                    help='Optimization method: ' +" | ".join(optimizer_options)+' (default: adam)')
parser.add_argument('--hidden', default=hidden, type=int, metavar='N',
                    help='number of hidden units of Recurrent Layer')

parser.add_argument('-best', dest='best', action='store_true',
                    help='show best result')

args = parser.parse_args()



if __name__ == '__main__':
    x = np.arange(1,1+epochs,1)
    xlim = [0, epochs +1]
    ylim = [0.4, 1.4]
    xlabel = 'number of epochs'
    ylabel = 'root mean square error'
    plots =[]
    val_plots =[]
    plots_labels=[]
    if args.exp == 'default':
        result_dir = './result'
        rmse_train, mae_train, rmse_test, mae_test = load_result(
            os.path.join(result_dir, "best_result_ "if args.best else 'result_' + args.activation + "_" + str(args.hidden) + "_" + args.optimizer))
        title = "Activation function of hidden layer: " + args.activation + "\n Number of hidden units: " + str(
            args.hidden) + "\n Optimization method: " + args.optimizer
        if args.best:
            print(title)
            print('root mean square error of best model on training data: ', rmse_train)
            print('root mean square error of best model on test data: ', rmse_test)
        else:
            plots = [rmse_train,rmse_test]
            plots_labels =['training', 'test']
            fig_file = args.activation + "_" + str(args.hidden) + "_" + args.optimizer

    else:
        result_dir = './test_cases/result'
        if args.exp == 'activation':
            title = "Number of hidden units: " + str(hidden)+"\n Optimization method: " + optimizer+"\n Results on validation data:"
            fig_file = "test_case_"+ args.exp + "_" + str(hidden) + "_" + optimizer
            for a in activation_options:
                rmse_train, mae_train, rmse_test, mae_test = load_result(
                    os.path.join(result_dir, "best_result_ " if args.best else 'result_' + a + "_" + str(hidden) + "_" + optimizer))
                if args.best:

                    print("Activation function of hidden layer: " + a + ", Number of hidden units: " + str(hidden) + ", Optimization method: " + optimizer)
                    print('root mean square error of best model on training data: ', rmse_train)
                    print('root mean square error of best model on validation data: ', rmse_test)
                else:
                    plots.append(rmse_test)

                    plots_labels.append(a)
        elif args.exp == 'optimizer':
            ylim = [0.4, 2.5]
            fig_file = "test_case_" + args.exp + "_" + activation+"_"+str(hidden)
            title = "Activation function of hidden layer: " + activation + "\nNumber of hidden units: " +str(hidden)+"\n Results on validation data:"
            for o in optimizer_options:
                rmse_train, mae_train, rmse_test, mae_test = load_result(
                    os.path.join(result_dir,
                                 "best_result_ " if args.best else 'result_' + activation + "_" + str(hidden) + "_" + o))
                if args.best:

                    print("Activation function of hidden layer: " + activation + ", Number of hidden units: " + str(
                        hidden) + ", Optimization method: " + o)
                    print('root mean square error of best model on training data: ', rmse_train)
                    print('root mean square error of best model on validation data: ', rmse_test)
                else:
                    plots.append(rmse_test)

                    plots_labels.append(o)
        elif args.exp == 'units_num':
            fig_file = "test_case_" + args.exp + "_" + activation + "_" + optimizer
            title = "Activation function of hidden layer: " + activation + "\n Optimization method: " + optimizer + "\n Results on validation data:"
            for u in units_options :
                rmse_train, mae_train, rmse_test, mae_test = load_result(
                    os.path.join(result_dir,
                                 "best_result_ " if args.best else 'result_' + activation + "_" + str(
                                     u) + "_" + optimizer))
                if args.best:

                    print("Activation function of hidden layer: " + activation + ", Number of hidden units: " + str(
                        u) + ", Optimization method: " + optimizer)
                    print('root mean square error of best model on training data: ', rmse_train)
                    print('root mean square error of best model on validation data: ', rmse_test)
                else:
                    plots.append(rmse_test)

                    plots_labels.append("# of units ="+str(u))




    draw(x, plots, plots_labels, xlim, ylim, xlabel, ylabel, fig_file, title)




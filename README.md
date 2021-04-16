# Recommendation-system
In this repo, I implemented AutoRec, the learning model proposed in the following paper:
S. Sedhain, A. K. Menon, S. Sanner, and L. Xie. Autorec:Autoencoders meet collaborative filtering. In WWW, pages 111–112, 2015

I evaluate AutoRec on 100K Movielens dataset.

First I explore and read data in dataprocessing.py  file using panda. I split it to training, test and dev sets using sklearn. In main.py I load data and convert them to dictionary, because they are sparse matrix. Dictionary keys are item ids and values are lists of two lists: one for user_id and one for rating.

I tune following hyperparameters using dev set. number of units in hidden layer. the activation function of hidden layer. batch_size and number of iterations.

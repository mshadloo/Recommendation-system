# Recommendation system using AutoRec
In this repo, I implemented AutoRec, a colabrative filtering model proposed in the paper [Autoencoders meet collaborative filtering](http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf) with Keras and tuned hyperparameters of this model using validation set.


![](/autorec.png)

## Dataset
I evaluate AutoRec on [100K Movielens dataset](https://grouplens.org/datasets/movielens/100k/).

I tune following hyperparameters using validation data: number of units in hidden layer=100, the activation function of hidden layer=tanh and the optimization method = adam. 

## Experiments:
- Comparing different number of hidden units:


![](/test_case_activation_100_adam.png)

- Comparing different activation functions:


![](/test_case_activation_100_adam.png)


- Comparing different optimation methods:

![](/test_case_optimizer_tanh_100.png)

# Recommendation system using AutoRec
In this repo, I implemented AutoRec, a colabrative filtering model proposed in the paper [Autoencoders meet collaborative filtering](http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf) with Keras and tuned hyperparameters of this model using validation set.

AutoRec is an autoencoder framework for collaborative filtering(CF). There are two variants of AutoRec depending on two types of inputs: item-based AutoRec (I-AutoRec) and user-based AutoRec (U-AutoRec). I implemented I-AutoRec in this repo. I-AutoRec, the input of the model is item-interaction vector. For item i, item-interaction vector of i is the ith column of the rating matrix.


![](/autorec.png)

## Dataset
I evaluate AutoRec on [100K Movielens dataset](https://grouplens.org/datasets/movielens/100k/).



## Experiments:

I tune following hyperparameters using validation data: number of units in hidden layer=100, the activation function of hidden layer=tanh and the optimization method = adam. 


Result of tuned autorec on Test set          |  Results for different number of hidden units 
:-------------------------:|:-------------------------:
![](/tanh_100_adam.png)  |  ![](/test_case_units_num_tanh_adam.png)

Results for different activation functions          |  Results for different optimization methods 
:-------------------------:|:-------------------------:
![](/test_case_activation_100_adam.png)  |  ![](/test_case_optimizer_tanh_100.png)


## How to run:
```
git clone https://github.com/mshadloo/Recommendation-system.git
cd Recommendation-system
chmod +x data.sh && ./data.sh
chmod +x run.sh && ./run.sh
```


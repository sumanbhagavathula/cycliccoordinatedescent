# cyclic coordinate descent

This code implements two algorithms: 

1. cyclic coordinate descent; and 
2. random coordinate descent.

Coordinate descent algorithm is used to solve the LASSO regularization problem where some of the coefficients will go to zero depending on the size of the hyperparameter we choose. This can be a useful method when we want to find out which features do not contribute much to the prediction power. In practice, a LASSO solution is used as a first step of the regression problem and another round of modelling is done on the data with just the remaining features that have non-zero coefficient values from LASSO. If the second method is a ordinary least squares or Ridge regression, it can give a model that has resulting coefficients that have interpretable values for prediction. 

In the code, I have a main coordinatedescent.py that covers all the functions required for my implementation of the coordinate descent algorithms. And includes the cross validation function as well.

There are several example files that have names starting with 'example_'. You may clone this directory and run any of the examples that you wish to follow through.

There are two datasets involved and both can be obtained using the datasets.py. The examples already have the codes required to download these datasets. The two datasets are: 1. a sample simulated dataset 2. Hitters dataset that is reference in the textbook mentioned below.

There are also examples that compare my version of the algorithm to the sklearn's corresponding implementations. 

References:
- Hastie, T. et. al. (2014). An Introduction to Statistical Learning with Applications in R, p. 219

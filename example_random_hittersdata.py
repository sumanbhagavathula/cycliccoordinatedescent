import numpy as np
import pandas as pd
import math
from IPython.core.display import display

import datasets
import coordinatedescent as cd

#get the hitters data
data, output = datasets.getdata(name='hitters',standardize=True,removeNA=True)

#split the dataset into train and test
x_train, x_test, y_train, y_test = cd.train_test_split(data, output, 0.25)

max_iters = 100
betas_initial = np.random.uniform(0,0,x_train.shape[1])

#run cross-validation to get the best lambda
bestlambdaCV=cd.coorddescentCV(x_train,y_train,method='random', max_iters=max_iters)

print('Best lambda from cross validation: ' + str(bestlambdaCV))
#run the random coordinate descent algorithm with the best lambda obtained from cross validation
randombetas = cd.randcoorddescent(x_train,y_train,betas_initial,bestlambdaCV,max_iters=max_iters)

print('beta values from the final iteration')
display(randombetas[-1])

#plot the objective value vs iterations
cd.objective_plot(bestlambdaCV, x_train, y_train, betas_cyclic = [], betas_random = randombetas)

test_mean_square_error = cd.computescore(randombetas[-1],x_test,y_test)

print('test mean square error: ' + str(test_mean_square_error))

print('done')
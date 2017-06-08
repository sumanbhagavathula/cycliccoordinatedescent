import numpy as np
import pandas as pd
import math
from IPython.core.display import display

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

import datasets
import coordinatedescent as cd

# generate the features
data, output = datasets.getdata(num_features=5, num_samples=40,standardize=True)


#split the dataset into train and test
x_train, x_test, y_train, y_test = cd.train_test_split(data, output, 0.25)

max_iters = 100
betas_initial = np.random.uniform(0,0,x_train.shape[1])

#run cross-validation to get the best lambda
bestlambdaCV=cd.coorddescentCV(x_train,y_train,max_iters=max_iters,method='random')

print('Best lambda from cross validation: ' + str(bestlambdaCV))
#run the random coordinate descent algorithm with the best lambda obtained from cross validation
randbetas = cd.randcoorddescent(x_train,y_train,betas_initial,bestlambdaCV,max_iters=max_iters)

print('beta values from the final iteration')
display(randbetas[-1])

#plot the objective value vs iterations
cd.objective_plot(bestlambdaCV, x_train, y_train, betas_cyclic = randbetas, betas_random = [])

test_mean_square_error = cd.computescore(randbetas[-1],x_test,y_test)

print('test mean square error with my algorithm: ' + str(test_mean_square_error))

#run cross validation using sklearn
lassoCV = LassoCV(eps=0.0001,max_iter=1000,cv=10,selection='random',random_state=0,fit_intercept=False,normalize=False)

lassoCV.fit(x_train,y_train)

print('Lasso CV alpha: ' + str(lassoCV.alpha_))
print('please note: alpha needs to be multiplied by 2 to compare with our bestlambdaCV due to the obj function diff')

lasso = Lasso(alpha=lassoCV.alpha_,selection='random',max_iter=1000,random_state=0,fit_intercept=False,normalize=False)

lasso.fit(x_train,y_train)

print('beta values from sklearn: ')
display(lasso.coef_)

#predict with lasso
ypred = lasso.predict(x_test)

#compute mean square error
lassomeansquareerror = np.mean((ypred-y_test)**2)

print('test mean square error with sklearn: ' + str(lassomeansquareerror))

print('done')

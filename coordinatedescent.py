import numpy as np
import pandas as pd
from IPython.core.display import display
import copy
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt
import math
#%matplotlib inline
from datetime import datetime


def computeobj(x,y,betas,lambduh):
    #this function is used to compute the linear regression objective value with lasso penalty
    return(np.sum(np.square(y-np.dot(x,betas)))/x.shape[0] + lambduh*np.sum(np.absolute(betas)))

def cycliccoorddescent(x, y, betas, lambduh, max_iters=1000):
    #this function is used to minimize the linear regression objective function with lasso penalty
    #using the coordinate descent method by minimizing one coordinate at a time
    #and looping through the coordinates in a cyclic fashion
    #until the maximum iterations specified is reached
    iter = 0
    coord = 0
    total_coords = x.shape[1]
    max_iters = max_iters * total_coords
    n = x.shape[0]
    beta_vals = []
    beta_vals = np.zeros((max_iters, x.shape[1]), dtype='float')
    beta_vals[0] = betas
    while iter < max_iters:
        xij = pd.DataFrame(x).iloc[:, coord]
        aj = 2 * np.sum(np.square(xij)) / n
        xminusj = pd.DataFrame(x).iloc[:, [j for j in range(0, x.shape[1]) if j != coord]]
        # print(xminusj.shape)
        betasminusj = np.delete(betas, coord, 0)
        # print(betasminusj.shape)
        temp = np.dot(xminusj, betasminusj)
        cj = 2 * np.dot(np.transpose(xij), y - temp) / n
        # print(cj)
        # print(aj)

        beta_j = computecoeff(cj, aj, lambduh)

        # print(beta_j)

        betas[coord] = beta_j
        beta_vals[iter] = betas

        coord = coord + 1
        if coord == total_coords:
            coord = 0
        iter = iter + 1

    return (beta_vals)


def pickcoord(d):
    #this function is called by the randcoorddescent algorithm to pick a coordinate at random
    return np.random.randint(0,d,size=1)

def randcoorddescent(x, y, betas, lambduh, max_iters=1000):
    #this function is used to minimize the linear regression objective function with lasso penalty
    #using the coordinate descent method by minimizing for a random coordinate at a time
    #and looping through the coordinates until the maximum iterations specified is reached
    iter = 0
    total_coords = x.shape[1]
    max_iters = max_iters * total_coords
    coord = pickcoord(total_coords)
    n = x.shape[0]
    beta_vals = np.zeros((max_iters, x.shape[1]))
    beta_vals[0] = betas
    while iter < max_iters:
        xij = pd.DataFrame(x).iloc[:, coord]
        aj = 2 * np.sum(np.square(xij)) / n
        xminusj = pd.DataFrame(x).iloc[:, [j for j in range(0, x.shape[1]) if j != coord]]
        betasminusj = np.delete(betas, coord, 0)
        temp = np.dot(xminusj, betasminusj)
        cj = 2 * np.dot(np.transpose(xij), y - temp) / n

        beta_j = computecoeff(cj, aj, lambduh)

        betas[coord] = beta_j
        beta_vals[iter] = betas

        prevcoord = coord

        while prevcoord == coord:
            coord = pickcoord(total_coords)

        iter = iter + 1

    return (beta_vals)


def objective_plot(lambduh, x, y, betas_cyclic = [], betas_random = []):
    #this function is used to plot the
    num_points = x.shape[0]
    objs_cyclic = []
    objs_random = []
    cyclic=False
    random=False
    if len(betas_cyclic) != 0:
        cyclic = True
    if len(betas_random) != 0:
        random=True

    if cyclic == True:
        objs_cyclic = np.zeros(num_points)
    if random == True:
        objs_random = np.zeros(num_points)

    for i in range(0, num_points):
        if cyclic == True:
            objs_cyclic[i] = computeobj(x,y,betas_cyclic[i, :], lambduh)
        if random == True:
            objs_random[i] = computeobj(x,y,betas_random[i, :], lambduh)

    fig, ax = plt.subplots()

    if cyclic == True:
        ax.plot(range(1, num_points + 1), objs_cyclic, label='cyclic coordinate descent')
    if random == True:
        ax.plot(range(1, num_points + 1), objs_random, c='red', label='random coordinate gradient')

    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('Objective value vs. iteration when lambda='+str(lambduh))
    ax.legend(loc='upper right')
    plt.show()

    return

def computescore(coeffs, xtest, ytest):
    #computes the mean square error of a prediction with actual output for a test dataset
    score = np.mean((np.dot(xtest, coeffs) - ytest)**2)
    return score

def coorddescentCV(x, y, L=5, max_iters=1000, method='cyclic', folds=10, epsilon=0.00001):
    #implements a cross validation solution for coordinate descent
    #using lambdas generated in logspace
    #no parallelization for the lambdas or folds

    if folds > x.shape[0]:
        folds = (int)(x.shape[0]/2)

    lambda_0 = math.log10(max(abs(np.dot(x.T, y))) / x.shape[0])

    lambda_vals = np.logspace(-8, lambda_0, L)

    lambda_vals = lambda_vals[np.argsort(-lambda_vals)][::-1]
    lambda_vals = lambda_vals[::-1]

    print('doing cross validation with the following values of lambda')
    print(lambda_vals)

    fold_indexes = range(0, x.shape[0])
    fold_indexes = np.random.permutation(fold_indexes)

    if method != 'random':
        method = 'cyclic'

    k = len(lambda_vals)
    score = np.zeros(k)
    for ki in range(0, k):
        subscore = np.zeros(folds)
        for iter in range(0, folds):
            x_train, x_test, y_train, y_test = train_test_split(x, y, folds / 100)
            betas_initial = np.random.uniform(0, 0, x_train.shape[1])
            if method == 'cyclic':
                cyclic_cd_betas = cycliccoorddescent(x_train, y_train, betas_initial, lambda_vals[ki], max_iters)
                subscore[iter] = computescore(cyclic_cd_betas[-1],x_test, y_test)
            else:
                random_cd_betas = randcoorddescent(x_train, y_train, betas_initial, lambda_vals[ki], max_iters)
                subscore[iter] = computescore(random_cd_betas[-1],x_test, y_test)
        score[ki] = np.mean(subscore)

    return lambda_vals[np.argmin(score)]

def train_test_split(x, y, ratio):
    trainindexes = np.random.randint(0, x.shape[0], round(x.shape[0] * ratio))
    testindexes = np.delete(np.arange(0, x.shape[0]), trainindexes)

    x_train = np.asarray(pd.DataFrame(x).iloc[trainindexes,])
    y_train = np.asarray(pd.DataFrame(y).iloc[trainindexes,0])

    x_test = np.asarray(pd.DataFrame(x).iloc[testindexes,])
    y_test = np.asarray(pd.DataFrame(y).iloc[testindexes,0])

    return x_train, x_test, y_train, y_test

def computecoeff(cj, aj, lambduh):
    if cj < -lambduh:
        beta_j = (cj + lambduh) / (aj)
    elif cj > lambduh:
        beta_j = (cj - lambduh) / (aj)
    else:
        beta_j = 0

    return beta_j


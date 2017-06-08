import numpy as np
import pandas as pd
import math

def getdata(name='sample', num_samples = 10, num_features = 2, standardize = False, removeNA = False, seed = 0):
    #returns a dataset as data and output
    #if name is provided and matches hitters, it returns the hitters dataset
    #Hitters dataset is discussed in the Introduction to Statistical Learning textbook by Trevor Hastie et al.
    #and for python, it can be downloaded from the link used in this code
    #if name is not provided or is provided and matches sample, it returns a sample simulated dataset
    #for the sample data, the number of samples and features can be controlled used num_samples and num_features
    #for other data, the num_samples and num_features are optional and are ignored even if provided
    #if standardize is specified as True, a data and output are standardized before returning
    #if removeNA is optional. If it specified as True, any NAs in the dataset will be removed before returning
    #for sample data, removeNA parameter is skipped since we do not simulate NAs in the first place
    #seed is an optional parameter used to control the seed for the random number generator for sample data
    #for other data, seed parameter is ignored
    data = []
    output = []
    np.random.seed(seed)
    if name == 'hitters':
        data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0)

        if removeNA:
            data = data.dropna()

        data = pd.get_dummies(data, drop_first=True)

        output = data['Salary']
        data.drop(['Salary'], axis=1, inplace=True)

    if name == 'sample':

        #generate the features
        num_samples = (int)(math.floor(num_samples / 3) * 3)
        indv_sample_size = (int)(num_samples / 3)
        x = pd.DataFrame([[0 for a in range(num_samples)] for b in range(num_features)])
        for j in range(0, num_features):
            # for i in range(0,math.floor(sample_size/3)):
            random_loc = np.random.uniform(0, 50)
            random_scale = np.random.uniform(200, 675)
            lo = 0
            hi = indv_sample_size
            x.iloc[j, lo:hi] = np.array(np.random.normal(random_loc, random_scale, indv_sample_size))

            random_lo = np.random.normal(120, 180)
            random_hi = np.random.normal(1500, 2100)
            lo = indv_sample_size
            hi = 2 * indv_sample_size
            x.iloc[j, lo:hi] = np.array(np.random.uniform(random_lo, random_hi, indv_sample_size))

            random_loc = np.random.uniform(800, 1100)
            random_scale = np.random.uniform(750, 1200)
            lo = 2 * indv_sample_size
            hi = 3 * indv_sample_size
            x.iloc[j, lo:hi] = np.array(np.random.normal(random_loc, random_scale, indv_sample_size))

        data = np.transpose(x)

        #generate output
        prob_ratio = (float)(np.random.sample(1))
        intercept = (float)(np.random.sample(1))
        output = intercept

        coeff_bits = np.random.choice([0, 1], size=num_features, p=[prob_ratio, 1-prob_ratio])
        coeffs = np.random.sample(num_features)

        randomized_data = data+np.random.sample(num_features*num_samples).reshape(num_samples,num_features)
        output = np.dot(randomized_data,coeffs*coeff_bits) + np.random.sample(num_samples)

    if standardize:
        data = np.asarray(pd.DataFrame(data).apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))))
        output = np.asarray((pd.DataFrame(output).apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))).iloc[:,0])

    data = np.asarray(data)
    output = np.asarray(output)

    return data, output




















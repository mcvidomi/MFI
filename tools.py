import numpy as np
import pickle
import copy
import pdb
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm.libsvm import decision_function
import numpy.matlib
import os

def read_data():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/usps.pkl"):
        import csv
        data = []

        # DOWNLOAD DATA
        import urllib2
        response = urllib2.urlopen('http://mldata.org/repository/data/download/csv/usps/')

        data = []
        for line in response:
            data.append(map(float, line.split(',')))

        data = np.vstack(data)
        ytrain = data[0:7291,0]
        xtrain = data[0:7291,1:]
        ytest  = data[7291:,0]
        xtest  = data[7291:,1:]

        fobj = open('data/usps.pkl','wb')
        pickle.dump([xtrain,ytrain,xtest,ytest],fobj)
        fobj.close()
    fobj = open('data/usps.pkl', 'rb')
    xtrain, ytrain, xtest, ytest = pickle.load(fobj)
    fobj.close()
    return xtrain,ytrain,xtest,ytest

def choose_numbers(x,y,numbers):
    tt = (y == numbers[0]) | (y == numbers[1])
    xred = x[tt]
    yred = y[tt]
    return xred,yred

def readdigits_numbers(filename, t, numbers):
    '''

    :param filename: path to data
    :param t: threshold for grayscale images. Keep grayscale images with t=-100. Generate binary images with treshold t in [-0.5,0.5].
    :param numbers: choose 2 numbers between 1 and 10, note 10 1:= 0, 2:= 1,..., 10:= 9
    :return: digits and labels
    '''

    fobj = open(filename, 'rb')
    [x, y, xb] = pickle.load(fobj)
    fobj.close()
    tt = (y == numbers[0]) | (y == numbers[1])
    x = x[tt]
    y = y[tt]
    xb = xb[tt]
    if t == -100:
        xb = x
    else:
        for i in range(len(x)):
            for j in range(len(x[0])):
                if x[i, j] <= t:
                    xb[i, j] = 0;
                else:
                    xb[i, j] = 1;

    return xb, y



def mfi_mb(prediction, x_rnd, metric, degree=2):
    '''
    :param prediction: prediction values of classifier for x_rnd
    :param x_rnd: random samples
    :param metric: kernel, for instance 'rbf', 'poly'
    :param degree: degree of polynomial kernel (if metric = 'poly')
    :return: heatmap of feature importances
    '''

    # KERNEL MATRIX FOR S(X)
    if metric == "poly":
        km_s = pairwise_kernels(prediction, Y=None, metric=metric, degree=degree)
    else:
        km_s = pairwise_kernels(np.reshape(prediction, (len(prediction), 1)), Y=None, metric=metric)
    # NORMALIZE
    row_sums = km_s.sum(axis=1)
    km_s = km_s / row_sums[:, np.newaxis]
    # KERNEL MATRIX FOR X_ij OF ALL SAMPLES
    mfi = np.zeros(len(x_rnd[0]))
    for i in range(len(x_rnd[0])):
        if metric == "poly":
            km_x = pairwise_kernels(np.reshape(x_rnd[:, i], (len(x_rnd), 1)), Y=None, metric=metric, degree=degree)
        else:
            km_x = pairwise_kernels(np.reshape(x_rnd[:, i], (len(x_rnd), 1)), Y=None, metric=metric)
        if metric != "linear" and metric != "cosine":
            row_sums = km_x.sum(axis=1)
        km_x = (km_x) / row_sums[:, np.newaxis]
        A = np.dot(km_s, km_x)
        mfi[i] = np.trace(A)

    return mfi


def mfi_ib(clf, prediction, x_rnd, x):

    mfi = np.zeros((len(x_rnd[0])))
    for i in range(len(x_rnd[0])):
        xi = copy.deepcopy(x_rnd)
        xi[:,i] = x[i]
        sx = getattr(clf,prediction)(xi)
        mfi[i] = np.sum(sx)
    return mfi



def pixel_flipping(clf, num, heatmap, N):
    index = np.argsort(heatmap)[::-1]
    nums = np.zeros((N, len(num)))
    nums[0, :] = copy.deepcopy(num)
    for j in range(1, N):
        num[index[j - 1]] = np.random.uniform(0, 1)
        nums[j, :] = copy.deepcopy(num)
    f = clf.decision_function(nums)
    f = np.array(f).flatten()
    return f



def flibo(clf,inst,mfi,N):
    index = np.argsort(mfi)[::-1]
    inst_flip = copy.deepcopy(inst)
    prediction = np.sign(clf.decision_function(inst))

    for j in range(1, N):
        inst_flip[index[j - 1]] = np.random.uniform(0, 1)
        if np.sign(clf.decision_function(inst_flip)) != prediction:
            return j+1
    return N

def flibo_relation(clf,inst,mfi,rnd_mfi,N):
    val_mfi = flibo(clf,inst,mfi,N)
    val_rnd = flibo(clf,inst,rnd_mfi,N)
    fliborel = (val_rnd-val_mfi)/(val_rnd-1.)
    return fliborel

def flibo_relation_rep(clf,inst,mfi,N,rep):
    np.random.seed(seed=2)
    flibo = np.array([])
    for r in range(rep):
        flibo = np.insert(flibo,len(flibo),flibo_relation(clf, inst, mfi, np.random.uniform(0, 1, len(mfi)), N))
    return np.mean(flibo)



def pixel_flipping_AOPC(f):
    L = len(f)
    AOPC = np.zeros(L)
    for k in range(L):
        AOPC[k] = (1. / (k + 1)) * np.sum(f[0] - f[0:k])

    return AOPC

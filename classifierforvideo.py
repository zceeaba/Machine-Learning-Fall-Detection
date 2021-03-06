from FallDetect.dataparser import wearable, normalizedwearable
from sklearn import model_selection
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from 'Machine_Learning-Fall-Detection'.addbbtosilhouette import returnvideo


def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    y_pred = y.copy()
    np.nan_to_num(X)
    # Iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    indexlist = []
    for i in range(len(y_pred)):
        indexlist.append(i)
    #    plt.figure()
    plt.scatter(indexlist, y_pred, color='red')
    plt.scatter(indexlist, y_true, color='blue')
    plt.show()
    return np.mean(y_true == y_pred)


def classifiers():
    df = returnvideo()

    X = df.filter(items=['angle', 'distance', 'bbcen','bb'])
    Y = df["groundtruthstate"]
    X = X.as_matrix().astype(np.float)
    Y = Y.as_matrix().astype(np.float)

    # optional StandardScaler()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Support vector machines:")
    print("%.3f" % accuracy(Y, run_cv(X, Y, SVC)))
    print("Random forest:")
    print("%.3f" % accuracy(Y, run_cv(X, Y, RF)))
    print("K-nearest-neighbors:")
    print("%.3f" % accuracy(Y, run_cv(X, Y, KNN)))


classifiers()
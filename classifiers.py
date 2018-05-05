from FallDetect.dataparser import wearable
from sklearn import model_selection
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import scikitplot as skplt


def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    y_pred = y.copy()
    np.nan_to_num(X)
    print(clf_class)
    # Iterate through folds
    if str(clf_class) == "<class 'sklearn.ensemble.forest.RandomForestClassifier'>":
        kwargs["n_estimators"] = 100
        print(clf_class(**kwargs))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            clf = clf_class(**kwargs)
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred
    elif str(clf_class) == "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>":
        kwargs["n_neighbors"] = 10
        print(clf_class(**kwargs))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            clf = clf_class(**kwargs)
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred
    else:
        print(clf_class(**kwargs))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            clf = clf_class(**kwargs)
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred


def accuracy(y_true, y_pred):
    indexlist = []
    for i in range(len(y_pred)):
        indexlist.append(i)
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    return np.mean(y_true == y_pred)


def classifiers():
    results = wearable()
    df = pd.DataFrame(results)

    X = df.filter(items=['accelerationx', 'accelerationy', 'accelerationz'])
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
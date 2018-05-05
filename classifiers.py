
from matplotlib import pyplot as plt

from sklearn import model_selection
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import scikitplot as skplt
from sklearn import metrics
import sklearn
from FallDetect.dataparser import wearable

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    y_pred = y.copy()
    np.nan_to_num(X)
    print(clf_class)
    # Iterate through folds
    if str(clf_class) == "<class 'sklearn.ensemble.forest.RandomForestClassifier'>":
        kwargs["n_estimators"] = 100
        print(clf_class(**kwargs))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        probas = clf.predict_proba(X_test)
        skplt.metrics.plot_roc_curve(y_test, probas)
        plt.show()
#        args = "bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False"
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            # Initialize a classifier with key word arguments
            clf = clf_class(**kwargs)
            clf.fit(X_train,y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred
    elif str(clf_class) == "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>":
        kwargs["n_neighbors"] = 10
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        probas = clf.predict_proba(X_test)
        skplt.metrics.plot_roc_curve(y_test, probas)
        plt.show()
        print(clf_class(**kwargs))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            # Initialize a classifier with key word arguments
            clf = clf_class(**kwargs)
            clf.fit(X_train,y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred
        
    else:
        kwargs["probability"] = True
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        probas = clf.predict_proba(X_test)
        skplt.metrics.plot_roc_curve(y_test, probas)
        plt.show()
        print(clf_class(**kwargs))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            # Initialize a classifier with key word arguments
            clf = clf_class(**kwargs)
            clf.fit(X_train,y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred
def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    indexlist=[]
    for i in range(len(y_pred)):
        indexlist.append(i)

    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    return np.mean(y_true == y_pred)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
def classifiers():
    # readt to merge to master
    results=wearable()
    df=pd.DataFrame(results)
#    print(df)
#    df.to_csv(r"C:\Users\Anmol-Sachdeva\Dekstop\AppliedDataScience\pdframe.csv", sep='\t', encoding='utf-8')
    X=df.filter(items=['magn','accelerationx','accelerationy','accelerationz','magnitude','average'])
    Y=df["groundtruthstate"]
    X = X.as_matrix().astype(np.float)
    Y = Y.as_matrix().astype(np.float)

    # optional StandardScaler()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    print("Support vector machines:")
    print("%.3f" % accuracy(Y, run_cv(X,Y,SVC)))
    print("Random forest:")
    print("%.3f" % accuracy(Y, run_cv(X,Y,RF)))
    print("K-nearest-neighbors:")
    print("%.3f" % accuracy(Y, run_cv(X,Y,KNN)))
    np.nan_to_num(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predicted_probas = nb.predict_proba(X_test)
    predicted = nb.predict(X_test)
    print("GaussianNB:")
    skplt.metrics.plot_roc_curve(y_test, predicted_probas)
    plt.show()
    print("%.3f" % accuracy(y_test, predicted))
    
classifiers()

import json
import datetime

import dateutil.parser

from matplotlib import pyplot as plt

def wearable():

    with open('wearable.json', 'r') as handle:
        json_data = [json.loads(line) for line in handle]
        
    finallist=[]
  
    xlist, ylist, zlist = [], [], []
    xtlist,ytlist,ztlist=[],[],[]
    xlista, ylista, zlista = [], [], []
    xtlista,ytlista,ztlista=[],[],[]
   
    flag=3
    groundnewlist=[]
    groundnewlistb=[]
    groundtlist=[]
    groundtlistb=[]
    
    for i in range(len(json_data)):
        importantdict = {}
        dictless={}

        if json_data[i].get('e'):
            uid=json_data[i]["uid"][-2:]
            t = json_data[i]["e"][0]["t"]
            accel=[0,0,0]
            id = json_data[i]["_id"]["$oid"]
            timestamp = json_data[i]["bt"]["$date"]
            for j in range(len(json_data[i]["e"])):
                v=json_data[i]["e"][j]["v"]
                if len(v)>0:
                    accel[0] += json_data[i]["e"][j]["v"][0]
                    accel[1] += json_data[i]["e"][j]["v"][1]
                    accel[2] += json_data[i]["e"][j]["v"][2]
     
            if uid=="c0" or uid=="c1":
                t=t/2
                parsedtimestamp = dateutil.parser.parse(timestamp)
                result = parsedtimestamp - datetime.timedelta(seconds=t)
                dictless["id"]=id
                dictless["uid"]=uid
                dictless["result"]=result
                dictless["acceleration"]=accel
                dictless["accelerationx"]=accel[0]
                dictless["accelerationy"]=accel[1]
                dictless["accelerationz"]=accel[2]
            else:
                parsedtimestamp = dateutil.parser.parse(timestamp)
                result = parsedtimestamp - datetime.timedelta(seconds=t)
                dictless["id"]=id
                dictless["uid"]=uid
                dictless["result"]=result
                dictless["acceleration"]=accel
                dictless["accelerationx"]=accel[0]
                dictless["accelerationy"]=accel[1]
                dictless["accelerationz"]=accel[2]
            parsedtimestamp = dateutil.parser.parse(timestamp)
            naive = parsedtimestamp.replace(tzinfo=None)
            times = [[8, 10, 21, 23], [41, 45, 57, 61], [75, 80, 93, 97], [110, 114, 126, 129], [142, 146, 159, 164],
                     [178, 183, 195, 198], [219, 222, 234, 237], [258, 262, 274, 278], [299, 303, 313, 317],
                     [332, 335, 349, 353], [378, 382, 392, 396], [420, 424, 435, 440], [452, 456, 466, 470],
                     [486, 491, 501, 507]]
            timesb = [[8, 10, 21, 25], [36, 39, 53, 56], [68, 72, 86, 91], [99, 102, 118, 122], [134, 138, 151, 155],
                      [168, 173, 190, 192], [210, 212, 230, 235], [249, 252, 269, 272], [281, 283, 302, 306],
                      [317, 320, 339, 342], [357, 360, 373, 377], [391, 394, 410, 413], [426, 429, 443, 446],
                      [456, 459, 471, 474]]

            if naive>datetime.datetime(2018, 3, 22, 17, 27, 56, 0) and naive<datetime.datetime(2018, 3, 22, 17,36,46 , 0) :
                if dictless["uid"]=="c0" or dictless["uid"]=="c1":

                    for list in times:
                        for index in range(len(list)):
                            eventimestamp = datetime.datetime(2018, 3, 22, 17, 27, 56, 0) + datetime.timedelta(0, list[index])
                            checktime=(eventimestamp - naive).total_seconds()
                            #print(checktime)
                            if checktime<1 and checktime >-1:
                                dictless["groundtruthstate"]=index
                                flag=index
                    if flag == 1:
                        dictless["groundtruthstate"] = 1
                    elif flag == 2:
                        dictless["groundtruthstate"] = 2
                    elif flag == 0:
                        dictless["groundtruthstate"] = 0
                    else:
                        dictless["groundtruthstate"] = 3

                    if abs(dictless["acceleration"][0])>7:
                        #print(naive)

                        xlist.append(dictless["acceleration"][0])
                        xtlist.append(i)
                    if  abs(dictless["acceleration"][1]) > 7:
                        #print(naive)

                        ylist.append(dictless["acceleration"][1])
                        ytlist.append(i)
                    if abs(dictless["acceleration"][2]) > 7:
                    
                        zlist.append(dictless["acceleration"][2])
                        ztlist.append(i)
                    importantdict["timestamp"]=naive
                    importantdict["groundtruth"]=dictless["groundtruthstate"]
                    groundnewlist.append(dictless["groundtruthstate"])
                    groundtlist.append(i)
                    #finallist.append(importantdict)
                    finallist.append(dictless)
            else:
                dictless["groundtruthstate"]=5
            if naive>datetime.datetime(2018, 3, 22, 17, 15, 56, 0) and naive<datetime.datetime(2018, 3, 22, 17,24,2 , 0) :
                if dictless["uid"]=="c0" or dictless["uid"]=="c1":
 
                    for list in timesb:
                        for index in range(len(list)):
                            eventimestamp = datetime.datetime(2018, 3, 22, 17, 15, 56, 0) + datetime.timedelta(0, list[index])
                            checktime=(eventimestamp - naive).total_seconds()
                            #print(checktime)
                            if checktime<1 and checktime >-1:
                                dictless["groundtruthstate"]=index
                                flag=index

                    if flag == 1:
                        dictless["groundtruthstate"] = 1
                    elif flag == 2:
                        dictless["groundtruthstate"] = 2
                    elif flag == 0:
                        dictless["groundtruthstate"] = 0
                    else:
                        dictless["groundtruthstate"] = 3

                    if abs(dictless["acceleration"][0])>7:
               
                        xlista.append(dictless["acceleration"][0])
                        xtlista.append(i)
                    if  abs(dictless["acceleration"][1]) > 7:
                       
                        ylista.append(dictless["acceleration"][1])
                        ytlista.append(i)
                    if abs(dictless["acceleration"][2]) > 7:
                      
                        zlista.append(dictless["acceleration"][2])
                        ztlista.append(i)
#                    print(dictless["groundtruthstate"])
                    importantdict["timestamp"]=naive
                    importantdict["groundtruth"]=dictless["groundtruthstate"]
                    groundnewlistb.append(dictless["groundtruthstate"])
                    groundtlistb.append(i)
                    #finallistb.append(importantdict)
                    finallist.append(dictless)
            #else:
            #    dictless["groundtruthstate"]=5

    return finallist


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
    X=df.filter(items=['accelerationx','accelerationy','accelerationz'])
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

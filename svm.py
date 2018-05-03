from FallDetect.dataparser import wearable
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def svm():
    #data=wearable()
    from sklearn import datasets
    from sklearn.svm import SVC
    results=wearable()
    df=pd.DataFrame(results)
    msk = np.random.rand(len(df)) < 0.9

    train = df[msk]

    test = df[~msk]
    array=train.values
    #X=array[:,0:5]
    #y=array[:,4]
    print(train)
    X=train.filter(items=['accelerationx','accelerationy','accelerationz'])
    Y=train["groundtruthstate"]
    Xresult=test["groundtruthstate"]
    Xresult=Xresult.copy()
    Xtest=test.filter(items=['accelerationx','accelerationy','accelerationz'])

    clf = SVC()
    clf.fit(X, Y)

    prediction=(clf.predict(Xtest))
    indexlist=[]
    for i in range(len(prediction)):
        indexlist.append(i)
    plt.scatter(indexlist,prediction,color='red')
    plt.scatter(indexlist,Xresult.values,color='blue')
    plt.show()
    #print(Xresult)
    #print(results.mean())
    #print(train)
    #print(test)


svm()
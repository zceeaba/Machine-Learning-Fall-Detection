from FallDetect.readpickle import readpickle
import cv2
from matplotlib import pyplot as plt
from FallDetect.readsilhouette import readsilhouette
import pandas as pd
import math
from skimage.measure import compare_ssim as ssim
import numpy as np
from FallDetect.pickledata import pickledata


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def compare_images(imageA, imageB):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    return m,s

#data=readpickle("testdataforbb.txt")
data=readpickle("finalsillbbdata.txt")
def imagewrite():
    for j in list(data):
        if len(data[j]) < 2:
            del data[j]
    count = 0
    for i in data:
        #if count==0:
        #print(i,data[i])
        filename='./imagesb/'+str(count)+'.png'
        with open(filename, 'wb') as f:
            f.write(data[i][0])
        count = count+1

    print(data)

"""
import numpy as np
for j in list(data):
    if len(data[j]) < 2:
        del data[j]

count=0
for i in data:
    filename = './imagesb/' +str(count)+ '.png'
    image1 = cv2.imread(filename)
    bb=data[i][1]
    bbcen=data[i][2]
    cv2.circle(image1, (bbcen[0], bbcen[1]), 3, (0, 0, 255), -1)
    cv2.rectangle(image1,(bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),3)
    cv2.imwrite(filename,image1)
    count+=1
"""
def returnvideo():
    data = readpickle("finalsillbbdata.txt")
    for j in list(data):
        if len(data[j]) < 2:
            del data[j]
    array = readsilhouette()
    #print(array)
    count=0
    print(len(array))
    length=len(array)
    keys=list(data.keys())
    def determine(value):
        if value in keys:
            return False
        else:
            return True
    newarray = [x for x in array if not determine(x["time"])]
    bb=[]
    bbcen=[]
    for i in data:
        bb.append(data[i][1])
        bbcen.append(data[i][2])
        count+=1
    def distance(pointa,pointb):
        dis=math.sqrt(((pointa[0]-pointb[0])**2+(pointa[1]-pointb[1])**2))
        return dis

    def angle(pointa,pointb):
        y=pointa[1]-pointb[1]
        x=pointa[0]-pointb[0]
        angle=math.atan2(y,x)
        degree=math.degrees(angle)
        return degree

    newarray[0]["bb"]=bb[0]
    newarray[0]["bbcen"]=bbcen[1]
    newarray[0]["distance"]=0
    newarray[0]["angle"]=0
    newarray[0]["mse"]=0
    newarray[0]["ssim"]=0

    for j in range(1,len(newarray)):
        print(j)
        newarray[j]["bb"]=bb[j]
        newarray[j]["distance"]=distance(bb[j],bb[j-1])
        newarray[j]["angle"]=angle(bb[j],bb[j-1])
        newarray[j]["bbcen"]=bbcen[j]
        firstimage=cv2.imread("imagesb/"+str(j-1)+".png")
        secondimage=cv2.imread("imagesb/"+str(j)+".png")
        a = cv2.cvtColor(firstimage, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(secondimage, cv2.COLOR_BGR2GRAY)
        m,s=compare_images(a,b)
        newarray[j]["ssim"]=s
        newarray[j]["mse"]=m

    pickledata(newarray,"videoclassifier")
    # import the necessary packages
    #numarray=np.array(newarray)

    pdarray = pd.DataFrame(newarray)
    return pdarray

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        probas = clf.predict_proba(X_test)
        skplt.metrics.plot_roc_curve(y_test, probas)
        plt.show()
        #        args = "bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False"
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            # Initialize a classifier with key word arguments
            clf = clf_class(**kwargs)
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred
    elif str(clf_class) == "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>":
        kwargs["n_neighbors"] = 10
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        probas = clf.predict_proba(X_test)
        skplt.metrics.plot_roc_curve(y_test, probas)
        plt.show()
        print(clf_class(**kwargs))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            # Initialize a classifier with key word arguments
            clf = clf_class(**kwargs)
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)
        return y_pred

    else:
        kwargs["probability"] = True
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        probas = clf.predict_proba(X_test)
        skplt.metrics.plot_roc_curve(y_test, probas)
        plt.show()
        print(clf_class(**kwargs))
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

    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    return np.mean(y_true == y_pred)


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def classifiers():
    # readt to merge to master
    vlist=readpickle("videoclassifier.txt")
    angles=[]
    distances=[]
    mses=[]
    for i in vlist:
        angles.append(i["angle"])
        distances.append(i["distance"])
        mses.append(i["mse"])
    for i in range(len(distances)):
        if distances[i]>1000:
            distances[i]=0
    maxd=max(distances)
    print(distances)
    print(maxd)
    maxa=max(angles)
    maxmse=max(mses)
    for i in range(len(vlist)):
        vlist[i]["angle"]=float(vlist[i]["angle"]/maxa)
        vlist[i]["distance"]=float(distances[i]/maxd)
        vlist[i]["mse"]=float(mses[i]/maxmse)
    df=pd.DataFrame(vlist)
    #df = pd.DataFrame(readpickle("videoclassifier.txt"))
    #    print(df)
    #    df.to_csv(r"C:\Users\Anmol-Sachdeva\Dekstop\AppliedDataScience\pdframe.csv", sep='\t', encoding='utf-8')
    X = df.filter(items=['angle', 'distance','ssim','mse'])
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

"""
count=0
for i in data:
    filename = './imagesb/' +str(count)+ '.png'
    image1 = cv2.imread(filename)
    bb=data[i][2]
    cv2.circle(image1, (bb[0], bb[1]), 1, (0, 0, 255), -1)
    cv2.imwrite(filename,image1)
    count+=1
"""


""""
filename = './images/' + timestr + "_0_" + str(count) + '.png'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(i[1])
"""
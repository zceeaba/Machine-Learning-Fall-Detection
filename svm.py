from FallDetect.dataparser import wearable
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer


def svm():
    #data=wearable()
    from sklearn import datasets
    from sklearn.svm import SVC
    iris = datasets.load_iris()
    clf = SVC()
    clf.fit(iris.data, iris.target)

    list(clf.predict(iris.data[:3]))

    clf.fit(iris.data, iris.target_names[iris.target])

    result=list(clf.predict(iris.data[:3]))
    print(result)


svm()
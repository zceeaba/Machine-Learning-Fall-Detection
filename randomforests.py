from FallDetect.dataparser import wearable
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
def randomforests():

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
    realX=[]
    #for j in X:
    #    realX.append(j[0])
    #X=X.values
    print(X)
    #print(Y)

    seed = 7
    num_trees = 100
    max_features = 1
    kfold = model_selection.KFold(n_splits=100, random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    
    print(results)
    print(results.mean())
    #print(train)
    #print(test)
    #for x in results:
    #    X.append(results["acceleration"])

    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(url, names=names)
    array = dataframe.values
    X = array[:,0:8]
    Y = array[:,8]
    print("X",X)
    print("Y",Y)

    seed = 7
    num_trees = 100
    max_features = 3
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())
    """

randomforests()
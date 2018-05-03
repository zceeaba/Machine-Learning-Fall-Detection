from FallDetect.readsilhouette import readsilhouette
import pandas as pd
def processsilhouette():
    array=readsilhouette()
    pdarray=pd.DataFrame(array)
    gts=pdarray["groundtruthstate"]
    print(pdarray)
    
processsilhouette()
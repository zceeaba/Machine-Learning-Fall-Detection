from FallDetect.dataparser import wearable,normalizedwearable
results=wearable()
import pandas as pd
d=pd.DataFrame(results)
normresults=normalizedwearable(results)

e=pd.DataFrame(normresults)
print(d,e)
import numpy as np
import pandas as pd
import docutils
from docutils import parsers
from reliability_functions import Load_Model_Algorithm

data=[]
with open("LDAT.rts", 'r') as file:
    for line in file:
        row=line.strip().split('.')
        for element in row:
            if element != '0':
                data.append((10**-len(element))*int(element))
data_np=2850*np.array(data)
data_np[data_np<2.86e-6]=2850
max=np.max(data_np)
min=np.min(data_np)
P,F=Load_Model_Algorithm(min,max,data_np)
print(P)
df_P=pd.DataFrame.from_dict(P,orient='index',columns=['P'])
df_F=pd.DataFrame.from_dict(F,orient='index',columns=['F'])
df=pd.concat([df_P,df_F],axis=1)
df.to_csv("Load_Model.csv")








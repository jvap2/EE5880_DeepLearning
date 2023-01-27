import numpy as np
import pandas as pd
import docutils
from docutils import parsers
from reliability_functions import Load_Model_Algorithm

# with open("LDAT.rts", 'r') as file:
#     for line in file:
#         row=line.strip().split('.')
#         for element in row:
#             if element != '0':
#                 data.append((10**-len(element))*int(element))
data_df=pd.read_csv("load_csv_data.csv")
data=data_df.to_numpy().flatten()
print(data)

# for j in range(1095):
#     for i in range(8):
#         data.append(data_df[j,i])
data_np=2850*np.array(data)
max=np.max(data_np)
print(len(data_np))
P,F=Load_Model_Algorithm(max,data_np)
df_P=pd.DataFrame.from_dict(P,orient='index',columns=['P'])
df_F=pd.DataFrame.from_dict(F,orient='index',columns=['F'])
df=pd.concat([df_P,df_F],axis=1)
df.to_csv("Load_Model.csv")








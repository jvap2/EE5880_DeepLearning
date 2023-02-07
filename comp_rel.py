from reliability_functions import Seq_MC
import pandas as pd
import numpy as np

data_df=pd.read_csv("load_csv_data.csv")
rows=len(data_df.axes[0])
cols=len(data_df.axes[1])
data_load=np.empty(shape=rows*cols)
for r in range(rows):
    for c in range(cols):
        data_load[r*cols+c]=data_df.iloc[r,c]*2850

data=pd.read_csv("Gen_Reliability.csv")
data_np=data.to_numpy()
size=data_np[:,0]
units=data_np[:,1]
total_units=np.sum(units)
gen=np.empty(shape=total_units)
i=0
for u_size, u in zip(size, units):
    gen[i:i+u]=u_size
    i+=u
MTTF=data_np[:,2]
MTTR=data_np[:,3]
'''We have now taken data, a pandas dataframe and converted it to a numpy ndarray
We have now then taken each column and created its own ndarray containing pertinent information
including the size which is the power of a specific unit, the units, which is the number of units
of a certain size. Additionally, we have included the MTTF and MTTR of the respective units of different
power'''
Maintenance=data_np[:,4]
failure_rate=np.divide(np.ones(shape=np.shape(MTTF)),MTTF)
repair_rate=np.divide(np.ones(shape=np.shape(MTTR)),MTTR)
'''in lines 17 and 18, we are finding the failure and repair rates by taking the inverse of MTTF an MTTR respectively'''
t,f,p=Seq_MC(failure_rate,repair_rate,data_load,gen,total_units,3405)
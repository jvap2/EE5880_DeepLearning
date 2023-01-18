import pandas as pd
import numpy as np

data=pd.read_csv("Gen_Reliability.csv")
data_np=data.to_numpy()
size=data_np[:,0]
units=data_np[:,1]
MTTF=data_np[:,2]
MTTR=data_np[:,3]
Maintenance=data_np[:,4]
MTTF=data_np[:,2]
failure_rate=np.divide(np.ones(shape=np.shape(MTTF)),MTTF)
repair_rate=np.divide(np.ones(shape=np.shape(MTTR)),MTTR)

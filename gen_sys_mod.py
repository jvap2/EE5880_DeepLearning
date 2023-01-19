import pandas as pd
import numpy as np
from reliability_functions import Unit_Addition_Algorithm

data=pd.read_csv("Gen_Reliability.csv")
data_np=data.to_numpy()
size=data_np[:,0]
units=data_np[:,1]
MTTF=data_np[:,2]
MTTR=data_np[:,3]
Maintenance=data_np[:,4]
failure_rate=np.divide(np.ones(shape=np.shape(MTTF)),MTTF)
repair_rate=np.divide(np.ones(shape=np.shape(MTTR)),MTTR)
total_units=np.sum(units)
failure_for_size={}
repair_for_size={}
units_for_size={}
for (i,value) in enumerate(size):
    failure_for_size[value]=failure_rate[i]
    repair_for_size[value]=repair_rate[i]
    units_for_size[value]=units[i]
C_P, C_F= Unit_Addition_Algorithm(units_for_size,failure_for_size,repair_for_size)
d_CP=pd.DataFrame.from_dict(C_P, orient='index',columns=['P'])
d_CF=pd.DataFrame.from_dict(C_F, orient='index',columns=['F'])
final_df=pd.concat([d_CP,d_CF],axis=1)
print(final_df)
final_df.to_csv("Gen_Sys_Reliability.csv")





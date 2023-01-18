import pandas as pd
import numpy as np
import math
from math import comb
import reliability_functions
from reliability_functions import Unit_Addition_Algorithm

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
total_units=np.sum(units)
combos=0
for i in range(1,total_units+1):
    combos+=comb(32,i)
failure_for_size={}
repair_for_size={}
units_for_size={}
for (i,value) in enumerate(size):
    failure_for_size[value]=failure_rate[i]
    repair_for_size[value]=repair_rate[i]
    units_for_size[value]=units[i]
Unit_Addition_Algorithm(units_for_size,failure_for_size,repair_for_size)




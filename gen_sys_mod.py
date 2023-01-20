import pandas as pd
import numpy as np
from reliability_functions import Unit_Addition_Algorithm
'''Below we are going to begin by reading the Gen_Reliability data'''
data=pd.read_csv("Gen_Reliability.csv")
data_np=data.to_numpy()
size=data_np[:,0]
units=data_np[:,1]
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
total_units=np.sum(units)
'''Line 20 above corresponds to finding the total number of units in this specific problem'''
failure_for_size={}
repair_for_size={}
units_for_size={}
'''We are creating three dictionaries which will have keys of the the size, and the values will contain failure rate,
repair rate, and number of units for the respective keys.
Below, we are iterating through the array of size to fill out the necessary dictionaries for our function'''
for (i,value) in enumerate(size):
    failure_for_size[value]=failure_rate[i]
    repair_for_size[value]=repair_rate[i]
    units_for_size[value]=units[i]
'''Now, below we are calling the functions and then converting the results to a csv file'''
C_P, C_F= Unit_Addition_Algorithm(units_for_size,failure_for_size,repair_for_size)
d_CP=pd.DataFrame.from_dict(C_P, orient='index',columns=['P'])
d_CF=pd.DataFrame.from_dict(C_F, orient='index',columns=['F'])
final_df=pd.concat([d_CP,d_CF],axis=1)
final_df.to_csv("Gen_Sys_Reliability.csv")





import numpy as np
import pandas as pd
from reliability_functions import Generation_Reserve

gen_sys_df=pd.read_csv('Gen_Sys_Reliability.csv')
load_def=pd.read_csv('Load_Model.csv')
np_gen=gen_sys_df.to_numpy()
np_load=load_def.to_numpy()
gen_sys_idx=np_gen[:,0]
P_G=np_gen[:,1]
F_G=np_gen[:,2]
P_L=np_load[:,0]
F_L=np_load[:,1]
load_def_idx=np.array(load_def.index)
P_G_dict={}
F_G_dict={}
P_L_dict={}
F_L_dict={}

for (i,load_idx) in enumerate(load_def_idx):
    P_L_dict[load_idx]=P_L[i]
    F_L_dict[load_idx]=F_L[i]

for (j, gen_idx) in enumerate(gen_sys_idx):
    P_G_dict[gen_idx]=P_G[j]
    F_G_dict[gen_idx]=F_G[j]

Generation_Reserve(P_L_dict,F_L_dict,P_G_dict,F_G_dict,load_def_idx,gen_sys_idx)


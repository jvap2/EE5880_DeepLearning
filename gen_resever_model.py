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
P_L=np_load[:,1]
F_L=np_load[:,2]
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
n_g=32
P,F=Generation_Reserve(P_L_dict,F_L_dict,P_G_dict,F_G_dict,load_def_idx,gen_sys_idx)
LOLP=P[0]
LOLF=F[0]
EPNS=0
for i in range(-1885,1,1):
    EPNS+=P[i]
EPNS-=.5*(P[0]+P[-1885])
P_df=pd.DataFrame.from_dict(P,orient='index',columns=['P'])
F_df=pd.DataFrame.from_dict(F,orient='index',columns=['F'])
final=pd.concat([P_df,F_df],axis=1)
final.to_csv("Gen_Res.csv")
Final_Res={"LOLP":LOLP,"LOLF":LOLF,"EPNS":EPNS}
Final_df=pd.DataFrame.from_dict(Final_Res,orient='index')
Final_df.to_csv("Import_stat.csv")
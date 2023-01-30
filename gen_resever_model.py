import numpy as np
import pandas as pd
from reliability_functions import Generation_Reserve, Get_LOLP_LOLF,Gen_Res_V2

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
P={}
F={}
P,F=Generation_Reserve(P_L_dict,F_L_dict,P_G_dict,F_G_dict,load_def_idx,gen_sys_idx)
LOLE=P[0]
LOLF=F[0]
LOLP=P[0]/(24*365)
print(LOLP)
EPNS=0
for i in range(-max(P_L_dict.keys())+1,1,1):
    EPNS+=P[i]
EPNS+=(-.5*(P[0]+min(P.keys())))
P_df=pd.DataFrame.from_dict(P,orient='index',columns=['P'])
F_df=pd.DataFrame.from_dict(F,orient='index',columns=['F'])
final=pd.concat([P_df,F_df],axis=1)
final.to_csv("Gen_Res.csv")
Final_Res={"LOLE":LOLE,"LOLF":LOLF,"EPNS":EPNS, "LOLP": LOLP}
Final_df=pd.DataFrame.from_dict(Final_Res,orient='index')
Final_df.to_csv("Import_stat.csv")
P_test={}
F_test={}
P_test,F_test=Gen_Res_V2(P_L_dict,F_L_dict,P_G_dict,F_G_dict)
P_df=pd.DataFrame.from_dict(P_test,orient='index',columns=['P'])
F_df=pd.DataFrame.from_dict(F_test,orient='index',columns=['F'])
final=pd.concat([P_df,F_df],axis=1)
final.to_csv("Gen_Res_2.csv")
LOLE=P_test[0]
LOLF=F_test[0]
LOLP=P_test[0]/(24*365)
print(LOLP)
EPNS=0
for i in range(-max(P_L_dict.keys())+1,1,1):
    EPNS+=P_test[i]
EPNS+=(-.5*(P_test[0]+min(P_test.keys())))
Final_Res={"LOLE":LOLE,"LOLF":LOLF,"EPNS":EPNS, "LOLP": LOLP}
Final_df=pd.DataFrame.from_dict(Final_Res,orient='index')
Final_df.to_csv("Import_stat_2.csv")
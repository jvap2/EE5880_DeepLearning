import numpy as np
import math
from math import floor

def Unit_Addition_Algorithm(unit,failure_rate,repair_rate):
    '''
    Input Parameters
    unit: dtype=dict, keys are the size of a unit, the values are the number of said units of a particular size
    failure_rate: d_type=dict, keys are the size of a unit, the values are the failure rates of the specific units
    repair_rate: d_type=dict, keys are the size of a unit, the values are the repair rates of the specific units

    Returns:
    Cap_P: dtype=dict, keys are the size of a unit, and the values are the Probability
    Cap_F: dtype=dict, keys are the size of a unit, and the values are the Frequency
    '''
    num_units=sum(unit.values())
    p=np.ndarray(shape=num_units)
    q=np.ndarray(shape=num_units)
    mu=np.ndarray(shape=num_units)
    state_added=np.ndarray(shape=num_units)
    ''' Create the probability of success and failure, and the repair rates into arrays'''
    Max_Cap=0
    last_key=0
    for (i,key) in enumerate(failure_rate.keys()):
        f=failure_rate[key]/(failure_rate[key]+repair_rate[key])
        r=1-f
        u=unit[key]
        repair=repair_rate[key]
        p[last_key:last_key+u]=r
        q[last_key:last_key+u]=f
        mu[last_key:last_key+u]=repair
        state_added[last_key:last_key+u]=key
        last_key+=u
        Max_Cap+=key*u
    Cap_P={}
    Cap_F={}
    cap_list=np.empty(shape=(num_units+1), dtype=np.int64)
    cap_list[0]=0
    last_key=0
    for i in range(Max_Cap):
        Cap_P[i]=0
        Cap_F[i]=0
    print(cap_list[0])
    Cap_P[0]=1
    Cap_F[0]=0
    for i in range(1,int(state_added[0]+1)):
        Cap_P[i]=q[0]
        Cap_F[i]=q[0]*mu[0]
    P_1={}
    F_1={}
    P_i={}
    F_i={}
    P_i={}
    F_i={}
    for i in range(Max_Cap):
        P_1[i]=0
        F_1[i]=0
        P_i[i]=0
        F_i[i]=0
        P_i[i]=0
        F_i[i]=0
    g=0
    for i in range(1,32):
        g+=state_added[i]
        P_test=Cap_P.copy()
        F_test=Cap_F.copy()
        for j in range(int(g+1)):
            P_i=P_test[j]
            F_i=F_test[j]
            if j-state_added[i]<=0:
                P_j=1
                F_j=0
            else:
                P_j=Cap_P[j-state_added[i]]
                F_j=Cap_F[j-state_added[i]]
            P_1[j]=P_i*p[i]+P_j*q[i]
            F_1[j]=F_i*p[i]+F_j*q[i]+(P_j-P_i)*q[i]*mu[i]
        Cap_P=P_1.copy()
        Cap_F=F_1.copy()
    return Cap_P, Cap_F


def Load_Model_Algorithm(min,max,load):
    Z=1
    N_L=floor(max)//Z+1
    print(N_L)
    P_L={}
    F_L={}
    load_levels=np.linspace(0,max,N_L, dtype=np.int64)
    N_1=0
    N_2=len(load)-1
    print(N_2)
    i=N_1
    N_H=N_2-N_1+1
    for elements in load_levels:
        P_L[elements]=0
        F_L[elements]=0
    J=1E6    
    while i<=N_2:
        i_hour_contrib=load[i]
        J_1=floor(i_hour_contrib)//Z+1
        for j in range(1,J_1):
            P_L[j]+=1
        if i+1<=N_2:
            J_1=floor(load[i+1])//Z
        if J_1>=J:
            for q in range(J,J_1):
                F_L[q]+=1
        J=J_1
        i+=1
    for i,(prob,freq) in enumerate(zip(P_L.values(),F_L.values())):
        P_L[i]=prob/N_H
        F_L[i]=freq/N_H
    P_L[0]=0
    # P_L={key: item for key, item in P_L.items() if key>964 }
    # F_L={key: item for key, item in F_L.items() if key>964 }
    return P_L,F_L


def Generation_Reserve(P_L,F_L,P_G,F_G,load_idx, gen_idx):
    '''Generate a list of possible M_i'''
    M=np.empty(shape=(len(gen_idx),len(load_idx)))
    for (i,l_idx) in enumerate(gen_idx):
        for(j,g_idx) in enumerate(load_idx):
            M[i,j]=g_idx-l_idx
    M=M.flatten()
    M_final=np.unique(M)
    M_final=np.array(M_final,dtype=np.int64)
    gen_idx=np.array(gen_idx,dtype=np.int64)
    print(M_final)
    C_c=int(np.max(gen_idx))
    P={}
    F={}
    for value in M_final:
        P[value]=0
        F[value]=0
    P_L_List=list(P_L.keys())
    P_L_Max=max(P_L_List)
    P_L_Min=min(P_L_List)
    for i in range(len(M_final)-1,-1,-1):
        for j in range(len(gen_idx)):
            idx=C_c-gen_idx[j]-M_final[i]
            if idx>P_L_Max or idx<P_L_Min:
                P[M_final[i]]+=0
            else:
                if j==len(gen_idx)-1:
                    P[M_final[i]]+=P_G[j]*P_L[idx]
                    F[M_final[i]]+=F_G[j]*P_L[idx]+P_G[j]*F_L[idx]
                else:
                    P[M_final[i]]+=(P_G[j]-P_G[j+1])*P_L[idx]
                    F[M_final[i]]+=(F_G[j]-F_G[j+1])*P_L[idx]+(P_G[j]-P_G[j+1])*F_L[idx]
    for value in M_final:
        P[value]*=(365*24)
        F[value]*=(365*24)
    return P,F



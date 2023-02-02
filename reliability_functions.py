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


def Load_Model_Algorithm(max,load):
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
    P_L[max+1]=0
    F_L[max+1]=0
    while i<=N_2:
        i_hour_contrib=load[i] ## find the exact value at the the ith hour
        J_1=floor(i_hour_contrib)//Z+1 ##find the integer contribution from the ith hour
        for j in range(J_1+1):
            P_L[j]+=1
        if J_1>J:
            for q in range(J,J_1):
                F_L[q]+=1
        J=J_1
        i+=1
    for i,(prob,freq) in enumerate(zip(P_L.values(),F_L.values())):
        P_L[i]=prob/N_H
        F_L[i]=freq/N_H
    P_L[0]=1.0
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
    M_final=M_final[M_final>(-max(P_L.keys()))]
    gen_idx=np.array(gen_idx,dtype=np.int64)
    print(M_final)
    print(np.isin(-1,M_final))
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
            if idx>P_L_Max or idx<0:
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

def Get_LOLP_LOLF(P_G,F_G,P_L,F_L):
    PM,FM=0,0
    for i in range(1544):
        PCO=P_G[i]-P_G[i+1]
        FCO=F_G[i]-F_G[i+1]
        if (max(list(P_G.keys()))-i+2)>=max(list(P_L.keys())):
            pl=0
            fl=0
        else:
            pl=P_L[(max(list(P_G.keys()))-i+2)]
            fl=F_L[(max(list(P_G.keys()))-i+2)]
        PM+=PCO*pl
        FM+=FCO*pl
        FM+=PCO*fl
    PM*=(24*365)
    FM*=(24*365)
    return PM,FM


def Gen_Res_V2(P_L,F_L,P_G,F_G):
    Load=np.array(list(P_L.keys()))
    Gen=max(list(P_G.keys()))-list(P_G.keys())
    C=np.array(list(P_G.keys()))
    M=np.empty(shape=(len(Gen),len(Load)))
    for (i,l_idx) in enumerate(Gen):
        for(j,g_idx) in enumerate(Load):
            M[i,j]=g_idx-l_idx
    M=M.flatten()
    M_final=np.unique(M)
    M_final=np.array(M_final,dtype=np.int64)
    M_final=M_final[M_final>(-max(P_L.keys()))]
    P={}
    F={}

    PM=0
    FM=0
    C_c=max(list(P_G.keys()))
    count=0
    for M in M_final:
        P[M]=0
        F[M]=0
        for (j,Cap) in enumerate(C):
            idx=int(C_c-Cap-M)
            if idx>max(P_L.keys()) or idx<min(P_L.keys()):
                pl=0
                fl=0
            else:
                pl=P_L[idx]
                fl=F_L[idx]
            if j==len(C)-1:
                PCO=P_G[j]
                FCO=F_G[j]
            else:
                PCO=P_G[j]-P_G[j+1]
                FCO=P_G[j]-F_G[j+1]
            PM+=PCO*pl
            FM+=FCO*pl
            FM+=PCO*fl
        P[M]=PM 
        F[M]=FM 
        FM,PM=0,0
    for value in M_final:
        P[value]*=(365*24)
        F[value]*=(365*24)
    return P,F


def Seq_MC(fail,success,N):
    if len(fail)!=N or len(success)!=N:
        return
    MaxIter=1000000
    k=0
    t=0
    t_total=np.empty(shape=MaxIter)
    for i in range(MaxIter):
        state=np.ones(N)
        rng=np.random.default_rng(k)
        rand_num=rng.random(size=N)
        time=np.divide(-np.log(rand_num),fail)
        while not np.all(state==0):
            low_time=np.min(time)
            low_index=np.where(time==low_time)
            time-=low_time*np.ones(shape=N)
            t+=low_time
            low_rand_num=rng.random(size=len(low_index))
            if state[low_index]==1:
                time[low_index]=np.divide(-np.log(low_rand_num),fail[low_index])
                state[low_index]=0
            else:
                time[low_index]=np.divide(-np.log(low_rand_num),success[low_index])
                state[low_index]=1  
        k+=1
        t_total[i]=t
        t=0
    t_mean=np.mean(t_total)
    f=k/sum(t_total)
    p=f/np.sum(success)
    return t_mean,f,p
          

fail=np.array([.01,.012])
success=np.array([.125,1/6])
N=2
t,freq,prob=Seq_MC(fail,success,N)
print(t)
print(freq)
print(prob)
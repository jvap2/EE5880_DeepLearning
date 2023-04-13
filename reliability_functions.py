import numpy as np
import math
from math import floor
from random import random
from statistics import variance, mean
import pandas as pd
from scipy.optimize import differential_evolution,linprog, NonlinearConstraint
from scipy.special import logsumexp
from nn_reliability import Network
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import device
from nn_reliability import weights_init, Model


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


def Seq_MC(fail,success,load,gen,N,maxCap):
    err_tol=1e10
    LLD=[]
    LLO=[]
    ENS=[]
    LOLE=[]
    LOLF=[]
    LOEE=[]
    LLD_yr=0
    LLO_yr=0
    ENS_yr=0
    check_down=0
    n=0
    time=np.zeros(shape=N)
    Cap=0
    old_var=0
    while err_tol>100 and n<20000:
        n+=1
        state=np.ones(shape=N)
        rand_val=np.random.uniform(0,1,N)
        time=np.int_(np.floor(np.divide(-np.log(rand_val),fail)))
        t_n=0
        hr=0
        while hr <8759:
            T=np.min(time)
            T_idx=np.where(time==T)
            down_state_idx=np.where(state==0)
            time=time-T
            hr+=T
            if hr>8759:
                hr=8759
            Cap=maxCap-np.sum(gen[down_state_idx])
            for t in range(t_n,hr):
                if load[t]>=Cap:
                    if check_down==0:
                        LLO_yr+=1
                        check_down=1
                    LLD_yr+=1
                    ENS_yr+=abs(load[t]-Cap)
                else:
                    check_down=0
            t_n=hr
            for idx in T_idx:
                for value in idx:
                    if state[value]==0:
                        state[value]=1
                        time[value]=np.int_(np.floor(-np.log(np.random.rand(1))/fail[value]))
                    else:
                        state[value]=0
                        time[value]=np.int_(np.floor(-np.log(np.random.rand(1))/success[value]))
        LLD.append(LLD_yr)
        LLO.append(LLO_yr)
        ENS.append(ENS_yr)
        LLD_yr,LLO_yr,ENS_yr=0,0,0
        LOLE.append(mean(LLD))
        LOLF.append(mean(LLO))
        LOEE.append(mean(ENS))
        mu_LOLE=np.mean(LOLE)
        mu_LOLF=np.mean(LOLF)
        mu_LOEE=np.mean(LOEE)
        if n>1:
            var=max(variance(LOEE),variance(LOLF),variance(LOEE))
            err_tol=abs(var-old_var)
        print(err_tol)
        print(n)
    return mu_LOLE,mu_LOLF,mu_LOEE
        
            
def Seq_MC_Comp(load,gen,N,maxCap,A,T,T_max,W,Load_Buses,Load_Data,Gen_data):
    print(type(Gen_data))
    err_tol=1e10
    LLD=[]
    LLO=[]
    ENS=[]
    LOLE=[]
    LOLF=[]
    LOEE=[]
    LLD_yr=0
    LLO_yr=0
    ENS_yr=0
    check_down=0
    n=0
    time=np.zeros(shape=N)
    Cap=0
    old_var=0
    Curt=np.empty(shape=np.shape(A)[1])
    while err_tol>1e-6 and n<180:
        print("In progress, n=",n)
        n+=1
        state=np.ones(shape=N)
        rand_val=np.random.rand(N)
        count=0
        for i in range(Gen_data.shape[0]):
            rng = np.random.default_rng(count)
            Gen_data.iloc[i,5]=int(-np.log(rng.random())/Gen_data.iloc[i,2])
        t_n=0
        hr=0
        Pg=Gen_data.copy()
        while hr <8759:
            C=np.zeros(shape=np.shape(A)[1])
            time=Gen_data.iloc[:,5].min()
            T_idx_bus=Gen_data.index[Gen_data.iloc[:,5]==time].tolist()
            # print(T_idx_bus)
            # print(Gen_data['State Time'])
            down_state_idx=Gen_data.index[Gen_data.iloc[:,4]==0].tolist()
            # print(down_state_idx)
            Power_Down=Gen_data.loc[down_state_idx,'Cap'].sum()
            Gen_data.iloc[:,5]=Gen_data.iloc[:,5]-time
            state=Gen_data["State"].to_numpy()
            # print(state)
            hr+=time
            if hr>8759:
                hr=8759
            Cap=maxCap-Power_Down
            check=False
            for s in state:
                if s==0:
                    check=True
                    break
            if(check):
                for t in range(t_n,hr):
                    Temp_Load=np.array(np.copy(Load_Data),dtype=np.float64)
                    C=PSO_rel(A,T,T_max,Gen_data,load[t],Load_Buses,Temp_Load,Curt,W,Power_Down,alpha=0,beta=0)
                    # C=Linear_Programming(A,T,T_max,Gen_data,Load_Buses,Temp_Load,Curt)
                    count=0
                    if C[0]!=-1.0:
                            for i,_ in enumerate(Load_Buses):
                                Temp_Load[i]-=C[i]
                    if load[t]>=np.sum(Temp_Load) or C.all()==-1 or Temp_Load.any()<0:
                        if check_down==0:
                            LLO_yr+=1
                            check_down=1
                        LLD_yr+=1
                        ENS_yr+=abs(load[t]-Cap)
                    else:
                        check_down=0
            t_n=hr
            for value in T_idx_bus:
                if state[value]==0:
                    # print(value)
                    Gen_data.loc[value,'State']=1
                    Gen_data.loc[value,'State Time']=np.int_(np.floor(-np.log(np.random.rand(1))/Gen_data.loc[value,'Failure Rate']))
                    Gen_data.loc[value,'Cap']=Pg.loc[value,'Cap']
                else:
                    # print(value)
                    Gen_data.loc[value,'State']=0
                    Gen_data.loc[value,'State Time']=np.int_(np.floor(-np.log(np.random.rand(1))/Gen_data.loc[value,'Repair Rate']))
                    Gen_data.loc[value,'Cap']=0
        LLD.append(LLD_yr)
        LLO.append(LLO_yr)
        ENS.append(ENS_yr)
        LLD_yr,LLO_yr,ENS_yr=0,0,0
        LOLE.append(mean(LLD))
        LOLF.append(mean(LLO))
        LOEE.append(mean(ENS))
        mu_LOLE=np.mean(LOLE)
        mu_LOLF=np.mean(LOLF)
        mu_LOEE=np.mean(LOEE)
        if n>1:
            var=max(variance(LOEE),variance(LOLF),variance(LOEE))
            err_tol=abs(var-old_var)
            print(err_tol)
            old_var=var
        print(n)
    return mu_LOLE,mu_LOLF,mu_LOEE


def PSO_rel(A,T,T_max,Gen_Data,Load,Load_Buses,Load_Data,C,W,Pl,alpha=0,beta=0):
    '''
    1.) Swarm parameters such as weighting factor,
    acceleration constants and swarm size (S) are entered.
    2.)Initially particles are randomly generated whose
    vector dimensions will be equal swarm size.
    3.)Now, at each particle Cij is calculated considering
    all network constraints and it represents the initial
    random position of Pbest.
    4.)The particle which formulates the minimum Cij
    value will represent initial Gbest.
    5.)Velocity (Vx) and position (Xx) of each particle x in the
    swarm S is updated by,
        w=((wmax-wmin)/itermax)*iter
        w= inertia weight
        wmax and min are the respective maximum and minimum inertia weights
        itermax is the maximum number of iterations
    Vx(k+1)=w*Vx(k)+c1r1(Pbest_x(k)-X_x(k))+c2r2(Gbest(k)-X_x(k))
        c1 and c2 are the acceleration constants
        r1 and r2 are random numbers between [0,1]
        Pbest_x is the best objective function value for particle x at iteration k
        Gbest(k) is the best objective value among all particles
    Xx(k+1)=Xx(k)+Vx(k+1)
    6.) iter <- iter + 1
    7.) Objective function is calculated for all constraints (see Canadian dood paper)
    8.) Update Pbest and Gbest
    9.) Repeat until convergence
    '''
    '''We need to realize the load at each bus, hence this will be the best value'''
    '''This is our objective function'''
    max_iter = 200
    LD=np.zeros(shape=(np.shape(A)[1]))
    GD=np.zeros(shape=(np.shape(A)[1]))
    alpha=np.ones(shape=np.shape(C))
    bus_list = [list(set([val for _,val in Gen_Data.loc[:,'Bus'].items()]))]
    for val in bus_list[0]:
        GD[val-1]=Gen_Data.loc[Gen_Data['Bus']==val,'Cap'].sum()
    
    for (i,bus) in enumerate(Load_Buses):
        LD[bus-1]=Load_Data[i]
    L=np.shape(Load_Data)
    bounds=[]
    for i in range(len(C)):
        bounds.append((0,LD[i]))
    # lb=-np.inf*np.ones(shape=np.shape(T_max))
    # ub=T_max
    # nlc=NonlinearConstraint(NL_Constraint,lb,ub)
    x=differential_evolution(Constraints,bounds, args=(A,GD,LD,T_max,Pl))
    C=x.x
    print("Curtailment vector")
    print(C)
    T=np.matmul(A,(GD+C-LD))
    check=True
    for (i,val) in enumerate(T):
        if abs(val)>T_max[i]:
            check=False
            break
    if check and ((GD+C).all()==LD.all()) and sum(GD)<3405 and C.all()<=LD.all():
        return C
    else:
        return -1*np.ones(shape=np.shape(C))
    
def Linear_Programming(A,T,T_max,Gen_Data,Load_Buses,Load_Data,C):
    LD=np.empty(shape=(np.shape(A)[1]))
    GD=np.empty(shape=(np.shape(A)[1]))
    count=0
    for i in range(np.shape(A)[1]):
        if i==Gen_Data.loc[:,'Bus'].any()-1:
            GD[i]=Gen_Data.loc[i+1,'Cap']
        else:
            GD[i]=0
        if i==Load_Buses.any()-1:
            LD[i]=Load_Data[count]
            count+=1
        else:
            LD[i]=0
    print(GD)
    print(LD)
    Curt_Const=np.ones(shape=np.shape(C))
    T=np.reshape(T,newshape=((38,)))
    A_ub = A 
    b_ub = T_max - np.matmul(A,GD) + np.matmul(A,LD)
    A_eq = A
    b_eq = T- np.matmul(A,GD) + np.matmul(A,LD)
    bounds=[]
    for i in range(len(C)):
        bounds.append((0,LD[i]))
    result=linprog(Curt_Const,A_ub,b_ub,A_eq,b_eq,bounds,method="highs", integrality=2)
    sol=result.x
    print(result)
    if result.success!=False:
        if sol.all()<LD.all():
            print("Successful")
            return sol
        else:
            return -1*np.ones(shape=np.shape(C))
    else:
        return -1*np.ones(shape=np.shape(C))



def Constraints(C,A,GD,LD,T_max,Pl):
    # C=(LD*((GD.sum())-(LD.sum())-Pl))/(LD.sum())
    Curt=np.dot(np.ones(shape=np.shape(C)),C)
    T=np.matmul(A,(GD+C-LD))
    if T.all()<=T_max.all() and C.all()<=LD.all() and ((GD+C).all()==LD.all()) and sum(GD)<3405:
        return Curt
    else: 
        return 10
    

    

def Seq_MC_NN(load,gen,N,maxCap,A,T,T_max,W,Load_Buses,Load_Data,Gen_data):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    err_tol=1e10
    print(dev)
    LLD=[]
    LLO=[]
    ENS=[]
    LOLE=[]
    LOLF=[]
    LOEE=[]
    LLD_yr=0
    LLO_yr=0
    ENS_yr=0
    check_down=0
    n=0
    time=np.zeros(shape=N)
    Cap=0
    old_var=0
    LD=np.empty(shape=(np.shape(A)))
    GD=np.empty(shape=(np.shape(A)))
    mod=Model(3,10,1)
    weights_init(model=mod)
    while err_tol>1000 and n<20:
        print("In progress, n=",n)
        n+=1
        state=np.ones(shape=N)
        rand_val=np.random.rand(N)
        count=0
        for i in range(Gen_data.shape[0]):
            Gen_data.iloc[i,5]=int(-np.log(rand_val[count])/Gen_data.iloc[i,2])
        t_n=0
        hr=0
        Pg=Gen_data.copy()
        while hr <8759:
            Temp_Load=np.array(np.copy(Load_Data),dtype=np.float64)
            time=Gen_data.iloc[:,5].min()
            T_idx_bus=Gen_data.index[Gen_data.iloc[:,5]==time].tolist()
            down_state_idx=Gen_data.index[Gen_data.iloc[:,4]==0].tolist()
            Power_Down=Gen_data.loc[down_state_idx,'Cap'].sum()
            Gen_data.iloc[:,5]=Gen_data.iloc[:,5]-time
            hr+=time
            if hr>8759:
                hr=8759
            Cap=maxCap-Power_Down
            if(Gen_data.loc[:,"State"].any()==0):
                for t in range(t_n,hr):
                    count=0
                    for i in range(np.shape(A)[1]):
                        if i==Gen_data.loc[:,'Bus'].any()-1:
                            GD[i]=Gen_data.loc[i+1,'Cap']
                        else:
                            GD[i]=0
                        if i==Load_Buses.any()-1:
                            LD[i]=Load_Data[count]
                            count+=1
                        else:
                            LD[i]=0
                    input=np.empty(shape=(np.shape(A)[1],3))
                    input[:,0]=GD
                    input[:,1]=LD
                    input[:,2]=np.ones(np.shape(A)[1])*Power_Down
                    input=torch.from_numpy(input).float().requires_grad_().to(device=dev)
                    A_T=torch.from_numpy(A).float().to(device=dev)
                    T_max_T=torch.from_numpy(T_max).float().to(device=dev)
                    L_T=torch.tensor(load[t]).to(device=dev)
                    print("Evaluating Curtailment at hour ", t)
                    C=Network(mod,3,10,1,input,load[t],A_T,T_max_T).cpu().detach().numpy()
                    for i in range(np.shape(A)[1]):
                        count=0
                        if i==Load_Buses.any()-1:
                            Temp_Load[count]-=C[i]
                            count+=1
                    if load[t]>=np.sum(Temp_Load) or Temp_Load.any()<0:
                        if check_down==0:
                            LLO_yr+=1
                            check_down=1
                        LLD_yr+=1
                        ENS_yr+=abs(load[t]-Cap)
                    else:
                        ## G_2 or G_1
                        check_down=0
            t_n=hr
            for value in T_idx_bus:
                if state[value]==0:
                    Gen_data.loc[value,'State']=1
                    Gen_data.loc[value,'State Time']=np.int_(np.floor(-np.log(np.random.rand(1))/Gen_data.loc[value,'Failure Rate']))
                    Gen_data.loc[value,'Cap']=Pg.loc[value,'Cap']
                else:
                    Gen_data.loc[value,'State']=0
                    Gen_data.loc[value,'State Time']=np.int_(np.floor(-np.log(np.random.rand(1))/Gen_data.loc[value,'Repair Rate']))
                    Gen_data.loc[value,'Cap']=0
        LLD.append(LLD_yr)
        LLO.append(LLO_yr)
        ENS.append(ENS_yr)
        LLD_yr,LLO_yr,ENS_yr=0,0,0
        LOLE.append(mean(LLD))
        LOLF.append(mean(LLO))
        LOEE.append(mean(ENS))
        mu_LOLE=np.mean(LOLE)
        mu_LOLF=np.mean(LOLF)
        mu_LOEE=np.mean(LOEE)
        if n>1:
            var=max(variance(LOEE),variance(LOLF),variance(LOEE))
            err_tol=abs(var-old_var)
        print(err_tol)
        print(n)
    return mu_LOLE,mu_LOLF,mu_LOEE
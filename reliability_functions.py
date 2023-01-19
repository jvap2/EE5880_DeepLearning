import numpy as np

def Unit_Addition_Algorithm(unit,failure_rate,repair_rate):
    num_units=sum(unit.values())
    p=np.ndarray(shape=num_units)
    q=np.ndarray(shape=num_units)
    mu=np.ndarray(shape=num_units)
    Max_Cap=0
    last_key=0
    for (i,key) in enumerate(failure_rate.keys()):
        f=failure_rate[key]/(failure_rate[key]+repair_rate[key])
        r=1-f
        u=unit[key]
        repair=repair_rate[key]
        p[last_key:last_key+u]=f
        q[last_key:last_key+u]=r
        mu[last_key:last_key+u]=repair
        last_key+=unit[key]
        Max_Cap+=key*u
    Cap_P={}
    Cap_F={}
    cap_list=np.empty(shape=(num_units+1))
    cap_list[0]=0
    last_key=0
    for key in unit.keys():
        for j in range(unit[key]):
            cap_list[last_key+j+1]=cap_list[last_key+j]+key
        last_key+=unit[key]
    Cap_P[cap_list[0]]=1
    Cap_F[cap_list[0]]=0
    Cap_P[cap_list[1]]=q[0]
    Cap_P[cap_list[1]]=q[0]*mu[0]
    Old_Cap_P=Cap_P
    Old_Cap_F=Cap_F
    state=24
    state_index=2
    while state<Max_Cap:
        for i in range(1,state_index+1):
            for j in range(i):
                pass
    
    

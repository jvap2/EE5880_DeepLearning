import numpy as np

def Unit_Addition_Algorithm(unit,failure_rate,repair_rate):
    f=np.empty(shape=len(failure_rate))
    r=np.empty(shape=len(repair_rate))
    u=np.empty(shape=len(unit), dtype=np.int64)
    Max_Cap=0
    for (i,key) in enumerate(failure_rate.keys()):
        f[i]=failure_rate[key]/(failure_rate[key]+repair_rate[key])
        r[i]=1-f[i]
        u[i]=unit[key]
        Max_Cap+=key*u[i]
    num_units=np.sum(u)
    Cap_P={}
    Cap_F={}
    cap_list=np.empty(shape=(num_units+1))
    cap_list[0]=0
    last_key=0
    for key in unit.keys():
        for j in range(unit[key]):
            cap_list[last_key+j+1]=cap_list[last_key+j]+key
        last_key+=unit[key]
    
    

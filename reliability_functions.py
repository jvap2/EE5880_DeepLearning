import numpy as np

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
        last_key+=unit[key]
        Max_Cap+=key*u
    Cap_P={}
    Cap_F={}
    cap_list=np.empty(shape=(num_units+1), dtype=np.int64)
    cap_list[0]=0
    last_key=0
    for key in unit.keys():
        for j in range(unit[key]):
            cap_list[last_key+j+1]=cap_list[last_key+j]+key
            Cap_P[cap_list[last_key+j+1]]=0
            Cap_F[cap_list[last_key+j+1]]=0
        last_key+=unit[key]
    print(cap_list[0])
    Cap_P[cap_list[0]]=1
    Cap_F[cap_list[0]]=0
    Cap_P[cap_list[1]]=q[0]
    Cap_F[cap_list[1]]=q[0]*mu[0]
    Old_Cap_P,Old_Cap_F=Cap_P.copy(),Cap_F.copy()
    state=24
    state_index=2
    while state<Max_Cap:
        for i in range(1,state_index):
            for j in range(i):
                Cap_P[cap_list[i]]=Old_Cap_P[cap_list[i]]*p[state_index]+Old_Cap_P[cap_list[j]]*q[state_index]
                Cap_F[cap_list[i]]=Old_Cap_F[cap_list[i]]*p[state_index]+Old_Cap_F[cap_list[j]]*q[state_index]+(Old_Cap_P[cap_list[j]]-Old_Cap_P[cap_list[i]])*q[state_index]*mu[state_index]
        Old_Cap_P=Cap_P.copy()
        Old_Cap_F=Cap_F.copy()
        state=cap_list[state_index+1]
        state_index+=1
    return Cap_P, Cap_F
    

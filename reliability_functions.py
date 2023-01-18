import numpy as np

def Unit_Addition_Algorithm(unit,failure_rate,repair_rate):
    f=np.empty(shape=len(failure_rate))
    r=np.empty(shape=len(repair_rate))
    u=np.empty(shape=len(unit))
    for (i,key) in enumerate(failure_rate.keys()):
        f[i]=failure_rate[key]/(failure_rate[key]+repair_rate[key])
        r[i]=1-f[i]
        u[i]=unit[key]
    
import pandas as pd
import numpy as np
from scipy.linalg import inv
import scipy

df=pd.read_csv("Impedance.csv")
row=df['From'].to_numpy(dtype=np.int64)
col=df['To'].to_numpy(dtype=np.int64)
r_pu=df['R(pu)'].to_numpy()
x_pu=df['X(pu)'].to_numpy()
admit=1/r_pu

Y=np.zeros(shape=(max(col),max(col)))
for i,(r,c) in enumerate(zip(row,col)):
    Y[r-1,c-1]-=admit[i]
    Y[c-1,r-1]-=admit[i]


for i,q in enumerate(Y):
    Y[i,i]=-sum(q)

Z=inv(Y)

A=np.zeros(shape=(len(admit),max(col)))

for i,(r,c) in enumerate(zip(row,col)):
    A[i,:]=(Z[r-1,:]-Z[c-1,:])/x_pu[i]

print(A)

PD=pd.read_csv("Bus_Load_Data_RTS.csv")
PG=pd.read_csv("Generating_Units.csv")


PD_dict=PD.to_dict()
PG_dict=PG.to_dict()

print(PG_dict)
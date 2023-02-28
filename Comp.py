import pandas as pd
import numpy as np
from scipy.linalg import inv
import scipy

df=pd.read_csv("Impedance.csv")
row=df['From'].to_numpy(dtype=np.int64)
col=df['To'].to_numpy(dtype=np.int64)
r_pu=df['R(pu)'].to_numpy()
x_pu=df['X(pu)'].to_numpy()
length=df['Length(miles)'].to_numpy()
T_max=df['Rating'].to_numpy()
length=(length-np.min(length))/(np.max(length)-np.min(length))
admit=1/r_pu

Y=np.zeros(shape=(max(col),max(col)))
W=np.zeros(shape=(max(col),max(col)))
for i,(r,c) in enumerate(zip(row,col)):
    Y[r-1,c-1]-=admit[i]
    Y[c-1,r-1]-=admit[i]
    W[r-1,c-1]=length[i]
    W[c-1,r-1]=length[i]


for i,q in enumerate(Y):
    Y[i,i]=-sum(q)

Z=inv(Y)
L=len(admit)
NS=max(col)

A=np.zeros(shape=(L,NS))

for i,(r,c) in enumerate(zip(row,col)):
    A[i,:]=(Z[r-1,:]-Z[c-1,:])/x_pu[i]


PD=pd.read_csv("Bus_Load_Data_RTS.csv")
PG=pd.read_csv("Generating_Units.csv")


Load_Buses=PD['Bus'].to_numpy()
Loads=PD['MW'].to_numpy()
Gen_Buses=PG['Bus'].to_numpy()
Gen_Units_MW=np.transpose(np.array([PG['Unit 1'].to_numpy(),
                       PG['Unit 2'].to_numpy(),
                       PG['Unit 3'].to_numpy(),
                       PG['Unit 4'].to_numpy(),
                       PG['Unit 5'].to_numpy(),
                       PG['Unit 6'].to_numpy()]))

NC=len(Load_Buses)
alpha=np.empty(shape=(NC,3))
beta=np.empty(shape=(NC,3))
for i,row in enumerate(alpha):
    row=np.random.random(size=(1,3))
    row/=np.sum(row)
    alpha[i,:]=row
    beta[i,:]=row*Loads[i]
print(beta)

##Test
T=np.empty(shape=(L,1))
for i,row in enumerate(T):
    PG_sum=np.sum(Gen_Units_MW)-np.sum(Loads)
    row=np.sum(A[i,:])*PG_sum
    T[i]=row

import pandas as pd
import numpy as np
from scipy.linalg import inv
import scipy
from reliability_functions import Seq_MC_Comp,Seq_MC_NN

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


##Test
T=np.empty(shape=(L,1))
# for i,row in enumerate(T):
#     PG_sum=np.sum(Gen_Units_MW)-np.sum(Loads)
#     row=np.sum(A[i,:])*PG_sum
#     T[i]=row
data_df=pd.read_csv("load_csv_data.csv")
rows=len(data_df.axes[0])
cols=len(data_df.axes[1])
data_load=np.empty(shape=rows*cols)
for r in range(rows):
    for c in range(cols):
        data_load[r*cols+c]=data_df.iloc[r,c]*2850

data=pd.read_csv("Gen_Reliability.csv")
data_np=data.to_numpy()
size=data_np[:,0]
units=data_np[:,1]
total_units=np.sum(units)
# gen=np.empty(shape=total_units)
i=0
j=0
MTTF=data_np[:,2]
MTTR=data_np[:,3]
failure_rate={}
repair_rate={}

for u_size, u in zip(size, units):
    gen=u_size
    failure_rate[gen]=1/MTTF[j]
    repair_rate[gen]=1/MTTR[j]
    i+=u
    j+=1

Gen=[]
for i,bus in enumerate(Gen_Buses):
    for j,row in enumerate(Gen_Units_MW[i]):
        if row!=0:
            Gen.append({"Bus": bus,"Cap":row,"Failure Rate":failure_rate[row],"Repair Rate":repair_rate[row],"State":1,"State Time":0})


Gen_df=pd.DataFrame(Gen)

print(Gen_df)

LOLE,LOLF,LOEE=Seq_MC_Comp(data_load,gen,total_units,3405,A,T,T_max,W,Load_Buses,Loads,Gen_df)
print(LOLE)
print(LOLF)
print(LOEE)

# LOLE,LOLF,LOEE=Seq_MC_NN(data_load,gen,total_units,3405,A,T,T_max,W,Load_Buses,Loads,Gen_df)
# print(LOLE)
# print(LOLF)
# print(LOEE)

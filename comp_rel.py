from reliability_functions import Seq_MC
import pandas as pd

data_df=pd.read_csv("load_csv_data.csv")
data_dict={}
rows=len(data_df.axes[0])
cols=len(data_df.axes[1])
for r in range(rows):
    for c in range(cols):
        data_dict[r*cols+c+1]=data_df.iloc[r,c]*2850

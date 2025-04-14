import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
df=pd.read_csv('try1.csv',header=None,usecols=[0,1,2],skiprows=1)
df.columns=['classlabel','Water','Acid']
print("Original Database:")
print(df)
scaling=MinMaxScaler()
scaled_value=scaling.fit_transform(df[['Water','Acid']])
df[['Water','Acid']]=scaled_value
print("\n Dataframe after MinMax Sacling")
scaling=StandardScaler()
scaled_standardvalue=scaling.fit_transform(df[['Water','Acid']])
df[['Water','Acid']]=scaled_standardvalue
print("\n Dataframe after Standard Scaling")
print(df)

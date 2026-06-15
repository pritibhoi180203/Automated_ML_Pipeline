import pandas as pd

df = pd.read_csv("data/titanic.csv")

df.fillna(0,inplace=True)

print(df.isnull().sum())
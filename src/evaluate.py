import pandas as pd
import joblib

from sklearn.metrics import accuracy_score

model = joblib.load("models/model.pkl")

df = pd.read_csv("data/titanic.csv")

df.fillna(0,inplace=True)

X = df[['Pclass','Age','Fare']]
y = df['Survived']

pred = model.predict(X)

print(accuracy_score(y,pred))
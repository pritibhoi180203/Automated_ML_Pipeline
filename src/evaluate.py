import pandas as pd
import joblib

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/titanic.csv")

df = df[["Pclass","Sex","Age","Fare","Survived"]]

df["Age"].fillna(df["Age"].median(), inplace=True)

df["Sex"] = df["Sex"].map({
    "male":0,
    "female":1
})

X = df.drop("Survived",axis=1)
y = df["Survived"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = joblib.load("models/model.pkl")

pred = model.predict(X_test)

acc = accuracy_score(y_test,pred)

print("Accuracy:",acc)
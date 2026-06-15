import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/titanic.csv")

df = df[['Pclass','Sex','Age','Fare','Survived']]

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Sex'] = df['Sex'].map({
    'male': 0,
    'female': 1
})

X = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("data.csv")

X = df[["hours"]]
y = df["score"]

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
print("Model trained & saved")

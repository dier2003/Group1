import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

income = np.random.lognormal(10, 2, n)
age = np.random.randint(18, 70, n)
transactions = np.random.randint(1, 100, n)

risk_score = (
    0.000001 * income
    + 0.02 * transactions
    - 0.01 * age
)

target = (risk_score > np.median(risk_score)).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "transactions": transactions,
    "risk_score": risk_score,
    "target": target
})

df.to_csv("group1_dataset.csv", index=False)

print(df.head())
print(df.shape)
print(df["target"].value_counts())
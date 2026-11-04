import pandas as pd

# Dataset load karo
df = pd.read_csv("dataset/diabetes.csv")

# Dataset ki basic information
print("========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== OUTCOME DISTRIBUTION ==========")
print(df["Outcome"].value_counts())
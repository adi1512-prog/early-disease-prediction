import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("Starting preprocessing...")

# Load dataset
df = pd.read_csv("dataset/diabetes.csv")

# Replace invalid zeros
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

# Fill missing values
for col in cols:
    df[col] = df[col].fillna(df[col].median())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

print("Preprocessing completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Scaler saved successfully!")
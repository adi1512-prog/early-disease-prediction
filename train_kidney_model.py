import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Starting Kidney Disease training...")

df = pd.read_csv("dataset/kidney/kidney.csv")

# Remove ID
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Rename target
if "classification" in df.columns:
    df = df.rename(columns={"classification": "class"})

# Clean spaces
df = df.apply(lambda col: col.map(
    lambda x: x.strip() if isinstance(x, str) else x
))

# Replace ? with NaN
df = df.replace("?", np.nan)

# Numeric columns
numeric_cols = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Binary categorical columns
binary_cols = [
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "pe", "ane"
]

for col in binary_cols:
    df[col] = df[col].map({
        "normal": 0,
        "abnormal": 1,
        "notpresent": 0,
        "present": 1,
        "no": 0,
        "yes": 1
    })

# Appetite
df["appet"] = df["appet"].map({
    "good": 0,
    "poor": 1
})

# Target
df["class"] = df["class"].map({
    "notckd": 0,
    "ckd": 1
})

# Fill missing values
for col in df.columns:
    if col != "class":
        df[col] = df[col].fillna(df[col].median())

# Remove rows without target
df = df.dropna(subset=["class"])

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Prediction
pred = model.predict(X_test_scaled)

print("\n===== KIDNEY DISEASE MODEL =====")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))

# Save model
joblib.dump(
    model,
    "model/kidney/random_forest_model.pkl"
)

joblib.dump(
    scaler,
    "model/kidney/scaler.pkl"
)

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance.to_csv(
    "model/kidney/feature_importance.csv",
    index=False
)

print("\n========================================")
print("Kidney Disease model saved successfully!")
print("========================================")
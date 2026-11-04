import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/diabetes.csv")

# Replace invalid zeros
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

# Fill missing values
for col in cols:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------
# 1. Outcome Distribution
# -----------------------------

plt.figure(figsize=(6, 4))

sns.countplot(x="Outcome", data=df)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Number of Patients")

plt.savefig("outcome_distribution.png")
plt.show()


# -----------------------------
# 2. Glucose Distribution
# -----------------------------

plt.figure(figsize=(7, 4))

sns.histplot(
    data=df,
    x="Glucose",
    hue="Outcome",
    kde=True
)

plt.title("Glucose Distribution")
plt.savefig("glucose_distribution.png")
plt.show()


# -----------------------------
# 3. BMI vs Glucose
# -----------------------------

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Glucose",
    y="BMI",
    hue="Outcome"
)

plt.title("Glucose vs BMI")
plt.savefig("glucose_bmi.png")
plt.show()


# -----------------------------
# 4. Correlation Heatmap
# -----------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.savefig("correlation_heatmap.png")
plt.show()


# -----------------------------
# 5. Model Comparison
# -----------------------------

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_accuracy = accuracy_score(
    y_test,
    lr.predict(X_test)
)

rf_accuracy = accuracy_score(
    y_test,
    rf.predict(X_test)
)

print("\n===== MODEL COMPARISON =====")

print("Logistic Regression:",
      round(lr_accuracy * 100, 2), "%")

print("Random Forest:",
      round(rf_accuracy * 100, 2), "%")


# -----------------------------
# 6. Feature Importance
# -----------------------------

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)

plt.figure(figsize=(8, 5))

importance.plot(kind="bar")

plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()
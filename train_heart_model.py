import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("dataset/heart/heart.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", list(df.columns))


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop("target", axis=1)
y = df["target"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# CREATE HEART MODEL FOLDER
# =========================================================

os.makedirs("model/heart", exist_ok=True)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_scaled
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred
)

logistic_recall = recall_score(
    y_test,
    logistic_pred
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred
)


# =========================================================
# RANDOM FOREST
# =========================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest.fit(
    X_train_scaled,
    y_train
)

rf_pred = random_forest.predict(
    X_test_scaled
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n========================================")
print("HEART DISEASE MODEL RESULTS")
print("========================================")

print("\n----- Logistic Regression -----")

print(
    "Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(logistic_precision, 2)
)

print(
    "Recall:",
    round(logistic_recall, 2)
)

print(
    "F1 Score:",
    round(logistic_f1, 2)
)


print("\n----- Random Forest -----")

print(
    "Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(rf_precision, 2)
)

print(
    "Recall:",
    round(rf_recall, 2)
)

print(
    "F1 Score:",
    round(rf_f1, 2)
)


# =========================================================
# SAVE MODELS
# =========================================================

joblib.dump(
    logistic_model,
    "model/heart/logistic_model.pkl"
)

joblib.dump(
    random_forest,
    "model/heart/random_forest_model.pkl"
)

joblib.dump(
    scaler,
    "model/heart/scaler.pkl"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

print("\n----- Feature Importance -----")
print(importance)


# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

importance.to_csv(
    "model/heart/feature_importance.csv"
)


print("\n========================================")
print("Heart Disease models saved successfully!")
print("========================================")
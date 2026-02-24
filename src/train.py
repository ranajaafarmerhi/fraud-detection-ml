import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale Time & Amount
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Model
model = LogisticRegression(max_iter=1000, class_weight='balanced')

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC Score:", roc_auc)

thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]

print("\nThreshold Tuning:\n")

for t in thresholds:
    y_pred_thresh = (y_proba >= t).astype(int)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    print(f"Threshold: {t}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("-" * 30)

print("\n================ RANDOM FOREST ================\n")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("Classification Report:\n")
print(classification_report(y_test, rf_pred))

rf_roc_auc = roc_auc_score(y_test, rf_proba)
print("ROC-AUC Score:", rf_roc_auc)

# Threshold tuning for RF
print("\nThreshold Tuning (Random Forest):\n")

thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]

for t in thresholds:
    rf_pred_thresh = (rf_proba >= t).astype(int)
    precision = precision_score(y_test, rf_pred_thresh)
    recall = recall_score(y_test, rf_pred_thresh)
    print(f"Threshold: {t}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("-" * 30)



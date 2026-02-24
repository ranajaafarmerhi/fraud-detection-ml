import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

# Helper function: threshold evaluation
def evaluate_thresholds(y_true, y_proba, thresholds=[0.5, 0.7, 0.9, 0.95, 0.99]):
    for t in thresholds:
        y_pred_thresh = (y_proba >= t).astype(int)
        precision = precision_score(y_true, y_pred_thresh)
        recall = recall_score(y_true, y_pred_thresh)
        print(f"Threshold: {t}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("-" * 30)

# Load dataset
df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale Time & Amount
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)

# Predictions and probabilities
y_pred = log_model.predict(X_test)
y_proba = log_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

print("\nThreshold Tuning:")
evaluate_thresholds(y_test, y_proba)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===\n")
print(classification_report(y_test, rf_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_proba))

print("\nThreshold Tuning (Random Forest):")
evaluate_thresholds(y_test, rf_proba)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression with Pipeline to fix scaling & convergence
log_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', LogisticRegression(max_iter=3000, class_weight='balanced'))
])

log_cv_scores = cross_val_score(log_pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
print("\n=== Logistic Regression CV ===")
print("Scores:", log_cv_scores)
print("Mean:", np.mean(log_cv_scores))
print("Std:", np.std(log_cv_scores))
print("-" * 40)

# Random Forest CV
rf_model_cv = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_cv_scores = cross_val_score(rf_model_cv, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
print("\n=== Random Forest CV ===")
print("Scores:", rf_cv_scores)
print("Mean:", np.mean(rf_cv_scores))
print("Std:", np.std(rf_cv_scores))
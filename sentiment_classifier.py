import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Generate Binary Classification Dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5, 
    random_state=42
)

# Train-Test Split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Train Logistic Regression ---
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("="*50)
print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred_lr))

# --- 3. Save Confusion Matrix & ROC Curve Plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# ROC-AUC Curve
RocCurveDisplay.from_estimator(lr_model, X_test, y_test, ax=axes[1])
axes[1].set_title('Logistic Regression ROC Curve')

plt.tight_layout()
plt.savefig('logistic_regression_plots.png')
plt.show()

# --- 4. Train Random Forest Classifier & Compare ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("="*50)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred_rf))

# Model Comparison Summary
print("="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"{'Metric':<15} | {'Logistic Regression':<20} | {'Random Forest':<15}")
print("-" * 55)
print(f"{'Accuracy':<15} | {accuracy_score(y_test, y_pred_lr):<20.4f} | {accuracy_score(y_test, y_pred_rf):<15.4f}")
print(f"{'Precision':<15} | {precision_score(y_test, y_pred_lr):<20.4f} | {precision_score(y_test, y_pred_rf):<15.4f}")
print(f"{'Recall':<15} | {recall_score(y_test, y_pred_lr):<20.4f} | {recall_score(y_test, y_pred_rf):<15.4f}")
print(f"{'F1-Score':<15} | {f1_score(y_test, y_pred_lr):<20.4f} | {f1_score(y_test, y_pred_rf):<15.4f}")
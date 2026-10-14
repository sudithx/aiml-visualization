import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate a dataset
X, y = make_classification(
    n_samples=1000, n_features=6, n_informative=4, random_state=42
)

# 2. Train/Test Split (Preventing Data Leakage by splitting BEFORE scaling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}"
)

# 3. Scale features properly using only the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(
    X_test
)  # Apply transform only (no fit on test!)

# 4. Initialize model
model = RandomForestClassifier(random_state=42)

# 5. Cross-Validation (Stratified K-Fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model, X_train_scaled, y_train, cv=cv, scoring="accuracy"
)

print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. Final Evaluation on Test Set
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")
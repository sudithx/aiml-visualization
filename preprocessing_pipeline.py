import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Create a synthetic DataFrame to simulate real-world messy data
X_raw, y = make_classification(
    n_samples=1000, 
    n_features=4, 
    n_informative=3, 
    n_redundant=0, 
    n_repeated=0, 
    random_state=42
)
df = pd.DataFrame(X_raw, columns=["num_feature_1", "num_feature_2", "num_feature_3", "num_feature_4"])

# Introduce some missing values and a categorical feature for realism
df.loc[10:20, "num_feature_1"] = np.nan
df["cat_feature"] = np.random.choice(["Low", "Medium", "High"], size=len(df))

# 2. Train/Test Split (Always split before preprocessing/pipeline execution)
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define Preprocessing Steps for Numeric and Categorical Data
numeric_features = ["num_feature_1", "num_feature_2", "num_feature_3", "num_feature_4"]
categorical_features = ["cat_feature"]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 4. Create Full End-to-End Pipeline with Classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Evaluate Pipeline using Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

print("--- End-to-End Pipeline Evaluation ---")
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. Fit on full training data and evaluate on test data
pipeline.fit(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Test Set Accuracy: {test_score:.4f}")
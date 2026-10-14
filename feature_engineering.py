import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

# 0. Generate a synthetic dataset for practice
X_raw, y_raw = make_classification(
    n_samples=200, n_features=8, n_informative=5, random_state=42
)
df = pd.DataFrame(X_raw, columns=[f"num_feat_{i}" for i in range(8)])

# Add categorical columns to test encoders
df["city"] = np.random.choice(["New York", "London", "Tokyo"], size=200)
df["education_level"] = np.random.choice(
    ["High School", "Bachelor", "Master"], size=200
)
df["target"] = y_raw

print("--- Initial Dataset Shape ---")
print(df.shape)

# ==========================================
# 1. CATEGORICAL ENCODING
# ==========================================
print("\n--- Applying Encoders ---")

# LabelEncoder (for target or binary column)
le = LabelEncoder()
df["target_encoded"] = le.fit_transform(df["target"])

# OrdinalEncoder (for ordered categories: High School < Bachelor < Master)
oe = OrdinalEncoder(categories=[["High School", "Bachelor", "Master"]])
df["education_ordinal"] = oe.fit_transform(df[["education_level"]])

# OneHotEncoder (for nominal categories like city)
ohe = OneHotEncoder(sparse_output=False, drop="first")
city_encoded = ohe.fit_transform(df[["city"]])
city_encoded_df = pd.DataFrame(
    city_encoded, columns=ohe.get_feature_names_out(["city"])
)
df = pd.concat([df, city_encoded_df], axis=1)

print("Encoding complete!")

# ==========================================
# 2. FEATURE SCALING & PLOTTING
# ==========================================
print("\n--- Applying Scalers & Generating Plots ---")
numeric_cols = [f"num_feat_{i}" for i in range(5)]

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

# Scale data
scaled_std = std_scaler.fit_transform(df[numeric_cols])
scaled_minmax = minmax_scaler.fit_transform(df[numeric_cols])
scaled_robust = robust_scaler.fit_transform(df[numeric_cols])

# Plot distributions before and after scaling
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(df["num_feat_0"], kde=True, ax=axes[0, 0], color="blue")
axes[0, 0].set_title("Before Scaling")

sns.histplot(scaled_std[:, 0], kde=True, ax=axes[0, 1], color="orange")
axes[0, 1].set_title("StandardScaler (Mean=0, Std=1)")

sns.histplot(scaled_minmax[:, 0], kde=True, ax=axes[1, 0], color="green")
axes[1, 0].set_title("MinMaxScaler (0 to 1)")

sns.histplot(scaled_robust[:, 0], kde=True, ax=axes[1, 1], color="purple")
axes[1, 1].set_title("RobustScaler (Resilient to Outliers)")

plt.tight_layout()
plt.savefig("scaling_comparison.png")
print("Scaling comparison plot saved as 'scaling_comparison.png'!")

# ==========================================
# 3. FEATURE SELECTION (SelectKBest)
# ==========================================
print("\n--- Running SelectKBest ---")
X = df[numeric_cols]
y = df["target_encoded"]

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Top 5 Selected Features:", list(selected_features))
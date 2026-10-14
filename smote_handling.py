from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate an intentionally imbalanced dataset (90% majority, 10% minority)
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    weights=[0.9, 0.1],
    random_state=42
)

print("--- Original Dataset Class Distribution ---")
print(Counter(y))

# 2. Split data into train and test sets (always apply SMOTE only to the training set!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nBefore SMOTE (Train Set):", Counter(y_train))

# 3. Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE (Train Set):", Counter(y_train_resampled))

# 4. Save visualization comparing class distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Majority (0)', 'Minority (1)'], list(Counter(y_train).values()), color=['blue', 'orange'])
axes[0].set_title("Before SMOTE (Imbalanced)")
axes[0].set_ylabel("Count")

axes[1].bar(['Majority (0)', 'Minority (1)'], list(Counter(y_train_resampled).values()), color=['blue', 'orange'])
axes[1].set_title("After SMOTE (Balanced)")

plt.tight_layout()
plt.savefig("smote_comparison.png")
print("\nSMOTE distribution plot saved as 'smote_comparison.png'!")
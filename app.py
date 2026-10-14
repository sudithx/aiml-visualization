# Import the tools we installed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

print("Libraries imported successfully! Loading data...")

# 1. Load a built-in sample dataset from Seaborn
df = sns.load_dataset("tips")

# 2. Create a canvas with 3 side-by-side charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Chart 1: Distribution Plot
sns.histplot(data=df, x="total_bill", hue="time", kde=True, ax=axes[0], palette="Set2")
axes[0].set_title("Total Bill Distribution")

# Chart 2: Box Plot
sns.boxplot(data=df, x="day", y="total_bill", hue="sex", ax=axes[1], palette="pastel")
axes[1].set_title("Bills by Day & Gender")

# Chart 3: Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=axes[2])
axes[2].set_title("Correlation Heatmap")

# 3. Clean up layout so labels don't overlap
plt.tight_layout()

# 4. Save the figure as an image file
plt.savefig("my_charts.png", dpi=300)
print("Charts saved successfully as 'my_charts.png'!")

# 5. Display the charts on screen
plt.show()
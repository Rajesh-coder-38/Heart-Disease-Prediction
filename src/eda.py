
# ==========================================
# HEART DISEASE PREDICTION - EDA
# ==========================================

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

# Visualization style
sns.set_style("whitegrid")

# Create plots folder
os.makedirs("app/plots", exist_ok=True)

# ==========================================
# 1. HEART DISEASE DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="target")

plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("app/plots/heart_disease_distribution.png")
plt.close()


# ==========================================
# 2. AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="age", bins=20, kde=True)

plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("app/plots/age_distribution.png")
plt.close()


# ==========================================
# 3. GENDER-WISE HEART DISEASE
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="sex", hue="target")

plt.title("Heart Disease by Gender")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("app/plots/gender_heart_disease.png")
plt.close()


# ==========================================
# 4. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12, 9))

correlation = df.drop(columns=["num"]).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Heart Disease Dataset Correlation Heatmap")
plt.tight_layout()

plt.savefig("app/plots/correlation_heatmap.png")
plt.close()


# ==========================================
# COMPLETED
# ==========================================

print("\nEDA Completed Successfully!")
print("All graphs saved inside app/plots/")

print("\nGenerated Graphs:")
print("1. Heart Disease Distribution")
print("2. Age Distribution")
print("3. Gender-wise Heart Disease")
print("4. Correlation Heatmap")

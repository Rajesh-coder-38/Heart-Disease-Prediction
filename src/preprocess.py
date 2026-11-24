
import pandas as pd

# Load dataset
file_path = "dataset/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    na_values="?",
    skipinitialspace=True
)

print("Original Dataset Shape:", df.shape)

# Step 1: Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Step 2: Handle missing values
# Use mode (most frequent value)
for column in ["ca", "thal"]:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Step 3: Remove duplicate records
duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)

df = df.drop_duplicates()

# Step 4: Convert target into binary classification
df["target"] = (df["num"] > 0).astype(int)

# Step 5: Final dataset information
print("\nFinal Dataset Shape:", df.shape)

print("\nFinal Dataset Preview:")
print(df.head())

print("\nPreprocessing Completed Successfully!")

# Save cleaned dataset
output_path = "dataset/heart_disease_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nCleaned Dataset Saved Successfully!")
print("File Location:", output_path)


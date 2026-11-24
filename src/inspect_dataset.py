
import pandas as pd

# Dataset file path
file_path = "dataset/processed.cleveland.data"

# Column names
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num"
]

# Load dataset
df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    na_values="?",
    skipinitialspace=True
)

# Display first 5 records
print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

# Dataset shape
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Dataset information
print("\n========== DATASET INFORMATION ==========")
print(df.info())

# Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

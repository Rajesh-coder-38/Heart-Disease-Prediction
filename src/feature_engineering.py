
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

# Separate features and target
X = df.drop(columns=["num", "target"])
y = df["target"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

print("\nFeature Scaling Completed Successfully!")

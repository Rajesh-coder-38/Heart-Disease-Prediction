
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Load Dataset
df = pd.read_csv("dataset/heart_disease_cleaned.csv")

X = df.drop(columns=["num", "target"])
y = df["target"]

# Same Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load Best Model
model = joblib.load("models/best_heart_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Classification Report
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("app/confusion_matrix.png")
plt.show()

# ROC Curve
y_probability = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probability)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr, tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)
plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()
plt.savefig("app/roc_curve.png")
plt.show()

print("\nModel Evaluation Completed Successfully!")
print("ROC AUC:", round(roc_auc, 4))

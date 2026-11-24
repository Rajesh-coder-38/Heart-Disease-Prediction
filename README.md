# ❤️ Heart Disease Prediction and Patient Risk Analysis

A Machine Learning-based project that predicts the possibility of heart disease using patient health information. It includes data preprocessing, exploratory data analysis, model training, evaluation, and an interactive Streamlit dashboard.

## 📌 Project Overview

This project uses the UCI Heart Disease dataset to analyze patient health information and predict whether a patient may have heart disease.

The goal is to demonstrate how Machine Learning can support healthcare data analysis and risk prediction for educational purposes.

## ✨ Features

- Patient health dataset analysis
- Data cleaning and preprocessing
- Missing value handling
- Exploratory Data Analysis (EDA)
- Data visualization
- Heart disease prediction
- Model performance evaluation
- Interactive Streamlit dashboard

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## 🤖 Machine Learning Models

The current implementation includes:

1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier
4. Support Vector Machine (SVM)

The models are evaluated using accuracy, precision, recall, and F1-score. The best-performing model is saved for use in the Streamlit application.

## 📊 Model Performance

Using an 80/20 train-test split, the current evaluation produced the following results:

| Model | Test Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 86.89% | 86.67% |
| Random Forest | 90.16% | 90.00% |
| Decision Tree | 73.77% | 74.19% |
| SVM | 85.25% | 84.75% |

**Best-performing model:** Random Forest Classifier

*Note: These are results from one test split and do not guarantee performance on new or real-world patient data.*

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── app/
│   ├── streamlit_app.py
│   └── plots/
│
├── dataset/
│   ├── processed.cleveland.data
│   └── heart_disease_cleaned.csv
│
├── models/
│   └── best_heart_model.pkl
│
├── notebooks/
│
├── src/
│   ├── inspect_dataset.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── evaluate_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rajesh-coder-38/Heart-Disease-Prediction.git
```

### 2. Open the project folder

```bash
cd Heart-Disease-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app/streamlit_app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📚 Dataset Information

Dataset: Heart Disease

Source: UCI Machine Learning Repository

Creators:
- Andras Janosi
- William Steinbrunn
- Matthias Pfisterer
- Robert Detrano

DOI: https://doi.org/10.24432/C52P4X

License: CC BY 4.0

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not a medical diagnostic tool and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 👨‍💻 Author

**Rajesh Kumar Maji**

GitHub: https://github.com/Rajesh-coder-38

---

⭐ If you find this project useful, consider giving it a star!
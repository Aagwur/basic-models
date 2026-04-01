# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay

# %%
# constants

DATA_PATH="../datasets/Breast_Cancer.csv"
TARGET="Status"
RANDOM_STATE=13
TEST_SIZE=0.2

CATEGORICAL_FEATURES = ["Race", "Marital Status", "T Stage ", "N Stage", "6th Stage",
       "differentiate", "Grade", "A Stage", "Estrogen Status",
       "Progesterone Status"]

NUMERIC_FEATURES = ["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]

# %%
# data overview
data = pd.read_csv(DATA_PATH)

# pd.set_option('display.max_columns', None)
print(data.shape)
display(data.head())
data.info()
data.isna().sum()

# %%
X = data.drop(columns=TARGET)
y = data[TARGET].map({"Dead": 1, "Alive": 0})

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# %%
# data preprocessing pipelines

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, NUMERIC_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
])

# %%
# full model pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

# %%
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print(classification_report(y_test, y_pred))

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

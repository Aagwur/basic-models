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
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)

# %%
DATA_PATH = "../datasets/dermatology_database_1.csv"
TARGET = "class"
RANDOM_STATE = 13
TEST_SIZE = 0.2

# %%
data = pd.read_csv(DATA_PATH)

pd.set_option("display.max_columns", None)
print(data.shape)
display(data.head())
data.info()
data.isna().sum()

# %%
# find value that is string but not numeric and replace it with nan
data["age"] = pd.to_numeric(data["age"], errors="coerce")
data["age"] = data["age"].astype("Int64")

# %%
X = data.drop(columns=TARGET)
y = data[TARGET]

# %%
class_counts = y.value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Class distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# Classes distributed unevenly, so best scoring metric for this dataset will be macro f1-score

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# %%
numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, X.columns),
    ]
)

# %%
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]
)

# %%
# gridsearch, looking for combination of hyperparams with best metric

param_grid = {
    "clf__max_depth": [2, 3, 4, 5, 6, 8, None],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf": [1, 2, 5, 10],
    "clf__criterion": ["gini", "entropy", "log_loss"],
    "clf__ccp_alpha": [0.0, 0.0001, 0.001],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    refit=True,
)

# %%
grid.fit(X_train, y_train)

model = grid.best_estimator_

print("Best params:", grid.best_params_)
print("Best f1_macro score:", grid.best_score_)

# %%
y_pred = model.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("Accuracy:", accuracy)
print("f1_score macro:", f1)
print(classification_report(y_test, y_pred))

# %%
tree_model = model.named_steps["clf"]

plt.figure(figsize=(22, 12))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=[str(cls) for cls in tree_model.classes_],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Decision Tree (first 3 levels)")
plt.show()

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
train_sizes, train_scores, valid_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=cv,
    scoring="f1_macro",
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_scores_mean = train_scores.mean(axis=1)
valid_scores_mean = valid_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores_mean, marker="o", label="Train F1 macro")
plt.plot(train_sizes, valid_scores_mean, marker="o", label="Validation F1 macro")
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("F1 macro")
plt.legend()
plt.grid(True)
plt.show()

# ---
# jupyter:
#   jupytext:
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
import numpy as np

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    learning_curve,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# %%
DATA_PATH = "../datasets/Ames_Housing_Data.csv"
TARGET = "SalePrice"
RANDOM_STATE = 13
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "MS SubClass",
    "Lot Frontage",
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Years since Built",
    "Years since Remod/Add",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Years since Garage Blt",
    "Garage Cars",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Misc Val",
    "Mo Sold",
]
CATEGORICAL_FEATURES = [
    "MS Zoning",
    "Street",
    "Alley",
    "Lot Shape",
    "Land Contour",
    "Utilities",
    "Lot Config",
    "Land Slope",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Type",
    "Exter Qual",
    "Exter Cond",
    "Foundation",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Heating",
    "Heating QC",
    "Central Air",
    "Electrical",
    "Kitchen Qual",
    "Functional",
    "Fireplace Qu",
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Paved Drive",
    "Pool QC",
    "Fence",
    "Misc Feature",
    "Sale Type",
    "Sale Condition",
]

# %%
data = pd.read_csv(DATA_PATH)

pd.set_option("display.max_columns", None)
print(data.shape)
display(data.head())
data.info()
missing_counts = data.isna().sum()
missing_counts[missing_counts > 0]

# %%
data["Years since Built"] = data["Yr Sold"] - data["Year Built"]
data["Years since Remod/Add"] = data["Yr Sold"] - data["Year Remod/Add"]
data["Years since Garage Blt"] = data["Yr Sold"] - data["Garage Yr Blt"]

# %%
X = data.drop(
    columns=[TARGET, "Year Built", "Year Remod/Add", "Garage Yr Blt", "Yr Sold", "PID"]
)
y = data[TARGET]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# %%
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ],
    verbose_feature_names_out=False,
)

# %%
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ]
)

# %%
param_grid = {
    "regressor__max_depth": [4, 6, 8, None],
    "regressor__min_samples_split": [2, 10, 20],
    "regressor__min_samples_leaf": [1, 5, 10],
    "regressor__criterion": ["squared_error", "absolute_error"],
    "regressor__ccp_alpha": [0.0, 0.0001],
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    refit=True,
)

# %%
grid.fit(X_train, y_train)

model = grid.best_estimator_
fitted_preprocessor = model.named_steps["preprocessor"]
fitted_tree = model.named_steps["regressor"]

print("Best params:", grid.best_params_)
print("Best r2 score:", grid.best_score_)

# %%
y_pred = model.predict(X_test)

# %%
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
residuals = y_test - y_pred

print(f"{r2=}")
print(f"{mse=}")
print(f"{mae=}")

# %%
plt.figure(figsize=(25, 10))
plot_tree(
    fitted_tree,
    feature_names=fitted_preprocessor.get_feature_names_out(),
    filled=True,
    fontsize=10,
    max_depth=3,
)
plt.title("DecisionTreeRegressor")
plt.show()

# %%
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="black",
)
plt.xlabel("Actual sale prices")
plt.ylabel("Predicted sale prices")
plt.title(f"Actual vs Predicted (R² = {r2:.3f})")
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted sale prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

# %%
importances = pd.Series(
    fitted_tree.feature_importances_, index=fitted_preprocessor.get_feature_names_out()
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importances.head(10).plot(kind="bar")
plt.ylabel("Importance")
plt.title("Top 10 House Features")
plt.show()

# %%
train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=cv,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Train score")
plt.plot(train_sizes, val_mean, marker="o", label="Test score")
plt.xlabel("Training examples")
plt.ylabel("R2")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()

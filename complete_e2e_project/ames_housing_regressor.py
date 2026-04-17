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
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV,
    learning_curve,
    cross_validate,
)
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline

# %% [markdown]
# Declare constants.
# Define scoring metrics: r2, mean absolute error and root mean squarred error.
# Divide features to categorical and numeric.

# %%
DATA_PATH = "../datasets/Ames_Housing_Data.csv"
TARGET = "SalePrice"
RANDOM_STATE = 13
TEST_SIZE = 0.2

SCORING = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}

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

# %% [markdown]
# Read data.
# Get its shape, columns types and other info.
# Check if there are any n/a values.

# %%
data = pd.read_csv(DATA_PATH)

pd.set_option("display.max_columns", None)
print(data.shape)
display(data.head())
data.info()
missing_counts = data.isna().sum()
missing_counts[missing_counts > 0]

# %% [markdown]
# Add new columns calculated from existing years column to improve readability.
# Declare X, drop PID as not relevant, drop target and years column we transformed in previous step.
# Declare y.
# Convert categorical columns to pandas category dtype (to use hist boosting native category support).
# Split data to 80% train and 20% test.

# %%
data["Years since Built"] = data["Yr Sold"] - data["Year Built"]
data["Years since Remod/Add"] = data["Yr Sold"] - data["Year Remod/Add"]
data["Years since Garage Blt"] = data["Yr Sold"] - data["Garage Yr Blt"]

X = data.drop(
    columns=[TARGET, "Year Built", "Year Remod/Add", "Garage Yr Blt", "Yr Sold", "PID"]
)
y = data[TARGET]

for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# %% [markdown]
# Lets first try to work with parallel trees - RandomForestRegressor.
# It does not need scaling, so for numeric features we use only SimpleImputer, to fill missing values.
# For categorical features we also use SimpleImputer, then OneHotEncoder to get separate column for each category value.
# Declare preprocessor, then model pipeline.

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

random_forest_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=RANDOM_STATE)),
    ]
)

# %% [markdown]
# For random forest there are lots of tuning parameters, so we use RandomizedSearchCV to save time and pick only 20 combinations.

# %%
param_dist = {
    "regressor__n_estimators": [100, 200, 300, 500],
    "regressor__max_depth": [None, 10, 20, 30],
    "regressor__max_features": ["sqrt", 0.3, 0.5, 0.7],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4],
}

search = RandomizedSearchCV(
    estimator=random_forest_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring=SCORING,
    n_jobs=-1,
    random_state=13,
    refit="r2",
    return_train_score=True,
)

# %% [markdown]
# Train models.

# %%
search.fit(X_train, y_train)

best_random_forest_model = search.best_estimator_

print("Best params:", search.best_params_)
print("Best validation mean r2 score:", search.best_score_)

random_forest_results = (
    pd.DataFrame(search.cv_results_).sort_values("mean_test_r2", ascending=False).head()
)

random_forest_results[["mean_test_r2", "mean_test_mae", "mean_test_rmse"]]

# %% [markdown]
# Execution time is 3:32 min. Metrics are not bad for first try, but we need to compare it with other models to evaluate.

# %% [markdown]
# Now let's try consistent trees flow - HistGradientBoostingRegressor.
# Gradient boosting does not need scaling, handles missing feature values and categories encoding, so just straighforward model definition.

# %%
hist_boost_model = HistGradientBoostingRegressor(
    categorical_features="from_dtype",
    random_state=RANDOM_STATE,
)

# %% [markdown]
# Randomized search again with most common parameters for hist boosting.

# %%
param_dist = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [200, 300, 500],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [5, 10, 20],
    "l2_regularization": [0.0, 0.01, 0.1, 1.0],
}

search = RandomizedSearchCV(
    estimator=hist_boost_model,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv,
    scoring=SCORING,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    refit="r2",
    return_train_score=True,
    error_score="raise",
)

# %% [markdown]
# Train models.

# %%
search.fit(X_train, y_train)

best_hist_boost_model = search.best_estimator_

print("Best params:", search.best_params_)
print("Best validation mean r2 score:", search.best_score_)

hist_boost_results = (
    pd.DataFrame(search.cv_results_).sort_values("mean_test_r2", ascending=False).head()
)

hist_boost_results[["mean_test_r2", "mean_test_mae", "mean_test_rmse"]]

# %% [markdown]
# Execution time reduced significantly, to 1:51 min.
# Validation metrics got better.
# The best results so far.

# %% [markdown]
# Finally let's try simple linear regression with l2 regularization.
# Don't forget to add scaler for numeric values and imputer for both numeric and categorical.

# %%
numeric_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
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

ridge_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RidgeCV(alphas=np.logspace(-3, 3, 20))),
    ]
)

# %% [markdown]
# Perform cross validation.

# %%
cv_results = cross_validate(
    estimator=ridge_model,
    X=X_train,
    y=y_train,
    cv=cv,
    scoring=SCORING,
    n_jobs=-1,
    return_train_score=True,
)

ridge_results = pd.DataFrame(cv_results)

print("Validation mean r2 score:", ridge_results.test_r2.mean())
print("Validation mean MAE score:", ridge_results.test_mae.mean())
print("Validation mean RMSE score:", ridge_results.test_rmse.mean())

# %% [markdown]
# metrics are quite a bit worse than such in validation datasets from previous models.

# %% [markdown]
# Let's summarize all metrics across models

# %%
summary = pd.DataFrame(
    [
        {
            "model": "RandomForest",
            "cv_r2": random_forest_results["mean_test_r2"].iloc[0],
            "cv_mae": random_forest_results["mean_test_mae"].iloc[0],
            "cv_rmse": random_forest_results["mean_test_rmse"].iloc[0],
        },
        {
            "model": "HistGradientBoosting",
            "cv_r2": hist_boost_results["mean_test_r2"].iloc[0],
            "cv_mae": hist_boost_results["mean_test_mae"].iloc[0],
            "cv_rmse": hist_boost_results["mean_test_rmse"].iloc[0],
        },
        {
            "model": "RidgeCV",
            "cv_r2": ridge_results.test_r2.mean(),
            "cv_mae": ridge_results.test_mae.mean(),
            "cv_rmse": ridge_results.test_rmse.mean(),
        },
    ]
)

summary.sort_values("cv_r2", ascending=False)


# %% [markdown]
# Hist Gradient boosting has the best metrics, let's investigate it.

# %%
y_pred = best_hist_boost_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
residuals = y_test - y_pred

print("Test r2 score:", r2)
print("Test MAE score:", mae)
print("Test RMSE score:", rmse)

# %% [markdown]
# Test metrics are slightly worse, but not critical. It is still better than other models validation metrics.

# %% [markdown]
# Let's build a graph comparison of predicted and actual sales prices.
# From this graph we can see that prediction errors become more dispersed for more expensive houses.

# %%
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.3, color="red")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="black",
)
plt.grid(True)
plt.title(f"Actual vs Predicted (R2 = {r2:.3f})")
plt.xlabel("Actual sale prices")
plt.ylabel("Predicted sale prices")
plt.show()

# %% [markdown]
# Let's build Residual Plot graph, which shows how big error model makes depending on sales price.
# From this graph we can see - variance residuals increase for higher predicted prices, heteroscedasticity in smarter words.

# %%
plt.figure(figsize=(7, 7))
plt.scatter(y_pred, residuals, alpha=0.3, color="red")
plt.axhline(y=0, linestyle="--", color="black")
plt.grid(True)
plt.title("Residual plot")
plt.xlabel("Predicted sale prices")
plt.ylabel("Residuals")
plt.show()

# %% [markdown]
# Histogram of residuals shows the info from previous graph but in format: error/number of appearances.

# %%
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30)
plt.title("Residual distribution")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# %% [markdown]
# Permutation importances histogram shows quality decrease after shuffling feature values.
# From this histogram the most important features are:  Overall Quality and Ground Livivng Area, that are pretty obvious features.
# Low importance does not always mean that feature is useless, especially if features are correlated.

# %%
perm = permutation_importance(
    best_hist_boost_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="r2",
    n_jobs=-1,
)

raw_importances = pd.Series(
    perm.importances_mean,
    index=X_test.columns,
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
raw_importances.head(10).plot.bar(color="red")
plt.title("Permutation importance (raw features)")
plt.ylabel("Importance")
plt.show()

# %% [markdown]
# Error by price segment shows mean absolute error difference between price groups.
# Higher prices tend to have bigger MAE.

# %%
analysis_df = pd.DataFrame(
    {
        "actual": y_test,
        "pred": y_pred,
    }
)

analysis_df["abs_error"] = (analysis_df["actual"] - analysis_df["pred"]).abs()
analysis_df["price_bin"] = pd.qcut(analysis_df["actual"], q=4, duplicates="drop")

segment_error = analysis_df.groupby("price_bin")["abs_error"].agg(
    ["mean", "median", "count"]
)
display(segment_error)

plt.figure(figsize=(8, 5))
segment_error["mean"].plot.bar(color="red")
plt.title("Mean absolute error by price quartile")
plt.ylabel("MAE")
plt.xlabel("Actual price quartile")
plt.grid(True, axis="y")
plt.show()

# %% [markdown]
# Learning curve shows the dynamic of learning on train data and validation data.
# Model has significant оverfitting, but validation score rises with amount of data, so more data could help.

# %%
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_hist_boost_model,
    X=X_train,
    y=y_train,
    cv=cv,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Train score")
plt.plot(train_sizes, val_mean, marker="o", label="Validation score")

plt.fill_between(
    train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15
)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)

plt.xlabel("Training examples")
plt.ylabel("R2")
plt.title("Learning Curve with std")
plt.legend()
plt.grid(True)
plt.show()

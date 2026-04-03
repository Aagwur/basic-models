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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
DATA_PATH = "../datasets/Messy_Employee_dataset.csv"
TARGET = "Salary"
RANDOM_STATE = 13
TEST_SIZE = 0.2

CATEGORICAL_FEATURES = ["Status", "Department", "Region"]
NUMERIC_FEATURES = ["Age", "Experience", "Remote_Work", "Performance_Score"]

# %%
data = pd.read_csv(DATA_PATH)

print(data.shape)
display(data.head())
data.info()
data.isna().sum()

# %%
# drop rows with missing target
data = data.dropna(subset=[TARGET]).copy()

# split columns to 2 different
data[["Department", "Region"]] = data["Department_Region"].str.split("-", expand=True)

# parse join date
data["Join_Date"] = pd.to_datetime(
    data["Join_Date"], format="%m/%d/%Y", errors="coerce"
)

# fixed reference date for reproducibility
reference_date = data["Join_Date"].max()

# create experience in years
data["Experience"] = ((reference_date - data["Join_Date"]).dt.days / 365.25).round(1)

# # ordinal encoding for performance
data["Performance_Score"] = (
    data["Performance_Score"]
    .str.strip()
    .map(
        {
            "Poor": 1,
            "Average": 2,
            "Good": 3,
            "Excellent": 4,
        }
    )
)

# make remote_work numeric
data["Remote_Work"] = data["Remote_Work"].map(
    {"TRUE": 1, "FALSE": 0, True: 1, False: 0}
)

# drop unused columns
data = data.drop(
    columns=[
        "Employee_ID",
        "First_Name",
        "Last_Name",
        "Email",
        "Phone",
        "Department_Region",
        "Join_Date",
    ]
)

data.head()

# %%
X = data.drop(columns=TARGET)
y = data[TARGET]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

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
    ]
)

# %%
model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])

# %%
model.fit(X_train, y_train)
None

# %%
y_pred = model.predict(X_test)

# %%
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"{mse=}")
print(f"{mae=}")
print(f"{r2=}")


# %% [markdown]
# Metrics are really bad. R2 < 0 means that it predicts even worse than just mean salary

# %%
def plot_manual_regression(y_test, y_pred):
    residuals = y_test - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    ax1.scatter(y_test, y_pred, color="royalblue", alpha=0.5, edgecolors="white")

    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        lw=2,
        label="Ideal prediction",
    )

    ax1.set_title("Comparison: Real vs Predicted", fontsize=12)
    ax1.set_xlabel("Real salary")
    ax1.set_ylabel("Predicted salary")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Residual Plot
    ax2.scatter(y_pred, residuals, color="forestgreen", alpha=0.5, edgecolors="white")
    ax2.axhline(y=0, color="red", linestyle="--", lw=2)

    ax2.set_title("Residuals", fontsize=12)
    ax2.set_xlabel("Predicted Salary")
    ax2.set_ylabel("Error")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


plot_manual_regression(y_test, y_pred)

# %% [markdown]
# The graph here is very revealing.
# The model predicts the same salary for almost all rows, somewhere in the range of 82k–89k.
# The model does not see a useful signal in the features, so it slides to a forecast close to the average target value

# %%
for col in ["Status", "Department", "Region", "Remote_Work", "Performance_Score"]:
    print(f"\n--- {col} ---")
    print(
        data.groupby(col)["Salary"]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean")
    )

# %% [markdown]
# Salary does not differ much for different groups.

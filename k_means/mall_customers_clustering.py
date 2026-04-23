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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# %% [markdown]
# Let's set some constants, default clusters number to 3, features

# %%
DATA_PATH = "../datasets/Mall_Customers.csv"
RANDOM_STATE = 13
N_CLUSTERS = 3

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]

# %%
data = pd.read_csv(DATA_PATH)

display(data.shape)
display(data.head())
data.info()
display(data.isna().sum())

# %% [markdown]
# Converting gender to numeric value

# %%
data["Gender"] = data["Gender"].map({"Female": 0, "Male": 1})

X = data[FEATURES].copy()

# %% [markdown]
# Scaling is important for KMeans

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# Training model

# %%
kmeans = KMeans(
    n_clusters=N_CLUSTERS, init="k-means++", n_init=10, random_state=RANDOM_STATE
)

kmeans.fit(X_scaled)

labels = kmeans.labels_
centers_scaled = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"{inertia=}")
print("silhouette_score =", silhouette_score(X_scaled, labels))

# %% [markdown]
# Let's build graph which shows inertias under different k values.

# %%
k_values = range(1, 11)
inertias = []

for k in k_values:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# %% [markdown]
# We can not tell much from the Elbow method, no visible elbow.
# Let's check silhouete score with different cluster numbers.

# %%
k_values = range(2, 11)
sil_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    cluster_labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    sil_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(k_values, sil_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")
plt.grid(True)
plt.show()

# %% [markdown]
# From the graph above we can see that optimal number of clusters is 4, cause it gives a spike in silhouette score.
# Let's train model with updated k.

# %%
kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=RANDOM_STATE)

kmeans.fit(X_scaled)

labels = kmeans.labels_
centers_scaled = kmeans.cluster_centers_
inertia = kmeans.inertia_

# %% [markdown]
# Cause we have more 4 features, lets use PCA method to reduce them to 2.
# That way we can visualize it on 2D graph.

# %%
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clusters Visualized with PCA")
plt.show()

# %% [markdown]
# Dots form well-separated areas on the graph, we can call clustering successful

# %% [markdown]
# More detailed cluster separation visualization

# %%
sample_silhouette_values = silhouette_samples(X_scaled, labels)

plt.figure(figsize=(8, 6))
y_lower = 10

for cluster_id in range(N_CLUSTERS):
    cluster_silhouette_vals = sample_silhouette_values[labels == cluster_id]
    cluster_silhouette_vals.sort()

    size_cluster = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster

    plt.fill_betweenx(
        np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7
    )

    plt.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))
    y_lower = y_upper + 10

plt.axvline(x=silhouette_score(X_scaled, labels), linestyle="--")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster")
plt.title("Silhouette Plot")
plt.show()

# %% [markdown]
# Table with data and clusters

# %%
results = X.copy()
results["cluster"] = labels

display(results.head())
display(results["cluster"].value_counts().sort_index())

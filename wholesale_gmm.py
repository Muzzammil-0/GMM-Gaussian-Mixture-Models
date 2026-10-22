import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# ---------- 1. Load & Scale ----------
data = fetch_openml('wholesale-customers', version=1, as_frame=True, data_home='./data')
df = data.frame.copy()

y_true = df['Channel'].astype(int).values
spend_cols = ['V2', 'V3', 'V4', 'V5', 'V6', 'V7']
X = df[spend_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 2. Choosing N via AIC / BIC ----------
# BIC penalizes complexity more heavily than AIC (good for avoiding overfitting)
n_components_range = range(1, 8)
aic_scores, bic_scores = [], []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='diag', random_state=42, n_init=5).fit(X_scaled)
    aic_scores.append(gmm.aic(X_scaled))
    bic_scores.append(gmm.bic(X_scaled))

plt.figure(figsize=(8, 5))
plt.plot(n_components_range, aic_scores, marker='o', label='AIC')
plt.plot(n_components_range, bic_scores, marker='s', label='BIC')
plt.xlabel("Number of components")
plt.ylabel("Information Criterion (lower is better)")
plt.title("GMM — Model selection via AIC / BIC")
plt.legend(); plt.grid(alpha=0.3); plt.show()

# ---------- 3. Fit at best n ----------
# Usually, the minimum BIC is a safe choice. Adjust this if the plot says otherwise.
n_best = 2 
gmm = GaussianMixture(n_components=2,covariance_type='diag', random_state=42, n_init=5).fit(X_scaled)
labels = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

print(f"\nGMM with {n_best} components:")
print(f"  Cluster sizes: {np.bincount(labels)}")
print(f"  ARI vs Channel: {adjusted_rand_score(y_true, labels):.4f}")

# ---------- 4. Compare to K-Means directly ----------
km = KMeans(n_clusters=n_best, n_init=10, random_state=42).fit(X_scaled)
print(f"  K-Means ARI for same k: {adjusted_rand_score(y_true, km.labels_):.4f}")

# ---------- 5. The Magic: Soft Assignments ----------
# Find points where the model is highly uncertain (probs near 0.5)
uncertainty = np.min(probs, axis=1)
uncertain_indices = np.argsort(uncertainty)[::-1][:5] # top 5 most uncertain

print("\n--- Most uncertain points (Soft Assignments) ---")
for idx in uncertain_indices:
    print(f"  Point {idx:3d}: Prob Cluster 0 = {probs[idx, 0]:.3f}, Prob Cluster 1 = {probs[idx, 1]:.3f}")

# ---------- 6. Visualize ----------
pca = PCA(n_components=2).fit(X_scaled)
X_2d = pca.transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: GMM hard labels
for c in range(n_best):
    mask = labels == c
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"GMM Cluster {c}", alpha=0.6, s=30)
axes[0].set_title(f"GMM Hard Labels (k={n_best})")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Right: GMM Soft probabilities (color intensity based on confidence)
scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=probs[:, 0], cmap='coolwarm', alpha=0.7, s=30)
plt.colorbar(scatter, ax=axes[1], label='P(Cluster 0)')
axes[1].set_title("GMM Soft Assignments (Probabilities)")
axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
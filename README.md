# Gaussian Mixture Models (GMM) — Probabilistic Clustering

GMM applied to the Wholesale Customers dataset to demonstrate soft clustering, covariance modeling, and probabilistic model selection (AIC/BIC), compared directly against K-Means and DBSCAN.

## Dataset
- `fetch_openml('wholesale-customers', version=1)`
- 440 rows, 6 standardized spend features
- Ground truth (`Channel`) used only for post-hoc Adjusted Rand Index (ARI) evaluation

## Approach
1. Standardize spend features.
2. Sweep `n_components` (1 to 7) and plot AIC/BIC to select the optimal number of clusters.
3. Fit GMM at the BIC-optimal k (k=2) using `covariance_type='diag'` to prevent overfitting on the small 6D dataset.
4. Extract hard labels (`predict`) and soft probabilities (`predict_proba`).
5. Compare ARI against K-Means at the same k.
6. Inspect the most uncertain data points (probabilities nearest to 0.50).
7. Visualize hard labels vs. soft probability gradient via PCA.

## Results
- **Optimal k (BIC):** 2
- **GMM Cluster sizes:** [306, 134] (closely matches ground truth 298/142)
- **GMM ARI vs Channel:** 0.4042
- **K-Means ARI (same k):** 0.2352
- **Most uncertain points:** Probabilities ranging from 0.43 to 0.57, capturing customers whose spending behavior sits exactly on the boundary between Horeca and Retail.

## Key Observations
- **Covariance type matters:** Initializing with `covariance_type='full'` overfits the 6D data (ARI 0.09), finding nonsensical overlapping clusters. Switching to `'diag'` allows the model to capture the differing variance structures of the channels without overfitting, boosting ARI to 0.40.
- **GMM beats K-Means here:** Because Retail and Horeca have different spending variances, K-Means' assumption of spherical, equal-variance clusters limits its ability to separate them (ARI 0.23).
- **Soft clustering is actionable:** GMM doesn't just force points into a bucket. The boundary points (e.g., ~50/50 probability) represent "hybrid" customers who buy like both a restaurant and a retail shop. 
- **The 3-Way Conclusion:** 
  - *K-Means:* Best for simple, spherical partitions.
  - *DBSCAN:* Best for isolating outliers (the 12 bulk buyers).
  - *GMM:* Best for overlapping clusters with different variances, and for quantifying uncertainty.

## Stack
scikit-learn · numpy · pandas · matplotlib
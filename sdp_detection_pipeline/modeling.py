"""Core pipeline step: MCA reduction, clustering, and feature selection."""

from typing import List, Optional

import numpy as np
import pandas as pd
from prince import MCA
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
import fastcluster


class MCAClusterFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Chain MCA, clustering, and Random Forest feature selection.

    Produces a ranked set of alignment columns (Specificity‑Determining Positions).
    All intermediate results are stored as fitted attributes for downstream use.

    Parameters
    ----------
    mca_n_components : int, default 2
        Number of MCA dimensions to retain.
    cluster_method : {'single-linkage', 'k-means'}, default 'single-linkage'
        Clustering algorithm.
    min_clusters, max_clusters : int, defaults 2 and 10
        Search range for the optimal number of clusters (via silhouette score).
    rf_n_estimators : int, default 1000
        Number of trees in the Random Forest.
    random_state : int or None, default 42
        Seed for K‑Means and Random Forest.
    feature_threshold : str or float, default 'median'
        Threshold for `SelectFromModel` ('median', 'mean', or a float).
    importance_cutoff : float, default 0.9
        Cumulative importance fraction for automatic column selection.
    top_n : int or None, default None
        If provided, keep exactly the top‑N most important columns.
    """

    def __init__(
        self,
        mca_n_components: int = 2,
        cluster_method: str = "single-linkage",
        min_clusters: int = 2,
        max_clusters: int = 10,
        rf_n_estimators: int = 1000,
        random_state: Optional[int] = 42,
        feature_threshold: str = "median",
        importance_cutoff: float = 0.9,
        top_n: Optional[int] = None,
    ) -> None:
        self.mca_n_components = mca_n_components
        self.cluster_method = cluster_method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.rf_n_estimators = rf_n_estimators
        self.random_state = random_state
        self.feature_threshold = feature_threshold
        self.importance_cutoff = importance_cutoff
        self.top_n = top_n

    def fit(self, X: pd.DataFrame, y=None) -> "MCAClusterFeatureSelector":
        if self.cluster_method not in ("k-means", "single-linkage"):
            raise ValueError(
                f"cluster_method must be 'k-means' or 'single-linkage', got {self.cluster_method!r}"
            )

        # Preserve the original index so cluster assignments can be mapped back to headers.
        self.input_index_ = X.index
        self.unique_sequences_ = X.reset_index(drop=True)

        # 1. Dimensionality reduction via MCA
        self.mca_ = MCA(n_components=self.mca_n_components)
        self.mca_.fit(self.unique_sequences_)
        self.coordinates_ = np.array(self.mca_.transform(self.unique_sequences_))

        # 2. Clustering
        self.labels_ = self._optimal_clusters(self.coordinates_)

        # 3. One‑Hot Encoding of sequences
        self.encoder_ = OneHotEncoder(sparse_output=True)
        X_enc = self.encoder_.fit_transform(self.unique_sequences_)
        feature_names = self.encoder_.get_feature_names_out(
            self.unique_sequences_.columns.astype(str)
        )

        # 4. Random Forest classification using cluster labels
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state,
        )
        rf.fit(X_enc, self.labels_)

        # 5. Feature selection
        self.selector_ = SelectFromModel(rf, threshold=self.feature_threshold, prefit=True)
        selected_idx = self.selector_.get_support(indices=True)
        self.selected_feature_names_ = feature_names[selected_idx]

        # Aggregate importance per original alignment column
        col_imp_df = pd.DataFrame({
            "feature": self.selected_feature_names_,
            "importance": rf.feature_importances_[selected_idx],
        })
        # Column names are of the form "<col>_<aa>" – extract the column number
        col_imp_df["column"] = col_imp_df["feature"].str.rsplit("_", n=1).str[0].astype(int)
        self.column_importances_ = (
            col_imp_df.groupby("column")["importance"]
            .sum()
            .sort_values(ascending=False)
        )

        # 6. Decide which columns to keep
        cumulative = np.cumsum(self.column_importances_) / self.column_importances_.sum()
        if self.top_n is not None:
            self.selected_columns_: List[int] = self.column_importances_.index[:self.top_n].tolist()
        else:
            cutoff = int(np.searchsorted(cumulative.values, self.importance_cutoff))
            self.selected_columns_ = self.column_importances_.index[: cutoff + 1].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return `X` filtered to the selected alignment columns."""
        return X[sorted(self.selected_columns_)]

    # ------------------------------------------------------------------ internal helpers
    def _optimal_clusters(self, coords: np.ndarray) -> np.ndarray:
        k_range = range(self.min_clusters, self.max_clusters + 1)
        silhouette_scores: List[float] = []
        candidates: List[np.ndarray] = []

        if self.cluster_method == "single-linkage":
            Z = fastcluster.linkage(coords, method="ward")
            for k in k_range:
                labels = fcluster(Z, k, criterion="maxclust")
                silhouette_scores.append(silhouette_score(coords, labels))
                candidates.append(labels)
        else:  # k‑means
            for k in k_range:
                model = KMeans(
                    n_clusters=k, n_init="auto", random_state=self.random_state
                )
                labels = model.fit_predict(coords)
                silhouette_scores.append(silhouette_score(coords, labels))
                candidates.append(labels)

        best_idx = int(np.argmax(silhouette_scores))
        return candidates[best_idx]
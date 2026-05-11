"""
Experiment 2 — Behavioral clustering.

Represent each model as its 1,290-dim NED score vector.
Compute pairwise cosine similarity and apply hierarchical clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity


def get_score_vectors(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return model score vectors — transpose of the page × model matrix.

    Returns shape (n_models, n_pages).
    """
    return matrix.T


def compute_pairwise_cosine(vectors: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise cosine similarity between model score vectors.

    Returns symmetric (n_models, n_models) DataFrame, diagonal = 1.0.
    """
    sim = cosine_similarity(vectors.values)
    return pd.DataFrame(sim, index=vectors.index, columns=vectors.index)


def hierarchical_cluster(similarity_matrix: pd.DataFrame):
    """Run Ward hierarchical clustering on cosine distance (1 - similarity).

    Returns scipy linkage array suitable for dendrogram plotting.
    """
    dist = 1.0 - similarity_matrix.values
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method="ward")


def cluster_labels(similarity_matrix: pd.DataFrame, n_clusters: int) -> pd.Series:
    """Cut the dendrogram at n_clusters and return cluster label per model.

    Returns pd.Series indexed by model name, values are integer cluster labels.
    """
    Z = hierarchical_cluster(similarity_matrix)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return pd.Series(labels, index=similarity_matrix.index, name="cluster")

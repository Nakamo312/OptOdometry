"""
spectral.py
Spectral clustering для corridor-графа.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from graph import RegionGraph


@dataclass
class SpectralResult:
    embedding: np.ndarray
    cluster_labels: np.ndarray
    segment_image: np.ndarray
    eigenvalues: np.ndarray


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.clip(degree, 1e-6, None))
    d_inv = np.diag(inv_sqrt.astype(np.float32))
    identity = np.eye(adjacency.shape[0], dtype=np.float32)
    return identity - d_inv @ adjacency @ d_inv


def _kmeans_numpy(data: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    if len(data) < k:
        return np.arange(len(data), dtype=np.int32)

    rng = np.random.default_rng(7)
    centers = data[rng.choice(len(data), size=k, replace=False)].copy()
    labels = np.zeros(len(data), dtype=np.int32)

    for _ in range(max_iter):
        dists = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = data[labels == cluster_id]
            if len(members) > 0:
                centers[cluster_id] = members.mean(axis=0)
    return labels


def spectral_cluster(
    region_graph: RegionGraph,
    n_clusters: int = 3,
    n_eigenvectors: Optional[int] = None,
) -> SpectralResult:
    adjacency = region_graph.adjacency
    lap = normalized_laplacian(adjacency)
    eigenvalues, eigenvectors = np.linalg.eigh(lap)

    if n_eigenvectors is None:
        n_eigenvectors = max(2, n_clusters)
    start = 1 if len(eigenvectors) > 1 else 0
    end = min(start + n_eigenvectors, eigenvectors.shape[1])
    embedding = eigenvectors[:, start:end].astype(np.float32)

    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.clip(row_norms, 1e-6, None)
    cluster_labels = _kmeans_numpy(embedding, k=n_clusters)

    segment_image = np.zeros_like(region_graph.labels_image, dtype=np.int32)
    for node in region_graph.nodes:
        segment_image[region_graph.labels_image == node.node_id] = cluster_labels[node.node_id]

    return SpectralResult(
        embedding=embedding,
        cluster_labels=cluster_labels,
        segment_image=segment_image,
        eigenvalues=eigenvalues.astype(np.float32),
    )

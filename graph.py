"""
graph.py
Построение corridor-графа над superpixel-like регионами.

Вместо регулярной сетки используем кластеризацию пикселей по цвету и
координате. Это не классический SLIC, но уже даёт регионы, которые лучше
следуют реальным границам, чем клеточная дискретизация.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class RegionNode:
    node_id: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    mean_hsv: np.ndarray
    center_xy: np.ndarray
    area: int


@dataclass
class RegionGraph:
    nodes: List[RegionNode]
    features: np.ndarray
    adjacency: np.ndarray
    labels_image: np.ndarray
    grid_shape: Tuple[int, int]


def _node_feature(mean_hsv: np.ndarray, center_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    h_angle = float(mean_hsv[0]) * (2.0 * np.pi / 180.0)
    h_embed = np.array([np.cos(h_angle), np.sin(h_angle)], dtype=np.float32)
    return np.concatenate(
        [
            h_embed,
            np.array([mean_hsv[1] / 255.0, mean_hsv[2] / 255.0], dtype=np.float32),
            center_xy / np.array([max(width, 1), max(height, 1)], dtype=np.float32),
        ]
    ).astype(np.float32)


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    remap = {int(old): idx for idx, old in enumerate(unique)}
    out = np.empty_like(labels, dtype=np.int32)
    for old, new in remap.items():
        out[labels == old] = new
    return out


def build_region_superpixels(
    hsv_corridor: np.ndarray,
    target_regions: int = 24,
    compactness: float = 0.35,
) -> RegionGraph:
    """
    Построить superpixel-like регионы через k-means в пространстве color+xy.
    """
    h, w = hsv_corridor.shape[:2]
    n_pixels = h * w
    k = int(max(4, min(target_regions, max(4, n_pixels // 32))))

    lab = cv2.cvtColor(hsv_corridor, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xy = np.stack([xx / max(w, 1), yy / max(h, 1)], axis=-1)

    pixel_features = np.concatenate(
        [
            lab.reshape(-1, 3) / np.array([255.0, 255.0, 255.0], dtype=np.float32),
            compactness * xy.reshape(-1, 2),
        ],
        axis=1,
    ).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.2,
    )
    _compactness, labels_flat, _centers = cv2.kmeans(
        pixel_features,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels_flat.reshape(h, w).astype(np.int32)

    # Небольшое сглаживание меток большинством в 3x3 окне.
    smoothed = labels.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = labels[y - 1:y + 2, x - 1:x + 2].reshape(-1)
            vals, counts = np.unique(patch, return_counts=True)
            smoothed[y, x] = int(vals[np.argmax(counts)])
    labels = _compact_labels(smoothed)

    nodes: List[RegionNode] = []
    features: List[np.ndarray] = []
    unique_labels = np.unique(labels)
    for node_id, label in enumerate(unique_labels):
        ys, xs = np.nonzero(labels == label)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        patch = hsv_corridor[labels == label]
        mean_hsv = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
        center_xy = np.array([xs.mean(), ys.mean()], dtype=np.float32)
        area = int(xs.size)

        row = int(round(center_xy[1] / max(h, 1) * 10))
        col = int(round(center_xy[0] / max(w, 1) * 10))
        nodes.append(
            RegionNode(
                node_id=node_id,
                row=row,
                col=col,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                mean_hsv=mean_hsv,
                center_xy=center_xy,
                area=area,
            )
        )
        features.append(_node_feature(mean_hsv, center_xy, w, h))

    labels = _compact_labels(labels)
    return RegionGraph(
        nodes=nodes,
        features=np.asarray(features, dtype=np.float32),
        adjacency=np.zeros((len(nodes), len(nodes)), dtype=np.float32),
        labels_image=labels,
        grid_shape=(len(unique_labels), 1),
    )


def build_similarity_graph(
    region_graph: RegionGraph,
    sigma_color: float = 0.35,
    sigma_space: float = 0.20,
) -> RegionGraph:
    """
    Построить матрицу смежности для соседних регионов.
    """
    nodes = region_graph.nodes
    feats = region_graph.features
    labels = region_graph.labels_image
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=np.float32)

    neighbor_pairs = set()
    h, w = labels.shape
    for y in range(h):
        for x in range(w - 1):
            a = int(labels[y, x])
            b = int(labels[y, x + 1])
            if a != b:
                neighbor_pairs.add(tuple(sorted((a, b))))
    for y in range(h - 1):
        for x in range(w):
            a = int(labels[y, x])
            b = int(labels[y + 1, x])
            if a != b:
                neighbor_pairs.add(tuple(sorted((a, b))))

    for i, j in neighbor_pairs:
        color_dist = float(np.linalg.norm(feats[i, :4] - feats[j, :4]))
        space_dist = float(np.linalg.norm(feats[i, 4:] - feats[j, 4:]))
        weight = np.exp(
            -(color_dist ** 2) / max(2.0 * sigma_color ** 2, 1e-6)
            -(space_dist ** 2) / max(2.0 * sigma_space ** 2, 1e-6)
        )
        adjacency[i, j] = adjacency[j, i] = float(weight)

    region_graph.adjacency = adjacency
    return region_graph

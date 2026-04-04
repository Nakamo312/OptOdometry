"""
graph.py
Построение corridor-графа над регулярными регионами изображения.

Для первого research-прототипа используем регулярную сетку регионов
как замену суперпикселям: это проще, полностью детерминировано и
не требует внешних зависимостей.
"""

from dataclasses import dataclass
from typing import List, Tuple

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


def build_region_grid(
    hsv_corridor: np.ndarray,
    cell_size: int = 16,
) -> RegionGraph:
    """
    Разбить corridor на регулярную сетку регионов.
    """
    h, w = hsv_corridor.shape[:2]
    n_rows = max(1, int(np.ceil(h / cell_size)))
    n_cols = max(1, int(np.ceil(w / cell_size)))

    labels = -np.ones((h, w), dtype=np.int32)
    nodes: List[RegionNode] = []
    features = []
    node_id = 0

    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * cell_size
            y1 = min(h, (row + 1) * cell_size)
            x0 = col * cell_size
            x1 = min(w, (col + 1) * cell_size)
            patch = hsv_corridor[y0:y1, x0:x1]

            mean_hsv = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
            center_xy = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            area = int((y1 - y0) * (x1 - x0))

            labels[y0:y1, x0:x1] = node_id
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

            h_angle = mean_hsv[0] * (2.0 * np.pi / 180.0)
            h_embed = np.array([np.cos(h_angle), np.sin(h_angle)], dtype=np.float32)
            hsv_embed = np.concatenate([
                h_embed,
                np.array([mean_hsv[1] / 255.0, mean_hsv[2] / 255.0], dtype=np.float32),
                center_xy / np.array([max(w, 1), max(h, 1)], dtype=np.float32),
            ])
            features.append(hsv_embed)
            node_id += 1

    return RegionGraph(
        nodes=nodes,
        features=np.asarray(features, dtype=np.float32),
        adjacency=np.zeros((len(nodes), len(nodes)), dtype=np.float32),
        labels_image=labels,
        grid_shape=(n_rows, n_cols),
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
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=np.float32)

    for i, ni in enumerate(nodes):
        for j in range(i + 1, n):
            nj = nodes[j]
            dr = abs(ni.row - nj.row)
            dc = abs(ni.col - nj.col)
            if max(dr, dc) > 1:
                continue

            color_dist = float(np.linalg.norm(feats[i, :4] - feats[j, :4]))
            space_dist = float(np.linalg.norm(feats[i, 4:] - feats[j, 4:]))
            weight = np.exp(
                -(color_dist ** 2) / max(2.0 * sigma_color ** 2, 1e-6)
                -(space_dist ** 2) / max(2.0 * sigma_space ** 2, 1e-6)
            )
            adjacency[i, j] = adjacency[j, i] = float(weight)

    region_graph.adjacency = adjacency
    return region_graph

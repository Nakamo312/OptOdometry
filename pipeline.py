"""
pipeline.py
Новый corridor-based spectral pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from boundaries import SegmentBoundary, extract_segment_boundaries
from corridor import MotionCorridorSpec, extract_motion_corridor
from graph import RegionGraph, build_region_superpixels, build_similarity_graph
from measurement import BoundaryMeasurement, TrackedBoundary
from metrics import PipelineMetrics
from spectral import SpectralResult, spectral_cluster
from tracking import BoundaryTracker, BoundaryTrackingConfig


@dataclass
class PipelineConfig:
    corridor: MotionCorridorSpec = field(default_factory=MotionCorridorSpec)
    tracking: BoundaryTrackingConfig = field(default_factory=BoundaryTrackingConfig)
    target_regions: int = 24
    compactness: float = 0.35
    n_clusters: int = 3
    sigma_color: float = 0.35
    sigma_space: float = 0.20


@dataclass
class PipelineOutput:
    corridor_frame: np.ndarray
    region_graph: RegionGraph
    spectral: SpectralResult
    boundaries: List[SegmentBoundary]
    best_track: Optional[TrackedBoundary] = None
    measurement: Optional[BoundaryMeasurement] = None


class SpectralCorridorPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.metrics = PipelineMetrics()
        self.tracker = BoundaryTracker(self.config.tracking)

    def reset(self):
        self.metrics = PipelineMetrics()
        self.tracker.reset()

    def process_frame(
        self,
        hsv_frame: np.ndarray,
        heading_deg: float = 0.0,
        frame_number: int = 0,
        event_confirmed: bool = False,
    ) -> PipelineOutput:
        corridor_frame, roi = extract_motion_corridor(
            hsv_frame,
            self.config.corridor,
            heading_deg=heading_deg,
        )
        region_graph = build_region_superpixels(
            corridor_frame,
            target_regions=self.config.target_regions,
            compactness=self.config.compactness,
        )
        region_graph = build_similarity_graph(
            region_graph,
            sigma_color=self.config.sigma_color,
            sigma_space=self.config.sigma_space,
        )
        spectral_result = spectral_cluster(
            region_graph,
            n_clusters=self.config.n_clusters,
        )
        boundaries = extract_segment_boundaries(
            spectral_result.segment_image,
            frame_number=frame_number,
            roi_id=roi.corridor_id,
            crossing_y=float(corridor_frame.shape[0] * 0.5),
        )
        best_track = self.tracker.update(boundaries, frame_number=frame_number)
        measurement = self.tracker.build_measurement(
            best_track,
            frame_number=frame_number,
            event_confirmed=event_confirmed,
        )
        self.metrics.tracking = self.tracker.tracking_eval()
        return PipelineOutput(
            corridor_frame=corridor_frame,
            region_graph=region_graph,
            spectral=spectral_result,
            boundaries=boundaries,
            best_track=best_track,
            measurement=measurement,
        )

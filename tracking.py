"""
tracking.py
Temporal tracking для границ corridor-based spectral pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from boundaries import SegmentBoundary
from measurement import (
    BoundaryCrossingGeometry,
    BoundaryMeasurement,
    TrackedBoundary,
    default_measurement_covariance,
)
from metrics import TrackingEval


@dataclass
class BoundaryTrackingConfig:
    max_tracks: int = 8
    max_frame_gap: int = 3
    max_center_distance: float = 24.0
    max_phi_distance: float = 0.7
    min_hits_to_confirm: int = 3
    ema_alpha: float = 0.7


def _angle_distance(a: float, b: float) -> float:
    delta = float(a - b)
    return float(abs(np.arctan2(np.sin(delta), np.cos(delta))))


def _boundary_center(boundary: SegmentBoundary) -> np.ndarray:
    if boundary.points.shape[0] == 0:
        return np.zeros(2, dtype=np.float32)
    return boundary.points.mean(axis=0).astype(np.float32)


class BoundaryTracker:
    def __init__(self, config: Optional[BoundaryTrackingConfig] = None):
        self.config = config or BoundaryTrackingConfig()
        self._tracks: List[TrackedBoundary] = []
        self._next_track_id = 1

    def reset(self):
        self._tracks.clear()
        self._next_track_id = 1

    def get_tracks(self) -> List[TrackedBoundary]:
        return list(self._tracks)

    def update(
        self,
        boundaries: List[SegmentBoundary],
        frame_number: int,
    ) -> Optional[TrackedBoundary]:
        matched_track_ids = set()
        best_track: Optional[TrackedBoundary] = None
        best_score = float("-inf")

        for boundary in boundaries:
            if boundary.candidate is None or boundary.geometry is None:
                continue
            track = self._match_boundary(boundary)
            if track is None:
                track = self._create_track(boundary, frame_number)
            else:
                self._update_track(track, boundary, frame_number)
            matched_track_ids.add(track.track_id)
            if track.mean_score > best_score:
                best_score = track.mean_score
                best_track = track

        self._age_unmatched_tracks(frame_number, matched_track_ids)
        self._prune_tracks()
        return best_track

    def build_measurement(
        self,
        track: Optional[TrackedBoundary],
        frame_number: int,
        event_confirmed: bool = False,
    ) -> Optional[BoundaryMeasurement]:
        if track is None or track.crossing_geometry is None:
            return None
        if not track.is_confirmed(self.config.min_hits_to_confirm):
            return None
        return BoundaryMeasurement(
            frame_number=frame_number,
            geometry=track.crossing_geometry,
            confidence=float(np.clip(track.mean_score / 100.0, 0.0, 1.0)),
            class_old=track.class_old,
            class_new=track.class_new,
            covariance=default_measurement_covariance(),
            source_track_id=track.track_id,
            event_confirmed=event_confirmed,
        )

    def tracking_eval(self) -> TrackingEval:
        lifetimes = [track.age for track in self._tracks]
        hits = [track.hits for track in self._tracks]
        confirmed = sum(
            1 for track in self._tracks
            if track.is_confirmed(self.config.min_hits_to_confirm)
        )
        return TrackingEval(
            mean_track_lifetime=float(np.mean(lifetimes)) if lifetimes else 0.0,
            mean_track_hits=float(np.mean(hits)) if hits else 0.0,
            confirmed_tracks=confirmed,
            total_tracks=len(self._tracks),
        )

    def _match_boundary(self, boundary: SegmentBoundary) -> Optional[TrackedBoundary]:
        assert boundary.geometry is not None
        center = _boundary_center(boundary)
        best_track = None
        best_cost = float("inf")

        for track in self._tracks:
            if track.crossing_geometry is None or not track.history:
                continue
            prev_boundary = track.history[-1]
            prev_center = prev_boundary.curve_points.mean(axis=0).astype(np.float32)
            center_dist = float(np.linalg.norm(center - prev_center))
            phi_dist = _angle_distance(
                boundary.geometry.phi_cross,
                track.crossing_geometry.phi_cross,
            )
            if center_dist > self.config.max_center_distance:
                continue
            if phi_dist > self.config.max_phi_distance:
                continue
            cost = center_dist + 10.0 * phi_dist
            if cost < best_cost:
                best_cost = cost
                best_track = track
        return best_track

    def _create_track(
        self,
        boundary: SegmentBoundary,
        frame_number: int,
    ) -> TrackedBoundary:
        assert boundary.candidate is not None
        track = TrackedBoundary(
            track_id=self._next_track_id,
            model_type=boundary.candidate.model_type,
            state_params=boundary.candidate.model_params.copy(),
            crossing_geometry=boundary.geometry,
            last_frame=frame_number,
            roi_id=boundary.candidate.roi_id,
        )
        track.append_candidate(boundary.candidate)
        self._tracks.append(track)
        self._next_track_id += 1
        return track

    def _update_track(
        self,
        track: TrackedBoundary,
        boundary: SegmentBoundary,
        frame_number: int,
    ):
        assert boundary.candidate is not None
        assert boundary.geometry is not None

        alpha = self.config.ema_alpha
        prev = track.crossing_geometry
        if prev is None:
            smoothed = boundary.geometry
        else:
            smoothed_point = alpha * prev.point + (1.0 - alpha) * boundary.geometry.point
            smoothed_tangent = alpha * prev.tangent + (1.0 - alpha) * boundary.geometry.tangent
            norm = max(float(np.linalg.norm(smoothed_tangent)), 1e-6)
            smoothed_tangent = smoothed_tangent / norm
            smoothed = BoundaryCrossingGeometry(
                u_cross=float(alpha * prev.u_cross + (1.0 - alpha) * boundary.geometry.u_cross),
                phi_cross=float(alpha * prev.phi_cross + (1.0 - alpha) * boundary.geometry.phi_cross),
                curvature=float(alpha * prev.curvature + (1.0 - alpha) * boundary.geometry.curvature),
                tangent=smoothed_tangent.astype(np.float32),
                point=smoothed_point.astype(np.float32),
            )

        track.crossing_geometry = smoothed
        track.model_type = boundary.candidate.model_type
        track.last_frame = frame_number
        track.misses = 0
        track.append_candidate(boundary.candidate)

    def _age_unmatched_tracks(self, frame_number: int, matched_track_ids: set[int]):
        for track in self._tracks:
            if track.track_id in matched_track_ids:
                continue
            if track.last_frame >= 0 and frame_number > track.last_frame:
                track.misses += 1

    def _prune_tracks(self):
        survivors = []
        for track in self._tracks:
            if track.misses > self.config.max_frame_gap:
                continue
            survivors.append(track)
        survivors.sort(key=lambda item: (item.hits, item.mean_score), reverse=True)
        self._tracks = survivors[:self.config.max_tracks]

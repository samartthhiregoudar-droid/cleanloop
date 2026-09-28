"""Keeps a short movement history for each tracked person.

The event engine uses this to answer: "which person was near this spot
in the last few seconds?"
"""
import math
from collections import deque


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TrackHistory:
    def __init__(self, track_id, timestamp):
        self.track_id = track_id
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.last_bbox = None
        self.points = deque()  # (timestamp, (x, y) foot point)

    def add(self, timestamp, point, bbox):
        self.points.append((timestamp, point))
        self.last_seen = timestamp
        self.last_bbox = bbox

    def trim(self, now, keep_seconds):
        while self.points and now - self.points[0][0] > keep_seconds:
            self.points.popleft()

    @property
    def position(self):
        return self.points[-1][1] if self.points else None

    @property
    def name(self):
        return f"Person_{self.track_id}"


class PersonTracker:
    def __init__(self, history_seconds=5.0, forget_seconds=10.0):
        self.history_seconds = history_seconds
        self.forget_seconds = forget_seconds
        self.tracks = {}

    def update(self, detections, now):
        for det in detections:
            if det.track_id is None:
                continue
            track = self.tracks.get(det.track_id)
            if track is None:
                track = self.tracks[det.track_id] = TrackHistory(det.track_id, now)
            track.add(now, det.foot_point, det.bbox)

        for tid in list(self.tracks):
            track = self.tracks[tid]
            track.trim(now, self.history_seconds)
            if now - track.last_seen > self.forget_seconds:
                del self.tracks[tid]

    def visible(self, now, max_gap=0.5):
        """Tracks seen in the last max_gap seconds."""
        return [t for t in self.tracks.values() if now - t.last_seen <= max_gap]

    def nearest_recent(self, point, max_distance):
        """Track whose recent path came closest to point, within max_distance.

        Returns (track, distance) or (None, None).
        """
        candidates = self.find_recent_candidates(point, max_distance, limit=1)
        if candidates:
            return candidates[0][0], candidates[0][1]
        return None, None

    def find_recent_candidates(self, point, max_distance, limit=3):
        """Correlate static object position backward in time with recent ByteTrack trajectories.

        Returns list of tuples: [(track, min_distance, last_seen_time), ...] sorted by closest distance.
        """
        results = []
        for track in self.tracks.values():
            min_d = None
            for ts, p in track.points:
                d = distance(p, point)
                if d <= max_distance and (min_d is None or d < min_d):
                    min_d = d
            if min_d is not None:
                results.append((track, min_d, track.last_seen))

        # Sort primarily by spatial proximity to object, secondarily by recency
        results.sort(key=lambda item: (item[1], -item[2]))
        return results[:limit]

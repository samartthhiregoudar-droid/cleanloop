"""Finds new objects that appear in the scene and stay still.

Uses MOG2 background subtraction with a slow learning rate, removes areas
covered by people, and tracks the remaining blobs over time. When a blob
becomes stable, an edge comparison decides whether something was ADDED
(possible litter) or REMOVED (someone picked something up).
"""
import math

import cv2
import numpy as np


def _center(b):
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _edge_density(gray):
    if gray.size == 0:
        return 0.0
    edges = cv2.Canny(gray, 50, 150)
    return np.count_nonzero(edges) / edges.size


class StaticObject:
    _next_id = 1

    def __init__(self, bbox, now):
        self.id = StaticObject._next_id
        StaticObject._next_id += 1
        self.bbox = bbox
        self.first_seen = now
        self.last_seen = now
        self.stable_count = 1
        self.stable = False
        self.stable_since = None
        self.added = None  # True = placed, False = removed from background

    @property
    def center(self):
        return _center(self.bbox)

    @property
    def name(self):
        return f"Object_{self.id}"


class StaticObjectDetector:
    def __init__(self, min_blob_area_px=300, max_blob_area_px=20000, learning_rate=0.001,
                 stable_frames=15, warmup_seconds=3, match_distance_px=25,
                 missing_seconds=1.5, max_foreground_ratio=0.3, person_padding=0.1,
                 require_person_origin=True, target_zone_roi=None, **_):
        self.min_area = min_blob_area_px
        self.max_area = max_blob_area_px
        self.learning_rate = learning_rate
        self.stable_frames = stable_frames
        self.warmup_seconds = warmup_seconds
        self.match_distance = match_distance_px
        self.missing_seconds = missing_seconds
        self.max_fg_ratio = max_foreground_ratio
        self.person_padding = person_padding
        self.require_person_origin = require_person_origin
        self.target_zone_roi = target_zone_roi or [0.0, 0.0, 1.0, 1.0]
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.reset()

    def reset(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.objects = {}
        self.removed = []  # stable objects that disappeared this frame
        self.start_time = None
        self.mask = None
        self._boost = False

    # ---------- helpers ----------
    def _pad(self, box, w, h):
        x1, y1, x2, y2 = box
        px = int((x2 - x1) * self.person_padding)
        py = int((y2 - y1) * self.person_padding)
        return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))

    def _in_target_zone(self, box, w, h):
        """Check if center of blob is within target_zone_roi [ymin, xmin, ymax, xmax]."""
        cx, cy = _center(box)
        ymin, xmin, ymax, xmax = self.target_zone_roi
        return (xmin * w <= cx <= xmax * w) and (ymin * h <= cy <= ymax * h)

    def _is_added(self, frame, bbox):
        """More edges now than in the background = something was placed there."""
        bg_img = self.bg.getBackgroundImage()
        if bg_img is None:
            return True
        x1, y1, x2, y2 = bbox
        current = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        background = cv2.cvtColor(bg_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return _edge_density(current) >= _edge_density(background)

    @property
    def warming_up(self):
        return self.start_time is None or self._warming

    # ---------- main update ----------
    def update(self, frame, now, person_boxes):
        """Process one frame. Returns the list of stable objects."""
        self.removed = []
        if self.start_time is None:
            self.start_time = now
        self._warming = now - self.start_time < self.warmup_seconds

        fast = self._warming or self._boost
        fg = self.bg.apply(frame, learningRate=0.05 if fast else self.learning_rate)
        self._boost = False
        if self._warming:
            self.mask = fg
            return []

        # Clean the mask: drop shadows (value 127), remove noise, fill holes
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # Sudden big change = lighting change or camera shake: adapt quickly, skip
        if np.count_nonzero(fg) / fg.size > self.max_fg_ratio:
            self._boost = True
            self.mask = fg
            return self.stable_objects()

        # Ignore everything covered by people
        h, w = fg.shape
        people = [self._pad(b, w, h) for b in person_boxes]
        for x1, y1, x2, y2 in people:
            fg[y1:y2, x1:x2] = 0
        self.mask = fg

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            if self.min_area <= cv2.contourArea(c) <= self.max_area:
                x, y, bw, bh = cv2.boundingRect(c)
                bbox = (x, y, x + bw, y + bh)
                if self._in_target_zone(bbox, w, h):
                    blobs.append(bbox)

        self._match(blobs, now, frame, people)
        return self.stable_objects()

    def _match(self, blobs, now, frame, people):
        matched = set()
        for blob in blobs:
            c = _center(blob)
            best, best_d = None, self.match_distance
            for obj in self.objects.values():
                if obj.id in matched:
                    continue
                d = _dist(c, obj.center)
                if d <= best_d:
                    best, best_d = obj, d

            if best is None:
                obj = StaticObject(blob, now)
                self.objects[obj.id] = obj
                matched.add(obj.id)
                continue

            best.bbox = blob
            best.last_seen = now
            best.stable_count += 1
            matched.add(best.id)
            if not best.stable and best.stable_count >= self.stable_frames:
                best.stable = True
                best.stable_since = now
                best.added = self._is_added(frame, blob)

        for oid in list(self.objects):
            if oid in matched:
                continue
            obj = self.objects[oid]
            if any(_intersects(obj.bbox, p) for p in people):
                obj.last_seen = now  # hidden behind a person, keep it
                continue
            if now - obj.last_seen > self.missing_seconds:
                del self.objects[oid]
                if obj.stable:
                    self.removed.append(obj)

    def stable_objects(self):
        return [o for o in self.objects.values() if o.stable]

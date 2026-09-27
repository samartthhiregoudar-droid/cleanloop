"""YOLO wrapper: tracks people and labels object crops.

Two separate YOLO instances are used on purpose: one keeps ByteTrack state
for people, the other labels object crops without disturbing the tracker.
"""
from dataclasses import dataclass

import torch
from ultralytics import YOLO

PERSON_CLASS = 0


@dataclass
class Detection:
    track_id: int | None
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    conf: float
    label: str

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def foot_point(self):
        """Bottom-centre of the box: where the person touches the ground."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)


def resolve_device(device):
    if device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return 0
    return device


class Detector:
    def __init__(self, model_path="yolo11n.pt", min_person_conf=0.5, device="auto"):
        self.model_path = model_path
        self.min_person_conf = min_person_conf
        self.device = resolve_device(device)
        self.tracker_model = YOLO(model_path)
        self._labeler = None  # created on first use

    def track_people(self, frame):
        """Detect and track people. Returns a list of Detection with track IDs."""
        results = self.tracker_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS],
            conf=self.min_person_conf,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        detections = []
        for xyxy, conf, tid in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.id.int().tolist()):
            bbox = tuple(int(v) for v in xyxy)
            detections.append(Detection(tid, bbox, float(conf), "person"))
        return detections

    def label_crop(self, crop, min_conf=0.25):
        """Best non-person COCO label for an object crop, else 'unknown object'."""
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 10:
            return "unknown object", 0.0
        if self._labeler is None:
            self._labeler = YOLO(self.model_path)

        results = self._labeler(crop, conf=min_conf, device=self.device, verbose=False)
        boxes = results[0].boxes
        best = None
        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            if int(cls) == PERSON_CLASS:
                continue
            if best is None or conf > best[1]:
                best = (self._labeler.names[int(cls)], float(conf))
        return best or ("unknown object", 0.0)

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
    bbox_extended: tuple = None  # Extended box including held/carried objects

    def __post_init__(self):
        if self.bbox_extended is None:
            self.bbox_extended = self.bbox

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
        """Detect and track people along with held objects as body extensions."""
        results = self.tracker_model.track(
            frame,
            persist=True,
            conf=self.min_person_conf,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None:
            return []

        # Extract people and all non-person object candidate boxes in the frame
        people_boxes = []
        object_boxes = []

        cls_list = boxes.cls.tolist() if boxes.cls is not None else []
        conf_list = boxes.conf.tolist() if boxes.conf is not None else []
        xyxy_list = boxes.xyxy.tolist() if boxes.xyxy is not None else []
        ids_list = boxes.id.int().tolist() if boxes.id is not None else [None] * len(xyxy_list)

        for xyxy, conf, cls, tid in zip(xyxy_list, conf_list, cls_list, ids_list):
            bbox = tuple(int(v) for v in xyxy)
            if int(cls) == PERSON_CLASS:
                people_boxes.append((tid, bbox, float(conf)))
            else:
                object_boxes.append(bbox)

        detections = []
        for tid, p_box, conf in people_boxes:
            px1, py1, px2, py2 = p_box
            ex1, ey1, ex2, ey2 = px1, py1, px2, py2

            # Merge any overlapping/touching object boxes into person's body extension
            for ox1, oy1, ox2, oy2 in object_boxes:
                # Check overlap or proximity
                if not (ox2 < px1 - 30 or ox1 > px2 + 30 or oy2 < py1 - 30 or oy1 > py2 + 30):
                    ex1, ey1 = min(ex1, ox1), min(ey1, oy1)
                    ex2, ey2 = max(ex2, ox2), max(ey2, oy2)

            ext_bbox = (ex1, ey1, ex2, ey2)
            detections.append(Detection(tid, p_box, conf, "person", bbox_extended=ext_bbox))

        return detections

    def label_crop(self, crop, min_conf=0.25):
        """Best non-person COCO label for an object crop, else 'unknown object'."""
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 10:
            return "unknown object", 0.0

        # Resize small crops up to a minimum of 224x224 px for much higher YOLO detection accuracy
        h, w = crop.shape[:2]
        if max(h, w) < 224:
            scale = 224.0 / max(h, w)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        if self._labeler is None:
            self._labeler = YOLO(self.model_path)

        results = self._labeler(crop, conf=min_conf, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return "unknown object", 0.0

        best = None
        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            if int(cls) == PERSON_CLASS:
                continue
            if best is None or conf > best[1]:
                best = (self._labeler.names[int(cls)], float(conf))
        return best or ("unknown object", 0.0)

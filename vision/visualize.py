"""Drawing helpers for the debug window and evidence snapshots."""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
PERSON_COLOR = (255, 160, 0)  # blue-ish (BGR)
STATUS_COLOR = (0, 255, 0)


def draw_people(frame, detections, tracker=None):
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_COLOR, 2)
        label = f"Person_{det.track_id} {det.conf:.2f}"
        cv2.putText(frame, label, (x1, max(15, y1 - 8)), FONT, 0.5, PERSON_COLOR, 2)

        if tracker and det.track_id in tracker.tracks:
            pts = [p for _, p in tracker.tracks[det.track_id].points]
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, a, b, PERSON_COLOR, 2)


def draw_status(frame, text):
    cv2.putText(frame, text, (10, 25), FONT, 0.6, STATUS_COLOR, 2)

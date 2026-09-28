"""Drawing helpers for the debug window and evidence snapshots."""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
PERSON_COLOR = (255, 160, 0)   # blue
STATUS_COLOR = (0, 255, 0)     # green
CANDIDATE_COLOR = (0, 255, 255)  # yellow: new, not yet still
ADDED_COLOR = (0, 140, 255)    # orange: new object, still
REMOVED_COLOR = (160, 160, 160)  # gray: something was picked up


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


def draw_objects(frame, objects):
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        if not obj.stable:
            color, text, thick = CANDIDATE_COLOR, obj.name, 1
        elif obj.added:
            color, text, thick = ADDED_COLOR, f"{obj.name} NEW", 2
        else:
            color, text, thick = REMOVED_COLOR, f"{obj.name} REMOVED", 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, text, (x1, max(15, y1 - 6)), FONT, 0.45, color, 2)


def draw_status(frame, text):
    cv2.putText(frame, text, (10, 25), FONT, 0.6, STATUS_COLOR, 2)


CASE_COLORS = {
    "CANDIDATE": (0, 255, 255),   # yellow
    "PERSON_LEFT": (0, 165, 255),  # orange
    "NUDGED": (0, 0, 255),        # red
}


def draw_cases(frame, cases, tracker=None, now=None):
    for case in cases:
        x1, y1, x2, y2 = case.bbox
        color = CASE_COLORS.get(case.state, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        text = "LITTERING" if case.state == "NUDGED" else case.state
        if case.state == "PERSON_LEFT" and now is not None:
            text += f" {now - case.left_since:.1f}s"
        cv2.putText(frame, f"{text} ({case.person_name})", (x1, y2 + 18), FONT, 0.5, color, 2)

        if tracker and case.person_id in tracker.tracks:
            pos = tracker.tracks[case.person_id].position
            if pos:
                cv2.line(frame, case.center, pos, color, 1)


def draw_banner(frame, text, color=(0, 0, 255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), color, -1)
    cv2.putText(frame, text, (10, h - 12), FONT, 0.6, (255, 255, 255), 2)


def draw_roi_zone(frame, roi):
    if not roi or roi == [0.0, 0.0, 1.0, 1.0]:
        return
    h, w = frame.shape[:2]
    ymin, xmin, ymax, xmax = roi
    x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
    cv2.putText(frame, "MONITORED SET ZONE", (x1 + 10, y1 + 20), FONT, 0.45, (255, 255, 0), 1)

"""Test static object detection together with person tracking.

Usage:
    python test_objects.py                    # webcam
    python test_objects.py eval/videos/a.mp4  # video file
Keys: Q = quit, R = reset background, M = show/hide the mask window
"""
import sys
import time

import cv2

from vision.capture import VideoSource
from vision.config import load_config
from vision.detector import Detector
from vision.static_objects import StaticObjectDetector
from vision.tracker import PersonTracker
from vision.visualize import draw_objects, draw_people, draw_status

config = load_config()
cam = config["camera"]
det_cfg = config["detection"]
source = sys.argv[1] if len(sys.argv) > 1 else cam["source"]

video = VideoSource(source, cam["frame_width"], cam["reconnect_seconds"]).start()
detector = Detector(det_cfg["model"], det_cfg["min_person_conf"], det_cfg["device"])
tracker = PersonTracker(config["event_engine"]["track_history_seconds"])
static = StaticObjectDetector(**config["static_objects"])

show_mask = False
frame_count, fps, t0 = 0, 0.0, time.time()

while True:
    frame, ts = video.read()
    if frame is None:
        if video.finished:
            print("End of video.")
            break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        continue

    people = detector.track_people(frame)
    tracker.update(people, ts)
    static.update(frame, ts, [p.bbox for p in people])

    for obj in static.removed:
        print(f"[{ts:.1f}s] {obj.name} disappeared (was {'NEW' if obj.added else 'REMOVED'})")

    frame_count += 1
    if time.time() - t0 >= 1.0:
        fps = frame_count / (time.time() - t0)
        frame_count, t0 = 0, time.time()

    display = frame.copy()
    draw_people(display, people, tracker)
    draw_objects(display, static.objects.values())
    state = "LEARNING BACKGROUND..." if static.warming_up else f"objects {len(static.stable_objects())}"
    draw_status(display, f"FPS {fps:.1f} | people {len(people)} | {state}")
    cv2.imshow("CleanLoop - object test", display)

    if show_mask and static.mask is not None:
        cv2.imshow("mask", static.mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        static.reset()
        print("Background reset")
    if key == ord("m"):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow("mask")

video.stop()
cv2.destroyAllWindows()

"""Test person detection + tracking.

Usage:
    python test_tracking.py                    # webcam
    python test_tracking.py eval/videos/a.mp4  # video file
Press Q to quit.
"""
import sys
import time

import cv2

from vision.capture import VideoSource
from vision.config import load_config
from vision.detector import Detector
from vision.tracker import PersonTracker
from vision.visualize import draw_people, draw_status

config = load_config()
cam = config["camera"]
det_cfg = config["detection"]
source = sys.argv[1] if len(sys.argv) > 1 else cam["source"]

video = VideoSource(source, cam["frame_width"], cam["reconnect_seconds"]).start()
detector = Detector(det_cfg["model"], det_cfg["min_person_conf"], det_cfg["device"])
tracker = PersonTracker(config["event_engine"]["track_history_seconds"])
print(f"Running on device: {detector.device}")

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

    frame_count += 1
    if time.time() - t0 >= 1.0:
        fps = frame_count / (time.time() - t0)
        frame_count, t0 = 0, time.time()

    display = frame.copy()
    draw_people(display, people, tracker)
    draw_status(display, f"FPS {fps:.1f} | people {len(people)} | tracks {len(tracker.tracks)}")
    cv2.imshow("CleanLoop - tracking test", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.stop()
cv2.destroyAllWindows()

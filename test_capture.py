"""Quick test for the capture module.

Usage:
    python test_capture.py                    # webcam from config.yaml
    python test_capture.py eval/videos/a.mp4  # a video file
Press Q to quit.
"""
import sys
import time

import cv2

from vision.capture import FrameBuffer, VideoSource
from vision.config import load_config

config = load_config()
cam = config["camera"]
source = sys.argv[1] if len(sys.argv) > 1 else cam["source"]

video = VideoSource(source, cam["frame_width"], cam["reconnect_seconds"]).start()
buffer = FrameBuffer(config["evidence"]["clip_seconds_before"] + 1)

frame_count, fps, t0 = 0, 0.0, time.time()
delay = int(1000 / video.fps) if video.is_file else 1  # play files at normal speed

while True:
    frame, ts = video.read()
    if frame is None:
        if video.finished:
            print("End of video.")
            break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        continue

    buffer.add(frame, ts)

    frame_count += 1
    if time.time() - t0 >= 1.0:
        fps = frame_count / (time.time() - t0)
        frame_count, t0 = 0, time.time()

    status = "LIVE" if video.connected else "RECONNECTING"
    display = frame.copy()
    cv2.putText(display, f"{status} | FPS {fps:.1f} | buffer {len(buffer)} frames",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("CleanLoop - capture test", display)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

video.stop()
cv2.destroyAllWindows()

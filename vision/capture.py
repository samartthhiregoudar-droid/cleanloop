"""Video input for CleanLoop: webcam, video file, or RTSP stream.

- Live sources (webcam/RTSP) are read on a background thread so we always
  process the newest frame, and they reconnect automatically if lost.
- Video files are read frame by frame, and timestamps come from the video's
  own clock so tests give the same result every run.
"""
import platform
import threading
import time
from collections import deque

import cv2


def parse_source(source):
    """Turn '0' into 0 (webcam index); leave paths and URLs as strings."""
    if isinstance(source, int):
        return source
    text = str(source).strip()
    return int(text) if text.isdigit() else text


class VideoSource:
    def __init__(self, source, frame_width=640, reconnect_seconds=5):
        self.source = parse_source(source)
        self.is_file = isinstance(self.source, str) and not self.source.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        self.frame_width = frame_width
        self.reconnect_seconds = reconnect_seconds

        self.cap = None
        self.connected = False
        self.finished = False  # only used for video files
        self.fps = 30.0
        self.frame_index = 0

        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    # ---------- opening / closing ----------
    def _open(self):
        if isinstance(self.source, int) and platform.system() == "Windows":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # faster on Windows
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            cap.release()
            self.connected = False
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 < fps < 120 else 30.0
        self.cap = cap
        self.connected = True
        return True

    def start(self):
        if not self._open():
            if self.is_file:
                raise FileNotFoundError(f"Could not open video file: {self.source}")
            print(f"[capture] Could not open {self.source}, will keep retrying...")

        if not self.is_file:
            self._running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    # ---------- reading ----------
    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w == self.frame_width:
            return frame
        scale = self.frame_width / w
        return cv2.resize(frame, (self.frame_width, int(h * scale)))

    def _reader_loop(self):
        """Background thread for live sources: keep only the newest frame."""
        while self._running:
            if not self.connected:
                time.sleep(self.reconnect_seconds)
                print(f"[capture] Reconnecting to {self.source}...")
                if self._open():
                    print("[capture] Reconnected.")
                continue

            ok, frame = self.cap.read()
            if not ok:
                print("[capture] Stream lost.")
                self.connected = False
                self.cap.release()
                continue

            with self._lock:
                self._latest = (self._resize(frame), time.time())

    def read(self):
        """Return (frame, timestamp_seconds), or (None, None) if no new frame.

        For video files, (None, None) with self.finished == True means the end.
        """
        if self.is_file:
            ok, frame = self.cap.read()
            if not ok:
                self.finished = True
                return None, None
            timestamp = self.frame_index / self.fps
            self.frame_index += 1
            return self._resize(frame), timestamp

        with self._lock:
            latest, self._latest = self._latest, None
        return latest if latest else (None, None)


class FrameBuffer:
    """Keeps the last N seconds of frames, used for 'before the event' clips."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.frames = deque()

    def add(self, frame, timestamp):
        self.frames.append((timestamp, frame))
        while self.frames and timestamp - self.frames[0][0] > self.seconds:
            self.frames.popleft()

    def get_since(self, start_timestamp):
        return [(t, f) for t, f in self.frames if t >= start_timestamp]

    def __len__(self):
        return len(self.frames)

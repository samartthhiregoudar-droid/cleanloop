"""CleanLoop Core Pipeline: Wires capture, detection, tracking, event state machine, nudge, and evidence recording."""
import time

from vision.capture import FrameBuffer, VideoSource
from vision.config import load_config
from vision.detector import Detector
from vision.evidence import EvidenceManager
from vision.event_engine import EventEngine
from vision.nudge import AudioNudge
from vision.static_objects import StaticObjectDetector
from vision.tracker import PersonTracker
from vision.visualize import draw_cases, draw_objects, draw_people, draw_status


class CleanLoopPipeline:
    def __init__(self, config=None, event_callback=None):
        self.config = config or load_config()
        self.event_callback = event_callback

        cam_cfg = self.config["camera"]
        det_cfg = self.config["detection"]
        stat_cfg = self.config["static_objects"]
        eng_cfg = self.config["event_engine"]
        evid_cfg = self.config["evidence"]

        # Components
        self.video = VideoSource(cam_cfg["source"], cam_cfg["frame_width"], cam_cfg["reconnect_seconds"])
        self.buffer = FrameBuffer(evid_cfg["clip_seconds_before"] + evid_cfg["clip_seconds_after"] + 1)
        self.detector = Detector(det_cfg["model"], det_cfg["min_person_conf"], det_cfg["device"])
        self.tracker = PersonTracker(eng_cfg["track_history_seconds"])
        self.static_detector = StaticObjectDetector(**stat_cfg)
        self.engine = EventEngine(**eng_cfg)
        self.nudge = AudioNudge(enabled=True)
        self.evidence = EvidenceManager(
            clip_before_sec=evid_cfg["clip_seconds_before"],
            clip_after_sec=evid_cfg["clip_seconds_after"],
            retention_days=evid_cfg["retention_days"]
        )

        self.last_cleanup = 0.0

    def start(self):
        self.video.start()
        return self

    def stop(self):
        self.video.stop()
        self.nudge.stop()

    def process_frame(self):
        """Read and process one frame. Returns (annotated_frame, raw_frame, events, timestamp)."""
        frame, ts = self.video.read()
        if frame is None:
            return None, None, [], None

        # Add to clip ring buffer
        self.buffer.add(frame, ts)

        # 1. Track people
        people = self.detector.track_people(frame)
        self.tracker.update(people, ts)

        # 2. Detect static objects
        person_boxes = [p.bbox for p in people]
        objects = self.static_detector.update(frame, ts, person_boxes)

        # 3. Process events state machine
        events = self.engine.update(ts, self.static_detector.objects, self.static_detector.removed, self.tracker)

        # 4. Handle events (audio nudge, evidence snapshots/clips, callbacks)
        annotated = frame.copy()
        draw_people(annotated, people, self.tracker)
        draw_objects(annotated, self.static_detector.objects.values())
        draw_cases(annotated, self.engine.cases.values(), self.tracker, ts)

        for event in events:
            self._handle_event(event, annotated, frame, ts)

        # Periodic retention cleanup (once every hour)
        if time.time() - self.last_cleanup > 3600:
            self.evidence.cleanup_old_evidence()
            self.last_cleanup = time.time()

        return annotated, frame, events, ts

    def _handle_event(self, event, annotated_frame, raw_frame, ts):
        kind = event.kind
        case = event.case

        if kind == "CONFIRMED":
            # Label object crop using YOLO labeler
            crop = raw_frame[case.bbox[1]:case.bbox[3], case.bbox[0]:case.bbox[2]]
            label, conf = self.detector.label_crop(crop)
            case.label = label

            # Save snapshot and JSON incident
            record = self.evidence.create_incident_record(case, annotated_frame, self.tracker, label=label)

            # Play audio nudge reminder
            self.nudge.play_nudge()

            if self.event_callback:
                self.event_callback("CONFIRMED", record)

        elif kind == "REDEEMED":
            self.nudge.play_thankyou()
            if case.incident_id:
                self.evidence.update_incident_record(case.incident_id, {"status": "REDEEMED", "state": "REDEEMED"})

            if self.event_callback:
                self.event_callback("REDEEMED", {"incident_id": case.incident_id, "object_id": case.object_id})

        elif kind == "REVIEW_READY":
            # Save 10s video clip (5s before + 5s after)
            if case.incident_id:
                recent_frames = self.buffer.get_since(ts - self.evidence.clip_before_sec)
                clip_path = self.evidence.save_clip(case.incident_id, recent_frames, fps=self.video.fps)
                self.evidence.update_incident_record(case.incident_id, {"status": "PENDING_REVIEW", "state": "REVIEW_READY"})

            if self.event_callback:
                self.event_callback("REVIEW_READY", {"incident_id": case.incident_id, "object_id": case.object_id})

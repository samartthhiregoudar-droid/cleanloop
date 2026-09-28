"""Evidence management for CleanLoop: snapshots, clips, JSON logs, and retention cleanup."""
import json
import os
import time
from pathlib import Path

import cv2

from vision.config import ROOT


class EvidenceManager:
    def __init__(self, storage_dir=None, clip_before_sec=5, clip_after_sec=5, retention_days=7):
        self.base_dir = Path(storage_dir) if storage_dir else ROOT / "storage"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.clips_dir = self.base_dir / "clips"
        self.incidents_dir = self.base_dir / "incidents"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.incidents_dir.mkdir(parents=True, exist_ok=True)

        self.clip_before_sec = clip_before_sec
        self.clip_after_sec = clip_after_sec
        self.retention_days = retention_days

        # Active clip recording sessions: case_id -> recording state
        self._active_clips = {}

    def save_snapshot(self, incident_id, frame):
        """Save annotated frame snapshot."""
        filename = f"{incident_id}.jpg"
        filepath = self.snapshots_dir / filename
        cv2.imwrite(str(filepath), frame)
        return f"storage/snapshots/{filename}"

    def create_incident_record(self, case, frame, tracker=None, label="unknown object"):
        """Save snapshot and initial JSON incident record when CONFIRMED/NUDGED."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        incident_id = f"INC_{timestamp_str}_OBJ{case.object_id}"
        case.incident_id = incident_id

        snapshot_rel_path = self.save_snapshot(incident_id, frame)

        record = {
            "incident_id": incident_id,
            "created_at": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "object_id": case.object_id,
            "person_id": case.person_id,
            "person_name": case.person_name,
            "status": "PENDING_REVIEW",
            "state": case.state,
            "confidence": getattr(case, "confidence", 0.8),
            "label": label,
            "bbox": case.bbox,
            "snapshot_path": snapshot_rel_path,
            "clip_path": None,
            "explanation": f"{case.person_name} dropped an object ({label}) and moved away without retrieving it."
        }

        json_path = self.incidents_dir / f"{incident_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return record

    def update_incident_record(self, incident_id, updates):
        """Update existing JSON incident record (e.g. status REDEEMED, clip_path)."""
        json_path = self.incidents_dir / f"{incident_id}.json"
        if not json_path.exists():
            return None

        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        record.update(updates)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return record

    def save_clip(self, incident_id, frames_with_ts, fps=30.0):
        """Save sequence of (timestamp, frame) tuples to an MP4 video clip."""
        if not frames_with_ts:
            return None

        filename = f"{incident_id}.mp4"
        filepath = self.clips_dir / filename

        first_frame = frames_with_ts[0][1]
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))

        for _, f in frames_with_ts:
            writer.write(f)
        writer.release()

        rel_path = f"storage/clips/{filename}"
        self.update_incident_record(incident_id, {"clip_path": rel_path})
        return rel_path

    def cleanup_old_evidence(self):
        """Delete evidence older than retention_days."""
        cutoff = time.time() - (self.retention_days * 86400)
        deleted_count = 0

        for folder in (self.snapshots_dir, self.clips_dir, self.incidents_dir):
            for item in folder.glob("*"):
                if item.is_file() and item.stat().st_mtime < cutoff:
                    try:
                        item.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"[evidence] Cleanup error deleting {item}: {e}")

        return deleted_count

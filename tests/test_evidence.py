"""Unit tests for evidence manager and retention cleanup."""
import json
import time
from pathlib import Path

import numpy as np

from vision.evidence import EvidenceManager


class FakeCase:
    def __init__(self):
        self.object_id = 99
        self.person_id = 1
        self.person_name = "Person_1"
        self.bbox = (100, 100, 200, 200)
        self.state = "CONFIRMED"
        self.confidence = 0.85
        self.incident_id = None


def test_evidence_creation_and_retention(tmp_path):
    manager = EvidenceManager(storage_dir=tmp_path, clip_before_sec=1, clip_after_sec=1, retention_days=7)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    case = FakeCase()

    record = manager.create_incident_record(case, dummy_frame, label="coffee cup")

    assert record["incident_id"].startswith("INC_")
    assert Path(tmp_path / "snapshots" / f"{record['incident_id']}.jpg").exists()
    assert Path(tmp_path / "incidents" / f"{record['incident_id']}.json").exists()

    # Test clip creation
    frames = [(time.time(), dummy_frame) for _ in range(5)]
    clip_rel = manager.save_clip(record["incident_id"], frames, fps=10.0)
    assert clip_rel is not None
    assert Path(tmp_path / "clips" / f"{record['incident_id']}.mp4").exists()

    # Test retention cleanup
    deleted = manager.cleanup_old_evidence()
    assert deleted == 0  # Recently created, should not be deleted

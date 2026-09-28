"""Unit tests for audio nudge module."""
from vision.nudge import AudioNudge


def test_audio_nudge_initialization():
    nudge = AudioNudge(enabled=False)
    assert nudge.enabled is False
    nudge.play_nudge()
    nudge.play_thankyou()
    nudge.stop()


def test_audio_nudge_worker():
    nudge = AudioNudge(enabled=True)
    nudge.play_nudge("Test message")
    nudge.stop()

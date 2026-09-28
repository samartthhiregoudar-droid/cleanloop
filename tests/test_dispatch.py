"""Unit tests for BBMP garbage black spot escalation dispatcher."""
from vision.dispatch import BBMPEscalator
from vision.static_objects import StaticObject


def test_bbmp_escalator_threshold():
    config = {
        "bbmp_escalation": {
            "enabled": True,
            "pile_area_threshold_px": 5000,
            "cooldown_seconds": 60,
            "phone_number": "+91-80-22660000"
        }
    }
    escalator = BBMPEscalator(config)

    # 1. Below threshold object
    small_obj = StaticObject((0, 0, 50, 50), 0)  # area = 2500 px
    small_obj.stable = True
    alert = escalator.check_and_dispatch([small_obj], now=100)
    assert alert is None

    # 2. Above threshold object
    large_obj = StaticObject((0, 0, 100, 100), 0)  # area = 10000 px
    large_obj.stable = True
    alert = escalator.check_and_dispatch([large_obj], now=100)
    assert alert is not None
    assert alert["event"] == "BBMP_ESCALATED"
    assert alert["phone_number"] == "+91-80-22660000"
    assert alert["total_area_px"] == 10000

    # 3. Cooldown check
    alert_cooldown = escalator.check_and_dispatch([large_obj], now=110)
    assert alert_cooldown is None  # Blocked by 60s cooldown

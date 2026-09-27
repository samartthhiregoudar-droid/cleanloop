"""Unit tests for the littering event engine, using simulated people/objects."""
from vision.detector import Detection
from vision.event_engine import EventEngine
from vision.static_objects import StaticObject
from vision.tracker import PersonTracker

SPOT = (100, 290, 120, 310)   # object box, centre (110, 300)
NEAR = (100, 300)             # person standing next to the object
FAR = (400, 300)              # person far away


def person(tid, foot):
    x, y = foot
    return Detection(tid, (x - 20, y - 100, x + 20, y), 0.9, "person")


def make_obj(bbox=SPOT, added=True):
    obj = StaticObject(bbox, 0)
    obj.stable, obj.added = True, added
    return obj


def setup():
    return EventEngine(), PersonTracker(history_seconds=5)


def step(engine, tracker, t, people, objects, removed=()):
    tracker.update(people, t)
    return [e.kind for e in engine.update(t, objects, list(removed), tracker)]


def run_until_confirmed(engine, tracker, obj):
    objs = {obj.id: obj}
    kinds = step(engine, tracker, 0, [person(1, NEAR)], objs)
    assert kinds == ["CANDIDATE"]
    for t in range(1, 7):
        kinds += step(engine, tracker, t, [person(1, FAR)], objs)
    return kinds


def test_true_littering_is_confirmed():
    engine, tracker = setup()
    kinds = run_until_confirmed(engine, tracker, make_obj())
    assert "CONFIRMED" in kinds


def test_put_down_and_pick_up_is_not_littering():
    engine, tracker = setup()
    obj = make_obj()
    objs = {obj.id: obj}
    kinds = step(engine, tracker, 0, [person(1, NEAR)], objs)
    kinds += step(engine, tracker, 1, [person(1, FAR)], objs)
    kinds += step(engine, tracker, 3, [person(1, NEAR)], objs)   # comes back
    kinds += step(engine, tracker, 4, [person(1, NEAR)], {}, removed=[obj])  # picks up
    assert "CONFIRMED" not in kinds
    assert kinds[-1] == "DISCARDED"


def test_redemption_after_nudge():
    engine, tracker = setup()
    obj = make_obj()
    run_until_confirmed(engine, tracker, obj)
    step(engine, tracker, 8, [person(1, NEAR)], {obj.id: obj})   # returns
    kinds = step(engine, tracker, 9, [], {}, removed=[obj])      # object gone
    assert kinds == ["REDEEMED"]


def test_object_gone_without_person_is_not_redemption():
    engine, tracker = setup()
    obj = make_obj()
    run_until_confirmed(engine, tracker, obj)
    kinds = step(engine, tracker, 8, [person(1, FAR)], {}, removed=[obj])
    assert kinds == ["OBJECT_GONE"]


def test_review_ready_after_redeem_window():
    engine, tracker = setup()
    obj = make_obj()
    run_until_confirmed(engine, tracker, obj)   # confirmed at t=6
    kinds = []
    for t in range(7, 37):
        kinds += step(engine, tracker, t, [person(1, FAR)], {obj.id: obj})
    assert kinds == ["REVIEW_READY"]


def test_object_without_person_is_ignored():
    engine, tracker = setup()
    obj = make_obj()
    kinds = step(engine, tracker, 0, [], {obj.id: obj})
    assert kinds == []


def test_removed_object_is_ignored():
    engine, tracker = setup()
    obj = make_obj(added=False)
    kinds = step(engine, tracker, 0, [person(1, NEAR)], {obj.id: obj})
    assert kinds == []


def test_cooldown_blocks_duplicate_alert_at_same_spot():
    engine, tracker = setup()
    first = make_obj()
    run_until_confirmed(engine, tracker, first)
    step(engine, tracker, 7, [person(1, FAR)], {}, removed=[first])
    second = make_obj()
    kinds = step(engine, tracker, 10, [person(2, NEAR)], {second.id: second})
    assert kinds == []


def test_two_people_link_to_the_closest():
    engine, tracker = setup()
    obj = make_obj()
    tracker.update([person(1, NEAR), person(2, (170, 300))], 0)
    events = engine.update(0, {obj.id: obj}, [], tracker)
    assert events[0].case.person_name == "Person_1"

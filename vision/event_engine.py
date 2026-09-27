"""Littering event engine.

Each new ADDED static object becomes a LitterCase with its own state machine:

    CANDIDATE -> PERSON_LEFT -> NUDGED (confirmed) -> REDEEMED / REVIEW_READY
         ^            |
         +------------+  (person came back before static_seconds)

The engine has no OpenCV or YOLO code, so it can be unit tested with fake data.
"""
import math
from dataclasses import dataclass

CANDIDATE = "CANDIDATE"
PERSON_LEFT = "PERSON_LEFT"
NUDGED = "NUDGED"


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_to_box(p, box):
    """Distance from a point to a box (0 if the point is inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - p[0], 0, p[0] - x2)
    dy = max(y1 - p[1], 0, p[1] - y2)
    return math.hypot(dx, dy)


@dataclass
class LitterCase:
    object_id: int
    bbox: tuple
    person_id: int | None
    person_name: str
    link_distance: float
    created_at: float
    state: str = CANDIDATE
    left_since: float | None = None
    confirmed_at: float | None = None
    last_person_near: float | None = None
    confidence: float = 0.0
    label: str = "unknown object"
    incident_id: object = None

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class EngineEvent:
    kind: str  # CANDIDATE, DISCARDED, CONFIRMED, REDEEMED, OBJECT_GONE, REVIEW_READY
    case: LitterCase
    timestamp: float


class EventEngine:
    def __init__(self, link_distance_px=80, leave_distance_px=150, static_seconds=5,
                 redeem_seconds=30, cooldown_seconds=60, cooldown_radius_px=50,
                 redeem_proximity_seconds=5, require_person=True, **_):
        self.link_distance = link_distance_px
        self.leave_distance = leave_distance_px
        self.static_seconds = static_seconds
        self.redeem_seconds = redeem_seconds
        self.cooldown_seconds = cooldown_seconds
        self.cooldown_radius = cooldown_radius_px
        self.redeem_proximity = redeem_proximity_seconds
        self.require_person = require_person

        self.cases = {}          # object_id -> LitterCase
        self.skip_ids = set()    # objects already evaluated
        self.recent_confirms = []  # (center, timestamp) for cooldown

    # ---------- helpers ----------
    def _someone_near(self, case, visible_tracks):
        return any(
            t.last_bbox and _point_to_box(case.center, t.last_bbox) <= self.link_distance / 2
            for t in visible_tracks
        )

    def _linked_person_left(self, case, tracker, now):
        if case.person_id is None:
            return True
        track = tracker.tracks.get(case.person_id)
        if track is None or now - track.last_seen > 0.5 or track.position is None:
            return True  # out of frame
        return _dist(track.position, case.center) > self.leave_distance

    def _in_cooldown(self, center):
        return any(_dist(c, center) <= self.cooldown_radius for c, _ in self.recent_confirms)

    def _confidence(self, case):
        """Explainable score: closer link + a known person = higher confidence."""
        link_score = 1 - min(case.link_distance, self.link_distance) / self.link_distance
        person_score = 1.0 if case.person_id is not None else 0.0
        return round(0.5 + 0.3 * link_score + 0.2 * person_score, 2)

    # ---------- main update ----------
    def update(self, now, objects, removed, tracker):
        """objects: dict id -> StaticObject; removed: objects that vanished this frame."""
        events = []
        removed_ids = {o.id for o in removed}
        visible = tracker.visible(now)
        self.recent_confirms = [(c, t) for c, t in self.recent_confirms if now - t < self.cooldown_seconds]
        self.skip_ids &= set(objects)

        # 1. Open a case for each new ADDED object linked to a person
        for obj in objects.values():
            if not obj.stable or not obj.added:
                continue
            if obj.id in self.cases or obj.id in self.skip_ids:
                continue
            self.skip_ids.add(obj.id)
            if self._in_cooldown(obj.center):
                continue

            track, d = tracker.nearest_recent(obj.center, self.link_distance)
            if track is None:
                if self.require_person:
                    continue
                pid, pname, d = None, "Unknown", float(self.link_distance)
            else:
                pid, pname = track.track_id, track.name

            case = LitterCase(obj.id, obj.bbox, pid, pname, d, now)
            self.cases[obj.id] = case
            events.append(EngineEvent("CANDIDATE", case, now))

        # 2. Advance every open case
        for oid, case in list(self.cases.items()):
            present = oid in objects and oid not in removed_ids
            if present:
                case.bbox = objects[oid].bbox
            near = self._someone_near(case, visible)
            if near:
                case.last_person_near = now

            if case.state in (CANDIDATE, PERSON_LEFT):
                if not present:
                    events.append(EngineEvent("DISCARDED", case, now))
                    del self.cases[oid]
                    continue
                if case.state == CANDIDATE:
                    if not near and self._linked_person_left(case, tracker, now):
                        case.state = PERSON_LEFT
                        case.left_since = now
                elif near:
                    case.state = CANDIDATE
                    case.left_since = None
                elif now - case.left_since >= self.static_seconds:
                    case.state = NUDGED
                    case.confirmed_at = now
                    case.confidence = self._confidence(case)
                    self.recent_confirms.append((case.center, now))
                    events.append(EngineEvent("CONFIRMED", case, now))

            elif case.state == NUDGED:
                if not present:
                    came_back = (
                        case.last_person_near is not None
                        and case.last_person_near >= case.confirmed_at
                        and now - case.last_person_near <= self.redeem_proximity
                    )
                    events.append(EngineEvent("REDEEMED" if came_back else "OBJECT_GONE", case, now))
                    del self.cases[oid]
                    continue
                if now - case.confirmed_at >= self.redeem_seconds:
                    events.append(EngineEvent("REVIEW_READY", case, now))
                    del self.cases[oid]

        return events

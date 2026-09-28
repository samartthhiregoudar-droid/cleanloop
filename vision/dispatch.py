"""BMRCL Garbage Pile Escalation & Automated Call / SMS Dispatcher.

Monitors accumulated garbage pile sizes. If total static object area exceeds the BMRCL threshold,
initiates an automated call / SMS alert to the BMRCL Sanitation & Maintenance department via Twilio API.
"""
import os
import time


class BMRCLEscalator:
    def __init__(self, config=None):
        cfg = config.get("bmrcl_escalation", {}) if config else {}
        self.enabled = cfg.get("enabled", True)
        self.pile_area_threshold = cfg.get("pile_area_threshold_px", 12000)
        self.cooldown_seconds = cfg.get("cooldown_seconds", 300)
        self.phone_number = cfg.get("phone_number", "+91-80-22969200")
        self.department_name = cfg.get("department_name", "BMRCL Maintenance & Sanitation Ops")
        self.alert_message = cfg.get(
            "alert_message",
            "Urgently required: Large garbage accumulation detected at BMRCL station camera zone."
        )

        twilio_cfg = cfg.get("twilio", {})
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID", twilio_cfg.get("account_sid", ""))
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN", twilio_cfg.get("auth_token", ""))
        self.from_number = os.environ.get("TWILIO_FROM_NUMBER", twilio_cfg.get("from_number", ""))

        self.last_call_time = 0.0

    def calculate_total_pile_area(self, static_objects):
        """Calculate cumulative area in pixels of all stable static objects in frame."""
        total_area = 0
        for obj in static_objects:
            if getattr(obj, "stable", False):
                x1, y1, x2, y2 = obj.bbox
                total_area += (x2 - x1) * (y2 - y1)
        return total_area

    def check_and_dispatch(self, static_objects, now=None):
        """Evaluate static objects and dispatch BMRCL call/SMS if area threshold is exceeded."""
        if not self.enabled:
            return None

        now = now or time.time()
        total_area = self.calculate_total_pile_area(static_objects)

        if total_area < self.pile_area_threshold:
            return None

        # Cooldown check
        if now - self.last_call_time < self.cooldown_seconds:
            return None

        self.last_call_time = now
        dispatch_result = self.make_bmrcl_call(total_area)
        return dispatch_result

    def make_bmrcl_call(self, total_area):
        """Initiate call/SMS via Twilio API or simulate dispatch in test mode."""
        dispatch_payload = {
            "event": "BMRCL_ESCALATED",
            "department": self.department_name,
            "phone_number": self.phone_number,
            "total_area_px": total_area,
            "threshold_px": self.pile_area_threshold,
            "message": self.alert_message,
            "timestamp": time.time(),
            "status": "DISPATCHED"
        }

        # Try Twilio call if credentials provided
        if self.account_sid and self.auth_token and self.from_number:
            try:
                from twilio.rest import Client
                client = Client(self.account_sid, self.auth_token)
                call = client.calls.create(
                    twiml=f'<Response><Say>{self.alert_message}</Say></Response>',
                    to=self.phone_number,
                    from_=self.from_number
                )
                dispatch_payload["twilio_sid"] = call.sid
                print(f"[BMRCL AUTOMATED CALL DISPATCHED] -> Call SID: {call.sid} to {self.phone_number}")
            except Exception as err:
                print(f"[BMRCL DISPATCH ERROR] Failed to send Twilio call: {err}")
                dispatch_payload["status"] = f"DISPATCH_ERROR: {err}"
        else:
            print(f"\n=======================================================")
            print(f"[🚨 BMRCL AUTOMATED CALL DISPATCH SIMULATION]")
            print(f"DIALING: {self.phone_number} ({self.department_name})")
            print(f"MESSAGE: \"{self.alert_message}\"")
            print(f"METRIC: Pile area {total_area} px exceeded threshold {self.pile_area_threshold} px")
            print(f"=======================================================\n")

        return dispatch_payload

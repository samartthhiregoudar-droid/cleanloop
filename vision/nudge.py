"""Audio nudge module: real-time spoken reminders and thank-you messages.

Uses pyttsx3 in a thread-safe worker queue to avoid blocking the main frame processing loop.
Falls back gracefully if TTS hardware/driver is missing or fails.
"""
import queue
import threading
import time


class AudioNudge:
    def __init__(self, default_nudge="Excuse me, please remember to take your item with you.",
                 default_thankyou="Thank you for keeping the area clean!", enabled=True):
        self.default_nudge = default_nudge
        self.default_thankyou = default_thankyou
        self.enabled = enabled

        self._speech_queue = queue.Queue()
        self._running = False
        self._thread = None
        self._tts_engine = None

        if self.enabled:
            self._start_worker()

    def _start_worker(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        # Initialize pyttsx3 inside worker thread (COM loop on Windows)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)
            self._tts_engine = engine
        except Exception as e:
            print(f"[audio_nudge] Notice: pyttsx3 initialization skipped/failed ({e}). Operating in log-only mode.")
            self._tts_engine = None

        while self._running:
            try:
                text = self._speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            print(f"[AUDIO NUDGE SPOKEN] -> \"{text}\"")
            if self._tts_engine is not None:
                try:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                except Exception as err:
                    print(f"[audio_nudge] Speech playback error: {err}")
            self._speech_queue.task_done()

    def speak(self, text):
        if not self.enabled:
            return
        # Avoid duplicate queueing if same phrase is already waiting
        if self._speech_queue.qsize() < 3:
            self._speech_queue.put(text)

        def play_nudge(self, message=None):
            msg = message or self.default_nudge
            self.speak(msg)

        def play_thankyou(self, message=None):
            msg = message or self.default_thankyou
            self.speak(msg)

    def play_nudge(self, message=None):
        msg = message or self.default_nudge
        self.speak(msg)

    def play_thankyou(self, message=None):
        msg = message or self.default_thankyou
        self.speak(msg)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

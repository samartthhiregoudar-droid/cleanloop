"""CleanLoop Main Vision Entry Point.

Usage:
    python run_vision.py                        # webcam (source in config.yaml)
    python run_vision.py eval/videos/demo.mp4   # a recorded video file
    python run_vision.py rtsp://user:pass@IP:554/stream  # an IP / CCTV camera

Keys:
    Q = Quit
    R = Reset background model
    M = Show / hide binary motion mask
"""
import sys
import time

import cv2

from vision.config import load_config
from vision.pipeline import CleanLoopPipeline
from vision.visualize import draw_status


def on_incident_event(kind, data):
    print(f"\n[EVENT ENGINE ALERT] -> {kind}: {data}")


def main():
    config = load_config()

    # Override source if command-line argument is passed
    if len(sys.argv) > 1:
        config["camera"]["source"] = sys.argv[1]

    print(f"Starting CleanLoop Vision Pipeline...")
    print(f"Camera source: {config['camera']['source']}")
    print(f"Detection model: {config['detection']['model']}")
    print("Press Q to quit, R to reset background model, M to toggle mask window.\n")

    pipeline = CleanLoopPipeline(config, event_callback=on_incident_event)
    pipeline.start()

    show_mask = False
    frame_count, fps, t0 = 0, 0.0, time.time()

    try:
        while True:
            annotated, raw, events, ts = pipeline.process_frame()

            if annotated is None:
                if pipeline.video.finished:
                    print("Reached end of video stream/file.")
                    break
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                continue

            frame_count += 1
            if time.time() - t0 >= 1.0:
                fps = frame_count / (time.time() - t0)
                frame_count, t0 = 0, time.time()

            # Status overlay
            state_str = "LEARNING BACKGROUND..." if pipeline.static_detector.warming_up else f"Objects: {len(pipeline.static_detector.stable_objects())}"
            draw_status(annotated, f"FPS: {fps:.1f} | People: {len(pipeline.tracker.visible(ts))} | {state_str}")

            cv2.imshow("CleanLoop - Vision Pipeline", annotated)

            if show_mask and pipeline.static_detector.mask is not None:
                cv2.imshow("CleanLoop - Motion Mask", pipeline.static_detector.mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break
            elif key == ord("r"):
                pipeline.static_detector.reset()
                print("[INFO] Background model reset.")
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("CleanLoop - Motion Mask")

    except KeyboardInterrupt:
        print("\nStopping vision pipeline...")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

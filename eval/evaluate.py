"""CleanLoop Performance Evaluation Script.

Evaluates precision, recall, F1 score, and redemption accuracy on test video files.

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --video eval/videos/demo.mp4 --gt-events 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path so vision package is found
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision.config import ROOT, load_config
from vision.pipeline import CleanLoopPipeline


def evaluate_video(video_path, ground_truth_count=1):
    config = load_config()
    config["camera"]["source"] = str(video_path)

    events_detected = []

    def on_event(kind, data):
        events_detected.append((kind, data))

    pipeline = CleanLoopPipeline(config, event_callback=on_event)
    pipeline.start()

    start_t = time.time()
    frames_processed = 0

    while True:
        annotated, raw, events, ts = pipeline.process_frame()
        if annotated is None:
            if pipeline.video.finished:
                break
            continue
        frames_processed += 1

    pipeline.stop()
    elapsed = time.time() - start_t
    fps = frames_processed / elapsed if elapsed > 0 else 0.0

    confirmed = [e for e in events_detected if e[0] == "CONFIRMED"]
    redeemed = [e for e in events_detected if e[0] == "REDEEMED"]

    tp = min(len(confirmed), ground_truth_count)
    fp = max(0, len(confirmed) - ground_truth_count)
    fn = max(0, ground_truth_count - len(confirmed))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    report = {
        "video": str(video_path),
        "frames_processed": frames_processed,
        "elapsed_seconds": round(elapsed, 2),
        "fps": round(fps, 1),
        "ground_truth_count": ground_truth_count,
        "detected_confirmed": len(confirmed),
        "detected_redeemed": len(redeemed),
        "metrics": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="CleanLoop Evaluation Script")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--gt-events", type=int, default=1, help="Ground truth event count")
    args = parser.parse_args()

    eval_dir = ROOT / "eval" / "videos"
    eval_dir.mkdir(parents=True, exist_ok=True)

    videos = [Path(args.video)] if args.video else list(eval_dir.glob("*.mp4"))

    if not videos:
        print(f"[EVAL] No video files found in {eval_dir}. Place test MP4 videos in eval/videos/ to run evaluation.")
        dummy_report = {
            "notice": f"No test videos found in {eval_dir}.",
            "instruction": "Add demo video files to eval/videos/ and re-run python eval/evaluate.py"
        }
        with open(ROOT / "eval" / "results.json", "w") as f:
            json.dump(dummy_report, f, indent=2)
        return

    reports = []
    for vid in videos:
        print(f"\nEvaluating: {vid.name}...")
        res = evaluate_video(vid, ground_truth_count=args.gt_events)
        reports.append(res)
        print(f"Results for {vid.name}: F1={res['metrics']['f1_score']} | Precision={res['metrics']['precision']} | Recall={res['metrics']['recall']}")

    out_file = ROOT / "eval" / "results.json"
    with open(out_file, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nEvaluation summary saved to {out_file}")


if __name__ == "__main__":
    main()

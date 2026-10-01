"""Optional GPU/Pod validation; not needed on the user's Windows PC."""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano.worker import NanoWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text-file", required=True)
    parser.add_argument("--text-file", default="samples/thai-700.txt")
    parser.add_argument("--runs", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--output-dir", default="benchmark-results")
    args = parser.parse_args()
    inp = {"text": Path(args.text_file).read_text(encoding="utf-8").strip(),
           "reference_text": Path(args.reference_text_file).read_text(encoding="utf-8").strip(),
           "reference_audio_b64": base64.b64encode(Path(args.reference).read_bytes()).decode("ascii"),
           "cfg_value": 2, "seed": 42}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    worker = NanoWorker()
    startup = time.perf_counter() - started
    try:
        for i in range(args.runs):
            result = worker.run(inp)
            output.joinpath(f"run-{i+1}.wav").write_bytes(base64.b64decode(result.pop("audio_b64")))
            result["process_startup_seconds"] = round(startup, 3)
            result["process_total_first_run_seconds"] = round(startup + result["handler_seconds"], 3) if i == 0 else None
            result["includes_runpod_provisioning"] = False
            output.joinpath(f"run-{i+1}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False))
    finally:
        worker.close()


if __name__ == "__main__":
    main()

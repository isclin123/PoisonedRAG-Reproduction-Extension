import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DATASETS = ["nq", "hotpotqa", "msmarco"]
QUERY_RESULTS_DIR = "paper_style_qwen35_2026-07-07"
MODEL_NAME = "ollama_qwen3.5"
REPEAT_TIMES = 10
M = 10
TOP_K = 5
ADV_PER_QUERY = 5


def result_name(dataset):
    return f"{dataset}-contriever-{MODEL_NAME}-top{TOP_K}-adv{ADV_PER_QUERY}-M{M}x{REPEAT_TIMES}"


def read_json_len(path):
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def parse_metrics(log_path):
    metrics = {
        "asr_mean": "",
        "precision_mean": "",
        "recall_mean": "",
        "f1_mean": "",
    }
    if not log_path.exists():
        return metrics
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "asr_mean": r"ASR Mean:\s*([0-9.]+)",
        "precision_mean": r"Precision mean:\s*([0-9.]+)",
        "recall_mean": r"Recall mean:\s*([0-9.]+)",
        "f1_mean": r"F1 mean:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = match.group(1)
    return metrics


def write_summary(rows, summary_path):
    fieldnames = [
        "dataset",
        "status",
        "returncode",
        "result_json",
        "result_iters",
        "log_path",
        "asr_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "started_at",
        "finished_at",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parents[1]
    output_root = (
        project_root
        / "02_reproduction_code_data"
        / "baseline_reproduction_outputs"
        / QUERY_RESULTS_DIR
    )
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "summary.csv"
    rows = []

    for dataset in DATASETS:
        name = result_name(dataset)
        result_json = repo_root / "results" / "query_results" / QUERY_RESULTS_DIR / f"{name}.json"
        log_path = log_dir / f"{name}.log"
        existing_iters = read_json_len(result_json)
        started_at = datetime.now().isoformat(timespec="seconds")

        if existing_iters >= REPEAT_TIMES:
            metrics = parse_metrics(log_path)
            row = {
                "dataset": dataset,
                "status": "skipped_complete",
                "returncode": 0,
                "result_json": str(result_json),
                "result_iters": existing_iters,
                "log_path": str(log_path),
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                **metrics,
            }
            rows.append(row)
            write_summary(rows, summary_path)
            print(f"[SKIP] {dataset}: existing complete result at {result_json}")
            continue

        cmd = [
            sys.executable,
            "main.py",
            "--eval_dataset",
            dataset,
            "--eval_model_code",
            "contriever",
            "--model_name",
            MODEL_NAME,
            "--top_k",
            str(TOP_K),
            "--attack_method",
            "LM_targeted",
            "--adv_per_query",
            str(ADV_PER_QUERY),
            "--score_function",
            "dot",
            "--repeat_times",
            str(REPEAT_TIMES),
            "--M",
            str(M),
            "--query_results_dir",
            QUERY_RESULTS_DIR,
            "--name",
            name,
        ]

        print(f"[RUN] {dataset}: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        with log_path.open("w", encoding="utf-8-sig") as log:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        finished_at = datetime.now().isoformat(timespec="seconds")
        result_iters = read_json_len(result_json)
        metrics = parse_metrics(log_path)
        status = "complete" if proc.returncode == 0 and result_iters >= REPEAT_TIMES else "failed_or_incomplete"
        row = {
            "dataset": dataset,
            "status": status,
            "returncode": proc.returncode,
            "result_json": str(result_json),
            "result_iters": result_iters,
            "log_path": str(log_path),
            "started_at": started_at,
            "finished_at": finished_at,
            **metrics,
        }
        rows.append(row)
        write_summary(rows, summary_path)
        print(f"[DONE] {dataset}: status={status}, returncode={proc.returncode}, iters={result_iters}")

        if proc.returncode != 0:
            print(f"[STOP] See log: {log_path}")
            return proc.returncode

    print(f"[SUMMARY] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

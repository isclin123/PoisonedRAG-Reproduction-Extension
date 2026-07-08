import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DATASETS = ["nq", "hotpotqa", "msmarco"]
MODELS = {
    "qwen35": {
        "label": "qwen3.5:9b",
        "config": "model_configs/ollama_qwen3.5_corpus_ultrafast_config.json",
        "max_context_chars": 80,
        "speed_profile": "ultrafast",
    },
    "llama31_8b": {
        "label": "llama3.1:8b",
        "config": "model_configs/ollama_llama3.1_8b_corpus_ultrafast_config.json",
        "max_context_chars": 80,
        "speed_profile": "ultrafast",
    },
    "mistral_7b": {
        "label": "mistral:7b",
        "config": "model_configs/ollama_mistral_7b_corpus_fast_config.json",
        "max_context_chars": 500,
        "speed_profile": "fast",
    },
}
RATES = ["0.000001", "0.000005", "0.00001", "0.00005"]
VARIANTS = ["original", "query_first_instruction_aware"]
SAMPLE_COUNT = 50
TOP_K = 5
MAX_CONTEXT_CHARS = 500


def repo_root():
    return Path(__file__).resolve().parents[1]


def project_root():
    return repo_root().parents[1]


def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def output_dir(dataset, model_code):
    root = project_root() / "03_low_rate_experiments" / "local_ollama_experiment_runs"
    existing = {
        ("nq", "mistral_7b"): "corpus_level_ollama_mistral_7b_fast_nq50_2026-07-07",
    }
    speed_profile = MODELS[model_code].get("speed_profile", "fast")
    name = existing.get(
        (dataset, model_code),
        f"corpus_level_ollama_{model_code}_{speed_profile}_{dataset}50_fullmatrix_2026-07-07",
    )
    return root / name


def completed_summary(summary_path):
    if not summary_path.exists():
        return False
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    expected_rows = 1 + len(RATES) * len(VARIANTS)
    if len(rows) < expected_rows:
        return False
    return all(str(row.get("n", "")) == str(SAMPLE_COUNT) for row in rows)


def status_path():
    return project_root() / "03_low_rate_experiments" / "local_ollama_experiment_runs" / "corpus_fullmatrix_runner_status_2026-07-07.csv"


def write_status(rows):
    path = status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "dataset",
        "model_code",
        "model_label",
        "status",
        "returncode",
        "output_dir",
        "summary_path",
        "started_at",
        "finished_at",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_command(cmd, log_path, cwd):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8-sig") as log:
        log.write(f"\n\n===== {timestamp()} RUN =====\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def main():
    rows = []
    root = repo_root()
    print(f"[START] corpus full matrix runner at {timestamp()}", flush=True)
    print(f"[STATUS] {status_path()}", flush=True)

    for dataset in DATASETS:
        for model_code, model_info in MODELS.items():
            out = output_dir(dataset, model_code)
            summary = out / "summary.csv"
            log = out / "corpus_fullmatrix_runner.log"
            started_at = timestamp()

            if completed_summary(summary):
                rows.append(
                    {
                        "stage": "corpus_level",
                        "dataset": dataset,
                        "model_code": model_code,
                        "model_label": model_info["label"],
                        "status": "skipped_complete",
                        "returncode": 0,
                        "output_dir": str(out),
                        "summary_path": str(summary),
                        "started_at": started_at,
                        "finished_at": timestamp(),
                    }
                )
                write_status(rows)
                print(f"[SKIP corpus] {dataset}/{model_code}: complete", flush=True)
                continue

            cmd = [
                sys.executable,
                "local_corpus_level_poison_rate_experiment.py",
                "--dataset",
                dataset,
                "--llm_config",
                model_info["config"],
                "--sample_count",
                str(SAMPLE_COUNT),
                "--top_k",
                str(TOP_K),
                "--rates",
                *RATES,
                "--variants",
                *VARIANTS,
                "--max_context_chars",
                str(model_info.get("max_context_chars", MAX_CONTEXT_CHARS)),
                "--output_dir",
                str(out),
                "--resume",
            ]
            print(f"[RUN corpus] {dataset}/{model_code}", flush=True)
            returncode = run_command(cmd, log, root)
            status = "complete" if returncode == 0 and completed_summary(summary) else "failed_or_incomplete"
            rows.append(
                {
                    "stage": "corpus_level",
                    "dataset": dataset,
                    "model_code": model_code,
                    "model_label": model_info["label"],
                    "status": status,
                    "returncode": returncode,
                    "output_dir": str(out),
                    "summary_path": str(summary),
                    "started_at": started_at,
                    "finished_at": timestamp(),
                }
            )
            write_status(rows)
            print(f"[DONE corpus] {dataset}/{model_code}: {status}", flush=True)

    print(f"[DONE] corpus full matrix runner at {timestamp()}", flush=True)
    print(f"[STATUS] {status_path()}", flush=True)


if __name__ == "__main__":
    main()

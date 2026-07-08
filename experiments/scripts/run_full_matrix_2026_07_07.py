import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DATASETS = ["nq", "hotpotqa", "msmarco"]
MODELS = {
    "qwen35": {
        "label": "qwen3.5:9b",
        "config": "model_configs/ollama_qwen3.5_config.json",
        "paper_model_name": "ollama_qwen3.5",
    },
    "llama31_8b": {
        "label": "llama3.1:8b",
        "config": "model_configs/ollama_llama3.1_8b_config.json",
        "paper_model_name": "ollama_llama3.1_8b",
    },
    "mistral_7b": {
        "label": "mistral:7b",
        "config": "model_configs/ollama_mistral_7b_config.json",
        "paper_model_name": "ollama_mistral_7b",
    },
}

RATES = ["0.01", "0.03", "0.05", "0.10"]
VARIANTS = ["original", "authority", "instruction_aware", "query_first_instruction_aware"]
SAMPLE_COUNT = 50
TOP_K = 5
ADV_PER_QUERY = 5
PAPER_M = 10
PAPER_REPEAT_TIMES = 10


def repo_root():
    return Path(__file__).resolve().parents[1]


def project_root():
    return repo_root().parents[1]


def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def completed_candidate_summary(summary_path):
    if not summary_path.exists():
        return False
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    expected_rows = 1 + len(RATES) * len(VARIANTS)
    if len(rows) < expected_rows:
        return False
    expected_n = str(SAMPLE_COUNT)
    return all(str(row.get("n", "")) == expected_n for row in rows)


def read_json_len(path):
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def paper_result_name(dataset, model_code):
    model_name = MODELS[model_code]["paper_model_name"]
    return f"{dataset}-contriever-{model_name}-top{TOP_K}-adv{ADV_PER_QUERY}-M{PAPER_M}x{PAPER_REPEAT_TIMES}"


def completed_paper_result(result_json):
    return read_json_len(result_json) >= PAPER_REPEAT_TIMES


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


def candidate_output_dir(dataset, model_code):
    root = project_root() / "03_low_rate_experiments" / "local_ollama_experiment_runs"
    existing = {
        ("nq", "qwen35"): "local_ollama_qwen35_queryfirst_nq50_2026-07-07",
        ("nq", "llama31_8b"): "local_ollama_llama31_8b_queryfirst_nq50_2026-07-07",
        ("nq", "mistral_7b"): "local_ollama_mistral_7b_queryfirst_nq50_2026-07-07",
    }
    name = existing.get(
        (dataset, model_code),
        f"local_ollama_{model_code}_queryfirst_{dataset}50_fullmatrix_2026-07-07",
    )
    return root / name


def run_candidate_matrix(status_rows):
    root = repo_root()
    for dataset in DATASETS:
        for model_code, model_info in MODELS.items():
            out_dir = candidate_output_dir(dataset, model_code)
            summary_path = out_dir / "summary.csv"
            log_path = out_dir / "fullmatrix_runner.log"
            started_at = timestamp()

            if completed_candidate_summary(summary_path):
                status_rows.append(
                    {
                        "stage": "candidate_pool",
                        "dataset": dataset,
                        "model_code": model_code,
                        "model_label": model_info["label"],
                        "status": "skipped_complete",
                        "returncode": 0,
                        "output_dir": str(out_dir),
                        "summary_path": str(summary_path),
                        "started_at": started_at,
                        "finished_at": timestamp(),
                    }
                )
                write_status(status_rows)
                print(f"[SKIP candidate] {dataset}/{model_code}: complete", flush=True)
                continue

            cmd = [
                sys.executable,
                "local_poison_rate_experiment.py",
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
                "--output_dir",
                str(out_dir),
                "--resume",
            ]
            print(f"[RUN candidate] {dataset}/{model_code}", flush=True)
            returncode = run_command(cmd, log_path, root)
            finished_at = timestamp()
            status = "complete" if returncode == 0 and completed_candidate_summary(summary_path) else "failed_or_incomplete"
            status_rows.append(
                {
                    "stage": "candidate_pool",
                    "dataset": dataset,
                    "model_code": model_code,
                    "model_label": model_info["label"],
                    "status": status,
                    "returncode": returncode,
                    "output_dir": str(out_dir),
                    "summary_path": str(summary_path),
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
            )
            write_status(status_rows)
            print(f"[DONE candidate] {dataset}/{model_code}: {status}", flush=True)


def run_paper_matrix(status_rows):
    root = repo_root()
    project = project_root()
    output_root = project / "02_reproduction_code_data" / "baseline_reproduction_outputs" / "paper_fullmatrix"
    log_dir = output_root / "logs"
    query_results_dir = "paper_fullmatrix"

    for dataset in DATASETS:
        for model_code, model_info in MODELS.items():
            if model_code == "qwen35":
                old_name = f"{dataset}-contriever-ollama_qwen3.5-top{TOP_K}-adv{ADV_PER_QUERY}-M{PAPER_M}x{PAPER_REPEAT_TIMES}"
                old_result = root / "results" / "query_results" / "paper_style_qwen35_2026-07-07" / f"{old_name}.json"
                old_log = (
                    project
                    / "02_reproduction_code_data"
                    / "baseline_reproduction_outputs"
                    / "paper_style_qwen35_2026-07-07"
                    / "logs"
                    / f"{old_name}.log"
                )
                if completed_paper_result(old_result):
                    status_rows.append(
                        {
                            "stage": "paper_style",
                            "dataset": dataset,
                            "model_code": model_code,
                            "model_label": model_info["label"],
                            "status": "skipped_existing_qwen35",
                            "returncode": 0,
                            "output_dir": str(old_result.parent),
                            "summary_path": str(old_log),
                            "started_at": timestamp(),
                            "finished_at": timestamp(),
                        }
                    )
                    write_status(status_rows)
                    print(f"[SKIP paper] {dataset}/{model_code}: existing qwen35 complete", flush=True)
                    continue

            name = paper_result_name(dataset, model_code)
            result_json = root / "results" / "query_results" / query_results_dir / f"{name}.json"
            log_path = log_dir / f"{name}.log"
            started_at = timestamp()

            if completed_paper_result(result_json):
                status_rows.append(
                    {
                        "stage": "paper_style",
                        "dataset": dataset,
                        "model_code": model_code,
                        "model_label": model_info["label"],
                        "status": "skipped_complete",
                        "returncode": 0,
                        "output_dir": str(result_json.parent),
                        "summary_path": str(log_path),
                        "started_at": started_at,
                        "finished_at": timestamp(),
                    }
                )
                write_status(status_rows)
                print(f"[SKIP paper] {dataset}/{model_code}: complete", flush=True)
                continue

            cmd = [
                sys.executable,
                "main.py",
                "--eval_dataset",
                dataset,
                "--eval_model_code",
                "contriever",
                "--model_name",
                model_info["paper_model_name"],
                "--model_config_path",
                model_info["config"],
                "--top_k",
                str(TOP_K),
                "--attack_method",
                "LM_targeted",
                "--adv_per_query",
                str(ADV_PER_QUERY),
                "--score_function",
                "dot",
                "--repeat_times",
                str(PAPER_REPEAT_TIMES),
                "--M",
                str(PAPER_M),
                "--query_results_dir",
                query_results_dir,
                "--name",
                name,
            ]
            print(f"[RUN paper] {dataset}/{model_code}", flush=True)
            returncode = run_command(cmd, log_path, root)
            finished_at = timestamp()
            status = "complete" if returncode == 0 and completed_paper_result(result_json) else "failed_or_incomplete"
            status_rows.append(
                {
                    "stage": "paper_style",
                    "dataset": dataset,
                    "model_code": model_code,
                    "model_label": model_info["label"],
                    "status": status,
                    "returncode": returncode,
                    "output_dir": str(result_json.parent),
                    "summary_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
            )
            write_status(status_rows)
            print(f"[DONE paper] {dataset}/{model_code}: {status}", flush=True)


def status_path():
    return project_root() / "03_low_rate_experiments" / "local_ollama_experiment_runs" / "fullmatrix_runner_status_2026-07-07.csv"


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


def main():
    rows = []
    print(f"[START] full matrix runner at {timestamp()}", flush=True)
    print(f"[STATUS] {status_path()}", flush=True)
    run_candidate_matrix(rows)
    run_paper_matrix(rows)
    print(f"[DONE] full matrix runner at {timestamp()}", flush=True)
    print(f"[STATUS] {status_path()}", flush=True)


if __name__ == "__main__":
    main()

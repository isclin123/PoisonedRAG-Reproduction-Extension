import csv
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(r"C:\Users\isclin\Desktop\Summer Project")
REPORT_DIR = ROOT / "04_reports_and_deliverables" / "final_experiment_report_2026-07-07"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"
STATUS_PATH = (
    ROOT
    / "03_low_rate_experiments"
    / "local_ollama_experiment_runs"
    / "fullmatrix_runner_status_2026-07-07.csv"
)
CORPUS_STATUS_PATH = (
    ROOT
    / "03_low_rate_experiments"
    / "local_ollama_experiment_runs"
    / "corpus_fullmatrix_runner_status_2026-07-07.csv"
)

DATASETS = ["nq", "hotpotqa", "msmarco"]
MODELS = ["qwen35", "llama31_8b", "mistral_7b"]
MODEL_LABELS = {
    "qwen35": "qwen3.5:9b",
    "llama31_8b": "llama3.1:8b",
    "mistral_7b": "mistral:7b",
}
DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
}
VARIANT_LABELS = {
    "original": "original",
    "authority": "authority",
    "instruction_aware": "instruction-aware",
    "query_first_instruction_aware": "query-first",
}
VARIANT_ORDER = ["original", "authority", "instruction_aware", "query_first_instruction_aware"]
RATE_ORDER = [1.0, 3.0, 5.0, 10.0]
CORPUS_VARIANT_ORDER = ["original", "query_first_instruction_aware"]
CORPUS_RATE_ORDER = [0.0001, 0.0005, 0.001, 0.005]

FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def parse_paper_metrics(log_path):
    text = Path(log_path).read_text(encoding="utf-8-sig", errors="replace")
    patterns = {
        "asr_mean": r"ASR Mean:\s*([0-9.]+)",
        "precision_mean": r"Precision mean:\s*([0-9.]+)",
        "recall_mean": r"Recall mean:\s*([0-9.]+)",
        "f1_mean": r"F1 mean:\s*([0-9.]+)",
    }
    values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        values[key] = float(match.group(1)) if match else None
    return values


def load_status():
    require(STATUS_PATH)
    status = pd.read_csv(STATUS_PATH)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    status.to_csv(TABLE_DIR / "experiment_matrix_completion_status.csv", index=False, encoding="utf-8-sig")
    return status


def load_corpus_status():
    require(CORPUS_STATUS_PATH)
    status = pd.read_csv(CORPUS_STATUS_PATH)
    status.to_csv(TABLE_DIR / "corpus_level_matrix_completion_status.csv", index=False, encoding="utf-8-sig")
    return status


def load_candidate_matrix(status):
    frames = []
    candidate = status[status["stage"] == "candidate_pool"].copy()
    for row in candidate.itertuples(index=False):
        summary_path = Path(row.summary_path)
        require(summary_path)
        frame = pd.read_csv(summary_path)
        frame["dataset"] = row.dataset
        frame["dataset_label"] = DATASET_LABELS[row.dataset]
        frame["model_code"] = row.model_code
        frame["model_label"] = row.model_label
        frame["poison_rate_percent"] = frame["requested_poison_rate"].astype(float) * 100
        frames.append(frame)

    matrix = pd.concat(frames, ignore_index=True)
    matrix["variant_label"] = matrix["variant"].map(lambda value: VARIANT_LABELS.get(value, value))
    matrix["is_clean"] = matrix["variant"] == "clean"
    matrix.to_csv(TABLE_DIR / "candidate_pool_full_matrix_summary.csv", index=False, encoding="utf-8-sig")

    focus = matrix[~matrix["is_clean"]].copy()
    for metric in [
        "asr",
        "correct_answer_rate",
        "retrieval_success_rate",
        "avg_poison_in_topk",
        "avg_poison_topk_share",
    ]:
        pivot = focus.pivot_table(
            index=["dataset", "model_label", "variant"],
            columns="poison_rate_percent",
            values=metric,
            aggfunc="first",
        ).reset_index()
        pivot.to_csv(TABLE_DIR / f"candidate_pool_full_matrix_{metric}_pivot.csv", index=False, encoding="utf-8-sig")

    clean = matrix[matrix["is_clean"]].copy()
    clean.to_csv(TABLE_DIR / "candidate_pool_clean_baseline_full_matrix.csv", index=False, encoding="utf-8-sig")
    return matrix


def load_paper_matrix(status):
    rows = []
    paper = status[status["stage"] == "paper_style"].copy()
    for row in paper.itertuples(index=False):
        log_path = Path(row.summary_path)
        require(log_path)
        metrics = parse_paper_metrics(log_path)
        rows.append(
            {
                "dataset": row.dataset,
                "dataset_label": DATASET_LABELS[row.dataset],
                "model_code": row.model_code,
                "model_label": row.model_label,
                "status": row.status,
                "log_path": str(log_path),
                **metrics,
            }
        )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(TABLE_DIR / "paper_style_full_matrix_summary.csv", index=False, encoding="utf-8-sig")
    return matrix


def load_corpus_matrix(status):
    frames = []
    for row in status.itertuples(index=False):
        summary_path = Path(row.summary_path)
        require(summary_path)
        frame = pd.read_csv(summary_path)
        frame["dataset"] = row.dataset
        frame["dataset_label"] = DATASET_LABELS[row.dataset]
        frame["model_code"] = row.model_code
        frame["model_label"] = row.model_label
        frame["generation_profile"] = "fast" if row.model_code == "mistral_7b" else "ultrafast"
        frames.append(frame)

    matrix = pd.concat(frames, ignore_index=True)
    matrix["variant_label"] = matrix["variant"].map(lambda value: VARIANT_LABELS.get(value, value))
    matrix["is_clean"] = matrix["variant"] == "clean"
    matrix.to_csv(TABLE_DIR / "corpus_level_full_matrix_summary.csv", index=False, encoding="utf-8-sig")

    focus = matrix[~matrix["is_clean"]].copy()
    for metric in [
        "asr",
        "correct_answer_rate",
        "retrieval_success_rate",
        "avg_poison_in_topk",
        "avg_poison_topk_share",
    ]:
        pivot = focus.pivot_table(
            index=["dataset", "model_label", "variant"],
            columns="requested_corpus_poison_rate_percent",
            values=metric,
            aggfunc="first",
        ).reset_index()
        pivot.to_csv(TABLE_DIR / f"corpus_level_full_matrix_{metric}_pivot.csv", index=False, encoding="utf-8-sig")

    clean = matrix[matrix["is_clean"]].copy()
    clean.to_csv(TABLE_DIR / "corpus_level_clean_baseline_full_matrix.csv", index=False, encoding="utf-8-sig")
    return matrix


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_candidate_metric_grid(matrix, metric, ylabel, title, filename, ylim=None):
    focus = matrix[(matrix["variant"] != "clean") & (matrix["variant"].isin(VARIANT_ORDER))].copy()
    colors = {
        "original": "#1f77b4",
        "authority": "#d62728",
        "instruction_aware": "#9467bd",
        "query_first_instruction_aware": "#2ca02c",
    }
    markers = {
        "original": "o",
        "authority": "s",
        "instruction_aware": "^",
        "query_first_instruction_aware": "D",
    }

    fig, axes = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(14.5, 10.5),
        sharex=True,
        sharey=ylim is not None,
    )
    for i, dataset in enumerate(DATASETS):
        for j, model_code in enumerate(MODELS):
            ax = axes[i][j]
            subset = focus[(focus["dataset"] == dataset) & (focus["model_code"] == model_code)]
            for variant in VARIANT_ORDER:
                rows = subset[subset["variant"] == variant].sort_values("poison_rate_percent")
                if rows.empty:
                    continue
                ax.plot(
                    rows["poison_rate_percent"],
                    rows[metric],
                    color=colors[variant],
                    marker=markers[variant],
                    linewidth=1.8,
                    markersize=4,
                    label=VARIANT_LABELS[variant],
                )
            ax.set_title(f"{DATASET_LABELS[dataset]} / {MODEL_LABELS[model_code]}", fontproperties=FONT, fontsize=10)
            ax.set_xticks(RATE_ORDER)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Poison rate (%)", fontproperties=FONT)
            if j == 0:
                ax.set_ylabel(ylabel, fontproperties=FONT)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), prop=FONT)
    fig.suptitle(title, fontproperties=FONT, fontsize=15, y=1.055)
    save_fig(FIG_DIR / filename)


def plot_corpus_metric_grid(matrix, metric, ylabel, title, filename, ylim=None):
    focus = matrix[(matrix["variant"] != "clean") & (matrix["variant"].isin(CORPUS_VARIANT_ORDER))].copy()
    colors = {
        "original": "#1f77b4",
        "query_first_instruction_aware": "#2ca02c",
    }
    markers = {
        "original": "o",
        "query_first_instruction_aware": "D",
    }

    fig, axes = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(14.5, 10.5),
        sharex=True,
        sharey=ylim is not None,
    )
    for i, dataset in enumerate(DATASETS):
        for j, model_code in enumerate(MODELS):
            ax = axes[i][j]
            subset = focus[(focus["dataset"] == dataset) & (focus["model_code"] == model_code)]
            for variant in CORPUS_VARIANT_ORDER:
                rows = subset[subset["variant"] == variant].sort_values("requested_corpus_poison_rate_percent")
                if rows.empty:
                    continue
                ax.plot(
                    rows["requested_corpus_poison_rate_percent"],
                    rows[metric],
                    color=colors[variant],
                    marker=markers[variant],
                    linewidth=1.8,
                    markersize=4,
                    label=VARIANT_LABELS[variant],
                )
            ax.set_title(f"{DATASET_LABELS[dataset]} / {MODEL_LABELS[model_code]}", fontproperties=FONT, fontsize=10)
            ax.set_xticks(CORPUS_RATE_ORDER)
            ax.tick_params(axis="x", rotation=30)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Corpus poison rate (%)", fontproperties=FONT)
            if j == 0:
                ax.set_ylabel(ylabel, fontproperties=FONT)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), prop=FONT)
    fig.suptitle(title, fontproperties=FONT, fontsize=15, y=1.055)
    save_fig(FIG_DIR / filename)


def plot_clean_baseline(matrix):
    clean = matrix[matrix["variant"] == "clean"].copy()
    metrics = [("asr", "Clean ASR"), ("correct_answer_rate", "Clean correct-answer rate")]
    ordered_columns = [MODEL_LABELS[model] for model in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (metric, title) in zip(axes, metrics):
        values = clean.pivot(index="dataset_label", columns="model_label", values=metric).reindex(
            [DATASET_LABELS[d] for d in DATASETS]
        )[ordered_columns]
        image = ax.imshow(values.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(values.shape[1]), values.columns, rotation=25, ha="right", fontproperties=FONT)
        ax.set_yticks(range(values.shape[0]), values.index, fontproperties=FONT)
        ax.set_title(title, fontproperties=FONT)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Clean baseline by dataset and LLM", fontproperties=FONT, fontsize=14)
    save_fig(FIG_DIR / "candidate_fullmatrix_clean_baseline.png")


def plot_corpus_clean_baseline(matrix):
    clean = matrix[matrix["variant"] == "clean"].copy()
    metrics = [("asr", "Clean ASR"), ("correct_answer_rate", "Clean correct-answer rate")]
    ordered_columns = [MODEL_LABELS[model] for model in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (metric, title) in zip(axes, metrics):
        values = clean.pivot(index="dataset_label", columns="model_label", values=metric).reindex(
            [DATASET_LABELS[d] for d in DATASETS]
        )[ordered_columns]
        image = ax.imshow(values.values, cmap="PuBuGn", vmin=0, vmax=1)
        ax.set_xticks(range(values.shape[1]), values.columns, rotation=25, ha="right", fontproperties=FONT)
        ax.set_yticks(range(values.shape[0]), values.index, fontproperties=FONT)
        ax.set_title(title, fontproperties=FONT)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Corpus-level clean baseline by dataset and LLM", fontproperties=FONT, fontsize=14)
    save_fig(FIG_DIR / "corpus_fullmatrix_clean_baseline.png")


def plot_paper_matrix(paper):
    metrics = [
        ("asr_mean", "ASR"),
        ("precision_mean", "Precision"),
        ("recall_mean", "Recall"),
        ("f1_mean", "F1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    ordered_columns = [MODEL_LABELS[model] for model in MODELS]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = paper.pivot(index="dataset_label", columns="model_label", values=metric).reindex(
            [DATASET_LABELS[d] for d in DATASETS]
        )[ordered_columns]
        image = ax.imshow(values.values, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(values.shape[1]), values.columns, rotation=25, ha="right", fontproperties=FONT)
        ax.set_yticks(range(values.shape[0]), values.index, fontproperties=FONT)
        ax.set_title(title, fontproperties=FONT)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Paper-style PoisonedRAG reproduction: 3 datasets x 3 LLMs", fontproperties=FONT, fontsize=14)
    save_fig(FIG_DIR / "paper_style_fullmatrix_metrics.png")


def plot_all(candidate, paper, corpus):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_candidate_metric_grid(
        candidate,
        "asr",
        "ASR",
        "Candidate-pool ASR: 3 datasets x 3 LLMs x 4 variants",
        "candidate_fullmatrix_asr_grid.png",
        ylim=(0, 1.05),
    )
    plot_candidate_metric_grid(
        candidate,
        "correct_answer_rate",
        "Correct-answer rate",
        "Candidate-pool correct-answer rate",
        "candidate_fullmatrix_correct_answer_rate_grid.png",
        ylim=(0, 1.05),
    )
    plot_candidate_metric_grid(
        candidate,
        "retrieval_success_rate",
        "Retrieval success",
        "Poison retrieval success: poison appears in top-k",
        "candidate_fullmatrix_retrieval_success_grid.png",
        ylim=(0, 1.05),
    )
    plot_candidate_metric_grid(
        candidate,
        "avg_poison_in_topk",
        "Avg poison in top-5",
        "Average number of poison passages in retrieved top-5",
        "candidate_fullmatrix_avg_poison_in_topk_grid.png",
        ylim=(0, 5.25),
    )
    plot_candidate_metric_grid(
        candidate,
        "avg_poison_topk_share",
        "Poison top-k share",
        "Average poison share in retrieved top-5",
        "candidate_fullmatrix_avg_poison_topk_share_grid.png",
        ylim=(0, 1.05),
    )
    plot_clean_baseline(candidate)
    plot_paper_matrix(paper)
    plot_corpus_metric_grid(
        corpus,
        "asr",
        "True ASR",
        "Corpus-level true ASR: 3 datasets x 3 LLMs x 2 variants",
        "corpus_fullmatrix_asr_grid.png",
        ylim=(0, 1.05),
    )
    plot_corpus_metric_grid(
        corpus,
        "correct_answer_rate",
        "Correct-answer rate",
        "Corpus-level correct-answer rate",
        "corpus_fullmatrix_correct_answer_rate_grid.png",
        ylim=(0, 1.05),
    )
    plot_corpus_metric_grid(
        corpus,
        "retrieval_success_rate",
        "Retrieval success",
        "Corpus-level poison retrieval success",
        "corpus_fullmatrix_retrieval_success_grid.png",
        ylim=(0, 1.05),
    )
    plot_corpus_metric_grid(
        corpus,
        "avg_poison_in_topk",
        "Avg poison in top-5",
        "Corpus-level average number of poison passages in retrieved top-5",
        "corpus_fullmatrix_avg_poison_in_topk_grid.png",
        ylim=(0, 5.25),
    )
    plot_corpus_metric_grid(
        corpus,
        "avg_poison_topk_share",
        "Poison top-k share",
        "Corpus-level average poison share in retrieved top-5",
        "corpus_fullmatrix_avg_poison_topk_share_grid.png",
        ylim=(0, 1.05),
    )
    plot_corpus_clean_baseline(corpus)


def validate(candidate, paper, corpus):
    expected_candidate_rows = len(DATASETS) * len(MODELS) * (1 + len(VARIANT_ORDER) * len(RATE_ORDER))
    expected_paper_rows = len(DATASETS) * len(MODELS)
    expected_corpus_rows = len(DATASETS) * len(MODELS) * (1 + len(CORPUS_VARIANT_ORDER) * len(CORPUS_RATE_ORDER))
    if len(candidate) != expected_candidate_rows:
        raise AssertionError(f"candidate rows: expected {expected_candidate_rows}, got {len(candidate)}")
    if len(paper) != expected_paper_rows:
        raise AssertionError(f"paper rows: expected {expected_paper_rows}, got {len(paper)}")
    if len(corpus) != expected_corpus_rows:
        raise AssertionError(f"corpus rows: expected {expected_corpus_rows}, got {len(corpus)}")
    for metric in ["asr", "correct_answer_rate", "retrieval_success_rate", "avg_poison_in_topk", "avg_poison_topk_share"]:
        if candidate[metric].isna().any():
            raise AssertionError(f"candidate metric has NA: {metric}")
        if corpus[metric].isna().any():
            raise AssertionError(f"corpus metric has NA: {metric}")
    for metric in ["asr_mean", "precision_mean", "recall_mean", "f1_mean"]:
        if paper[metric].isna().any():
            raise AssertionError(f"paper metric has NA: {metric}")


def main():
    status = load_status()
    corpus_status = load_corpus_status()
    candidate = load_candidate_matrix(status)
    paper = load_paper_matrix(status)
    corpus = load_corpus_matrix(corpus_status)
    validate(candidate, paper, corpus)
    plot_all(candidate, paper, corpus)
    print(TABLE_DIR / "candidate_pool_full_matrix_summary.csv")
    print(TABLE_DIR / "paper_style_full_matrix_summary.csv")
    print(TABLE_DIR / "corpus_level_full_matrix_summary.csv")
    print(FIG_DIR)


if __name__ == "__main__":
    main()

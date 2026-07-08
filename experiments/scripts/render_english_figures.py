from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

DATASETS = ["nq", "hotpotqa", "msmarco"]
MODELS = ["qwen35", "llama31_8b", "mistral_7b"]
DATASET_LABELS = {"nq": "NQ", "hotpotqa": "HotpotQA", "msmarco": "MS MARCO"}
MODEL_LABELS = {
    "qwen35": "qwen3.5:9b",
    "llama31_8b": "llama3.1:8b",
    "mistral_7b": "mistral:7b",
}
VARIANT_LABELS = {
    "original": "Original",
    "authority": "Authority",
    "instruction_aware": "Instruction-aware",
    "query_first_instruction_aware": "Query-first",
}
VARIANT_ORDER = ["original", "authority", "instruction_aware", "query_first_instruction_aware"]
CORPUS_VARIANT_ORDER = ["original", "query_first_instruction_aware"]
RATE_ORDER = [1.0, 3.0, 5.0, 10.0]
CORPUS_RATE_ORDER = [0.0001, 0.0005, 0.001, 0.005]


def read_table(name):
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finish(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_metric_grid(data, metric, ylabel, title, filename, rate_column, rates, variants, x_label, ylim=None):
    colors = {
        "original": "#2563eb",
        "authority": "#dc2626",
        "instruction_aware": "#7c3aed",
        "query_first_instruction_aware": "#16a34a",
    }
    markers = {
        "original": "o",
        "authority": "s",
        "instruction_aware": "^",
        "query_first_instruction_aware": "D",
    }
    focus = data[(data["variant"] != "clean") & (data["variant"].isin(variants))].copy()
    fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(14.5, 10.5), sharex=True, sharey=ylim is not None)

    for i, dataset in enumerate(DATASETS):
        for j, model_code in enumerate(MODELS):
            ax = axes[i][j]
            subset = focus[(focus["dataset"] == dataset) & (focus["model_code"] == model_code)]
            for variant in variants:
                rows = subset[subset["variant"] == variant].sort_values(rate_column)
                if rows.empty:
                    continue
                ax.plot(
                    rows[rate_column],
                    rows[metric],
                    color=colors[variant],
                    marker=markers[variant],
                    linewidth=1.8,
                    markersize=4,
                    label=VARIANT_LABELS[variant],
                )
            ax.set_title(f"{DATASET_LABELS[dataset]} / {MODEL_LABELS[model_code]}", fontsize=10)
            ax.set_xticks(rates)
            if len(rates) > 4:
                ax.tick_params(axis="x", rotation=30)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
            if i == len(DATASETS) - 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(ylabel)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(variants), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=15, y=1.055)
    finish(FIG_DIR / filename)


def plot_clean_baseline(data, filename, title, cmap):
    clean = data[data["variant"] == "clean"].copy()
    metrics = [("asr", "Clean ASR"), ("correct_answer_rate", "Clean correct-answer rate")]
    ordered_columns = [MODEL_LABELS[model] for model in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (metric, panel_title) in zip(axes, metrics):
        values = clean.pivot(index="dataset_label", columns="model_label", values=metric).reindex(
            [DATASET_LABELS[d] for d in DATASETS]
        )[ordered_columns]
        image = ax.imshow(values.values, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(values.shape[1]), values.columns, rotation=25, ha="right")
        ax.set_yticks(range(values.shape[0]), values.index)
        ax.set_title(panel_title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    finish(FIG_DIR / filename)


def plot_paper_matrix(paper):
    metrics = [
        ("asr_mean", "ASR"),
        ("precision_mean", "Precision"),
        ("recall_mean", "Recall"),
        ("f1_mean", "F1"),
    ]
    ordered_columns = [MODEL_LABELS[model] for model in MODELS]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    for ax, (metric, panel_title) in zip(axes.ravel(), metrics):
        values = paper.pivot(index="dataset_label", columns="model_label", values=metric).reindex(
            [DATASET_LABELS[d] for d in DATASETS]
        )[ordered_columns]
        image = ax.imshow(values.values, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(values.shape[1]), values.columns, rotation=25, ha="right")
        ax.set_yticks(range(values.shape[0]), values.index)
        ax.set_title(panel_title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Paper-style PoisonedRAG reproduction: 3 datasets x 3 LLMs", fontsize=14)
    finish(FIG_DIR / "paper_style_fullmatrix_metrics.png")


def plot_qwen_paper(paper_qwen):
    rows = paper_qwen.copy()
    rows["dataset_label"] = rows["dataset"].map(DATASET_LABELS)
    metrics = ["asr_mean", "precision_mean", "recall_mean", "f1_mean"]
    labels = ["ASR", "Precision", "Recall", "F1"]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for metric, label, offset in zip(metrics, labels, offsets):
        ax.bar([v + width * offset for v in x], rows[metric], width=width, label=label)
    ax.set_xticks(list(x), rows["dataset_label"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric value")
    ax.set_title("Paper-style reproduction with qwen3.5:9b")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    finish(FIG_DIR / "paper_style_reproduction_qwen35.png")


def plot_qwen_strategy(candidate):
    subset = candidate[(candidate["model_code"] == "qwen35") & (candidate["variant"] != "clean")].copy()
    grouped = subset.groupby(["variant", "poison_rate_percent"], as_index=False)["asr"].mean()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for variant in VARIANT_ORDER:
        rows = grouped[grouped["variant"] == variant].sort_values("poison_rate_percent")
        ax.plot(rows["poison_rate_percent"], rows["asr"], marker="o", linewidth=2, label=VARIANT_LABELS[variant])
    ax.set_xticks(RATE_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Candidate-pool poison rate (%)")
    ax.set_ylabel("Average ASR")
    ax.set_title("qwen3.5:9b candidate-pool ASR by poison strategy")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    finish(FIG_DIR / "qwen35_strategy_asr.png")


def plot_candidate_multi_model(candidate_multi, metric, ylabel, title, filename):
    subset = candidate_multi[candidate_multi["variant"] != "clean"].copy()
    subset["variant_label"] = subset["variant"].map(VARIANT_LABELS)
    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    for (model, variant), rows in subset.groupby(["model", "variant"]):
        rows = rows.sort_values("poison_rate_percent")
        label = f"{model} / {VARIANT_LABELS.get(variant, variant)}"
        ax.plot(rows["poison_rate_percent"], rows[metric], marker="o", linewidth=1.8, label=label)
    ax.set_xticks(RATE_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Candidate-pool poison rate (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    finish(FIG_DIR / filename)


def plot_corpus_mistral(corpus_mistral, metric, ylabel, title, filename, ylim=None):
    subset = corpus_mistral[corpus_mistral["variant"] != "clean"].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for variant, rows in subset.groupby("variant"):
        rows = rows.sort_values("corpus_rate_percent")
        ax.plot(rows["corpus_rate_percent"], rows[metric], marker="o", linewidth=2, label=VARIANT_LABELS.get(variant, variant))
    ax.set_xticks(CORPUS_RATE_ORDER)
    ax.tick_params(axis="x", rotation=30)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Corpus poison rate (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    finish(FIG_DIR / filename)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    candidate = read_table("candidate_pool_full_matrix_summary.csv")
    corpus = read_table("corpus_level_full_matrix_summary.csv")
    paper = read_table("paper_style_full_matrix_summary.csv")
    qwen_paper = read_table("paper_style_qwen35_summary.csv")
    candidate_multi = read_table("candidate_pool_multi_model_summary.csv")
    corpus_mistral = read_table("corpus_level_true_asr_mistral_summary.csv")

    plot_metric_grid(candidate, "asr", "ASR", "Candidate-pool ASR: 3 datasets x 3 LLMs x 4 variants",
                     "candidate_fullmatrix_asr_grid.png", "poison_rate_percent", RATE_ORDER, VARIANT_ORDER,
                     "Candidate-pool poison rate (%)", ylim=(0, 1.05))
    plot_metric_grid(candidate, "correct_answer_rate", "Correct-answer rate", "Candidate-pool correct-answer rate",
                     "candidate_fullmatrix_correct_answer_rate_grid.png", "poison_rate_percent", RATE_ORDER, VARIANT_ORDER,
                     "Candidate-pool poison rate (%)", ylim=(0, 1.05))
    plot_metric_grid(candidate, "retrieval_success_rate", "Retrieval success", "Candidate-pool poison retrieval success",
                     "candidate_fullmatrix_retrieval_success_grid.png", "poison_rate_percent", RATE_ORDER, VARIANT_ORDER,
                     "Candidate-pool poison rate (%)", ylim=(0, 1.05))
    plot_metric_grid(candidate, "avg_poison_in_topk", "Avg poison in top-5",
                     "Candidate-pool average number of poison passages in top-5",
                     "candidate_fullmatrix_avg_poison_in_topk_grid.png", "poison_rate_percent", RATE_ORDER, VARIANT_ORDER,
                     "Candidate-pool poison rate (%)", ylim=(0, 5.25))
    plot_metric_grid(candidate, "avg_poison_topk_share", "Poison top-k share",
                     "Candidate-pool average poison share in top-5",
                     "candidate_fullmatrix_avg_poison_topk_share_grid.png", "poison_rate_percent", RATE_ORDER, VARIANT_ORDER,
                     "Candidate-pool poison rate (%)", ylim=(0, 1.05))
    plot_clean_baseline(candidate, "candidate_fullmatrix_clean_baseline.png", "Candidate-pool clean baseline by dataset and LLM", "Blues")

    plot_metric_grid(corpus, "asr", "True ASR", "Corpus-level true ASR: 3 datasets x 3 LLMs x 2 variants",
                     "corpus_fullmatrix_asr_grid.png", "requested_corpus_poison_rate_percent", CORPUS_RATE_ORDER,
                     CORPUS_VARIANT_ORDER, "Corpus poison rate (%)", ylim=(0, 1.05))
    plot_metric_grid(corpus, "correct_answer_rate", "Correct-answer rate", "Corpus-level correct-answer rate",
                     "corpus_fullmatrix_correct_answer_rate_grid.png", "requested_corpus_poison_rate_percent",
                     CORPUS_RATE_ORDER, CORPUS_VARIANT_ORDER, "Corpus poison rate (%)", ylim=(0, 1.05))
    plot_metric_grid(corpus, "retrieval_success_rate", "Retrieval success", "Corpus-level poison retrieval success",
                     "corpus_fullmatrix_retrieval_success_grid.png", "requested_corpus_poison_rate_percent",
                     CORPUS_RATE_ORDER, CORPUS_VARIANT_ORDER, "Corpus poison rate (%)", ylim=(0, 1.05))
    plot_metric_grid(corpus, "avg_poison_in_topk", "Avg poison in top-5",
                     "Corpus-level average number of poison passages in top-5",
                     "corpus_fullmatrix_avg_poison_in_topk_grid.png", "requested_corpus_poison_rate_percent",
                     CORPUS_RATE_ORDER, CORPUS_VARIANT_ORDER, "Corpus poison rate (%)", ylim=(0, 5.25))
    plot_metric_grid(corpus, "avg_poison_topk_share", "Poison top-k share",
                     "Corpus-level average poison share in top-5",
                     "corpus_fullmatrix_avg_poison_topk_share_grid.png", "requested_corpus_poison_rate_percent",
                     CORPUS_RATE_ORDER, CORPUS_VARIANT_ORDER, "Corpus poison rate (%)", ylim=(0, 1.05))
    plot_clean_baseline(corpus, "corpus_fullmatrix_clean_baseline.png", "Corpus-level clean baseline by dataset and LLM", "PuBuGn")

    plot_paper_matrix(paper)
    plot_qwen_paper(qwen_paper)
    plot_qwen_strategy(candidate)
    plot_candidate_multi_model(candidate_multi, "asr", "ASR", "Candidate-pool ASR by local LLM", "candidate_pool_asr_multi_model.png")
    plot_candidate_multi_model(candidate_multi, "retrieval_success_rate", "Retrieval success",
                               "Candidate-pool poison retrieval success by local LLM",
                               "candidate_pool_retrieval_multi_model.png")
    plot_corpus_mistral(corpus_mistral, "asr", "True ASR", "Corpus-level true ASR with mistral:7b",
                        "corpus_level_true_asr_mistral.png", ylim=(0, 1.05))
    plot_corpus_mistral(corpus_mistral, "avg_poison_in_topk", "Avg poison in top-5",
                        "Corpus-level poison exposure with mistral:7b",
                        "corpus_level_poison_topk_mistral.png", ylim=(0, 5.25))

    rendered = sorted(p.name for p in FIG_DIR.glob("*.png"))
    print(f"Rendered {len(rendered)} PNG figures in {FIG_DIR}")
    for name in rendered:
        print(name)


if __name__ == "__main__":
    main()

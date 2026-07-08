import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

from local_poison_rate_experiment import (
    answer_contains,
    build_poison_text,
    score_poison_texts,
    sorted_retrieval_items,
    truncate_context,
    write_csv,
    write_json,
)
from src.models import create_model
from src.prompts import wrap_prompt
from src.utils import load_beir_datasets, load_models


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def make_corpus_level_poison_count(rate, corpus_size, min_poison=1, max_poison_docs=None):
    if rate <= 0:
        return 0
    n_poison = max(min_poison, round(rate * corpus_size))
    if max_poison_docs is not None:
        n_poison = min(n_poison, max_poison_docs)
    return int(n_poison)


def build_corpus_conditions(rates, variants, corpus_size, include_clean, max_poison_docs=None):
    conditions = []
    if include_clean:
        conditions.append(
            {
                "condition": "clean",
                "variant": "clean",
                "requested_corpus_poison_rate": 0.0,
                "n_poison": 0,
            }
        )

    for rate in rates:
        n_poison = make_corpus_level_poison_count(
            rate,
            corpus_size,
            max_poison_docs=max_poison_docs,
        )
        for variant in variants:
            conditions.append(
                {
                    "condition": f"{variant}_corpus_{rate * 100:.4f}%",
                    "variant": variant,
                    "requested_corpus_poison_rate": rate,
                    "n_poison": n_poison,
                }
            )
    return conditions


def expand_poison_texts(question, adv_texts, variant, n_poison, max_context_chars):
    if n_poison <= 0:
        return []
    if not adv_texts:
        raise ValueError("Target item does not contain adv_texts.")

    poison_texts = []
    for index in range(n_poison):
        adv_text = adv_texts[index % len(adv_texts)]
        poison_texts.append(
            truncate_context(
                build_poison_text(question, adv_text, variant),
                max_context_chars,
            )
        )
    return poison_texts


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[
            (
                row["condition"],
                row["variant"],
                row["requested_corpus_poison_rate"],
            )
        ].append(row)

    summaries = []
    for (condition, variant, rate), rows in sorted(grouped.items(), key=lambda x: (x[0][2], x[0][1])):
        n = len(rows)
        summaries.append(
            {
                "condition": condition,
                "variant": variant,
                "requested_corpus_poison_rate": rate,
                "requested_corpus_poison_rate_percent": rate * 100,
                "actual_corpus_poison_rate": sum(row["actual_corpus_poison_rate"] for row in rows) / n,
                "actual_corpus_poison_rate_percent": (
                    sum(row["actual_corpus_poison_rate"] for row in rows) / n
                )
                * 100,
                "corpus_size": rows[0]["corpus_size"],
                "n_poison": rows[0]["n_poison"],
                "n": n,
                "asr": sum(row["attack_success"] for row in rows) / n,
                "correct_answer_rate": sum(row["correct_answer_present"] for row in rows) / n,
                "retrieval_success_rate": sum(row["poison_in_topk"] > 0 for row in rows) / n,
                "avg_poison_in_topk": sum(row["poison_in_topk"] for row in rows) / n,
                "avg_poison_topk_share": sum(row["poison_topk_share"] for row in rows) / n,
            }
        )
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Corpus-level low-rate PoisonedRAG experiment")
    parser.add_argument("--dataset", default="nq", choices=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval_model_code", default="contriever")
    parser.add_argument("--llm_config", default="model_configs/ollama_qwen3.5_config.json")
    parser.add_argument("--sample_count", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[0.000001, 0.000005, 0.00001, 0.00005],
        help="Corpus-level rates as fractions, e.g. 0.000001 means 0.0001%.",
    )
    parser.add_argument("--variants", nargs="+", default=["original", "query_first_instruction_aware"])
    parser.add_argument("--score_function", default="dot", choices=["dot", "cos_sim"])
    parser.add_argument("--max_context_chars", type=int, default=900)
    parser.add_argument("--max_poison_docs", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_clean", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(0)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    project_root = Path(__file__).resolve().parents[2]
    default_output_root = project_root / "03_low_rate_experiments" / "local_ollama_experiment_runs"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_root / f"corpus_level_local_ollama_qwen35_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels = load_beir_datasets(args.dataset, args.split)
    corpus_size = len(corpus)

    metadata = {
        "created_at": timestamp,
        "dataset": args.dataset,
        "split": args.split,
        "eval_model_code": args.eval_model_code,
        "llm_config": args.llm_config,
        "sample_count": args.sample_count,
        "top_k": args.top_k,
        "rates": args.rates,
        "variants": args.variants,
        "score_function": args.score_function,
        "max_context_chars": args.max_context_chars,
        "max_poison_docs": args.max_poison_docs,
        "corpus_size": corpus_size,
        "resume": args.resume,
        "poison_rate_definition": "n_poison / full BEIR corpus size; poison documents are directly scored and inserted against the cached full-corpus retrieval top-k threshold",
        "poison_text_expansion": "If a target has fewer generated adversarial texts than n_poison, the existing adversarial texts are repeated to represent duplicated poisoned passages.",
    }
    write_json(output_dir / "metadata.json", metadata)

    with open(f"results/adv_targeted_results/{args.dataset}.json", encoding="utf-8") as f:
        adv_targets = json.load(f)
    with open(f"results/beir_results/{args.dataset}-{args.eval_model_code}.json", encoding="utf-8") as f:
        retrieval_results = json.load(f)

    retriever, _, tokenizer, get_emb = load_models(args.eval_model_code)
    retriever.eval()
    retriever.to("cuda")

    llm = create_model(args.llm_config)
    conditions = build_corpus_conditions(
        args.rates,
        args.variants,
        corpus_size,
        not args.no_clean,
        max_poison_docs=args.max_poison_docs,
    )

    target_items = list(adv_targets.values())[: args.sample_count]
    partial_path = output_dir / "records.partial.json"
    if args.resume and partial_path.exists():
        records = json.loads(partial_path.read_text(encoding="utf-8"))
        done_keys = {(row["query_id"], row["condition"]) for row in records}
        print(f"Resuming with {len(records)} existing records.", flush=True)
    else:
        records = []
        done_keys = set()

    for target_index, target in enumerate(target_items, start=1):
        qid = target["id"]
        question = target["question"]
        print(f"[{target_index}/{len(target_items)}] {qid}: {question}", flush=True)

        retrieval_items = sorted_retrieval_items(retrieval_results[qid])
        baseline_candidates = []
        for doc_id, score in retrieval_items[: args.top_k]:
            baseline_candidates.append(
                {
                    "label": "benign",
                    "doc_id": doc_id,
                    "score": float(score),
                    "context": truncate_context(corpus[doc_id]["text"], args.max_context_chars),
                }
            )

        for condition in conditions:
            n_poison = condition["n_poison"]
            variant = condition["variant"]
            condition_key = (qid, condition["condition"])
            if condition_key in done_keys:
                print(f"  {condition['condition']}: skipped existing record", flush=True)
                continue

            candidates = list(baseline_candidates)

            if n_poison:
                poison_texts = expand_poison_texts(
                    question,
                    target["adv_texts"],
                    variant,
                    n_poison,
                    args.max_context_chars,
                )
                poison_scores = score_poison_texts(
                    question,
                    poison_texts,
                    retriever,
                    tokenizer,
                    get_emb,
                    args.score_function,
                )
                for idx, (poison_text, score) in enumerate(zip(poison_texts, poison_scores)):
                    candidates.append(
                        {
                            "label": "poison",
                            "doc_id": f"poison_{idx}",
                            "score": float(score),
                            "context": poison_text,
                        }
                    )

            ranked = sorted(candidates, key=lambda row: row["score"], reverse=True)
            selected = ranked[: args.top_k]
            contexts = [row["context"] for row in selected]
            prompt = wrap_prompt(question, contexts, prompt_id=4)

            started = time.time()
            response = llm.query(prompt)
            latency_seconds = time.time() - started

            poison_in_topk = sum(row["label"] == "poison" for row in selected)
            actual_corpus_rate = n_poison / corpus_size if corpus_size else 0
            row = {
                "condition": condition["condition"],
                "variant": variant,
                "requested_corpus_poison_rate": condition["requested_corpus_poison_rate"],
                "requested_corpus_poison_rate_percent": condition["requested_corpus_poison_rate"] * 100,
                "actual_corpus_poison_rate": actual_corpus_rate,
                "actual_corpus_poison_rate_percent": actual_corpus_rate * 100,
                "corpus_size": corpus_size,
                "n_poison": n_poison,
                "n_benign_competitors": args.top_k,
                "top_k": args.top_k,
                "sample_index": target_index,
                "query_id": qid,
                "question": question,
                "correct_answer": target["correct answer"],
                "incorrect_answer": target["incorrect answer"],
                "response": response,
                "attack_success": answer_contains(target["incorrect answer"], response),
                "correct_answer_present": answer_contains(target["correct answer"], response),
                "poison_in_topk": poison_in_topk,
                "poison_topk_share": poison_in_topk / args.top_k,
                "latency_seconds": round(latency_seconds, 3),
                "selected_labels": "|".join(row["label"] for row in selected),
                "selected_doc_ids": "|".join(row["doc_id"] for row in selected),
                "selected_scores": "|".join(f"{row['score']:.6f}" for row in selected),
                "selected_context_previews": json.dumps(
                    [row["context"][:240] for row in selected],
                    ensure_ascii=False,
                ),
            }
            records.append(row)
            done_keys.add(condition_key)
            write_json(partial_path, records)
            print(
                f"  {row['condition']}: ASR={int(row['attack_success'])} "
                f"poison_topk={poison_in_topk}/{args.top_k} "
                f"response={response[:80]!r}",
                flush=True,
            )

    summaries = summarize(records)
    write_json(output_dir / "records.json", records)
    write_csv(output_dir / "records.csv", records)
    write_json(output_dir / "summary.json", summaries)
    write_csv(output_dir / "summary.csv", summaries)

    print(f"\nSaved outputs to {output_dir}", flush=True)
    for row in summaries:
        print(
            f"{row['condition']}: n={row['n']} ASR={row['asr']:.2f} "
            f"retrieval={row['retrieval_success_rate']:.2f} "
            f"avg_poison_topk={row['avg_poison_in_topk']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

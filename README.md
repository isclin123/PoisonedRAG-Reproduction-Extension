# PoisonedRAG Reproduction & Low-Rate Poisoning Study 🧪

<p align="center">
  <strong>Phase 1 materials for reproducing and extending RAG knowledge-corruption attacks under low poison rates.</strong>
</p>

<p align="center">
  <img alt="Phase" src="https://img.shields.io/badge/Phase-1%20Completed-brightgreen">
  <img alt="Datasets" src="https://img.shields.io/badge/Datasets-NQ%20%7C%20HotpotQA%20%7C%20MS%20MARCO-blue">
  <img alt="Models" src="https://img.shields.io/badge/Local%20LLMs-Qwen%20%7C%20Llama%20%7C%20Mistral-purple">
  <img alt="Status" src="https://img.shields.io/badge/Results-Released-orange">
</p>

---

## 🌱 What This Repository Is

This repository contains the **Phase 1** deliverables for a RAG security project on low-rate knowledge-corruption attacks. The starting point was the observation that existing PoisonedRAG-style attacks can be very strong when poisoned documents dominate retrieved evidence, but their success may drop when the corpus is mostly clean.

Phase 1 therefore asked:

> Can RAG poisoning still work when the corpus is mostly clean and the poison rate is low?

The short answer from this phase is: **yes, but the attack depends heavily on retrieval exposure, query targeting, model behavior, and the exact poison-document design.**

---

## ✅ Phase 1: What Was Done

### 1. Paper Reading and Project Framing 📚

- Read the PoisonedRAG paper and codebase.
- Reviewed the prior poster-paper motivation around RAG knowledge corruption.
- Reframed the project around a more realistic question: low poison rates in mostly clean corpora.
- Defined two experimental views:
  - **Candidate-pool poison rate**: poison ratio inside the retrieved/evaluated candidate pool.
  - **Corpus-level true ASR**: poison ratio relative to the full corpus size.

### 2. Reproduction Pipeline Setup 🛠️

- Cloned and adapted the PoisonedRAG pipeline for the local machine.
- Kept the Contriever-based retrieval pipeline as the main reproduction path.
- Added local Ollama model support.
- Added tests for the local experiment and Ollama wrapper logic.
- Preserved runnable scripts under `experiments/`.

### 3. Local LLM Matrix 🤖

The original paper models were not all practical to run locally, so Phase 1 used three local LLMs:

| Model | Role in Phase 1 |
| --- | --- |
| `qwen3.5:9b` | Main local reproduction model |
| `llama3.1:8b` | Additional local baseline |
| `mistral:7b` | Additional local baseline |

### 4. Dataset Matrix 🗂️

Experiments covered three BEIR-style datasets:

| Dataset | Corpus size used in corpus-level experiments | Eval sample |
| --- | ---: | ---: |
| NQ | 2,681,468 | 50 queries |
| HotpotQA | 5,233,329 | 50 queries |
| MS MARCO | 8,841,823 | 50 queries |

Raw BEIR corpora are **not committed** because they are large and can be rebuilt locally. See `data/dataset_manifest.csv`.

### 5. Poison-Document Designs 🧬

Phase 1 compared several poison-document variants:

| Variant | Description |
| --- | --- |
| `original` | Baseline PoisonedRAG-style poisoned document |
| `authority` | Adds authority-style language such as verified/preferred answer wording |
| `instruction-aware` | Adds natural instruction-aware conflict-resolution language |
| `query-first` | Places query-targeted wording earlier to improve retrieval and generation alignment |

### 6. Experiment Matrix 📊

Phase 1 completed three major result matrices:

| Experiment family | Matrix |
| --- | --- |
| Paper-style reproduction | 3 datasets × 3 LLMs |
| Candidate-pool low-rate poisoning | 3 datasets × 3 LLMs × 4 variants × 4 poison rates |
| Corpus-level true ASR | 3 datasets × 3 LLMs × 2 variants × 4 true corpus poison rates |

Candidate-pool poison rates:

`1%`, `3%`, `5%`, `10%`

Corpus-level true poison rates:

`0.0001%`, `0.0005%`, `0.001%`, `0.005%`

---

## 🔍 Key Findings

### Paper-style reproduction remained very strong.

Under the stronger paper-style setting, all three local LLMs showed high ASR across the three datasets.

| Dataset | `llama3.1:8b` ASR | `mistral:7b` ASR | `qwen3.5:9b` ASR |
| --- | ---: | ---: | ---: |
| HotpotQA | 0.98 | 0.99 | 1.00 |
| MS MARCO | 0.91 | 0.94 | 0.92 |
| NQ | 0.97 | 1.00 | 0.98 |

### Low candidate-pool poison rates were enough to cause high ASR.

At `1%` candidate-pool poison rate, many configurations already crossed 50% ASR. At `3%` to `5%`, many runs entered high-ASR territory.

Average ASR by candidate-pool strategy:

| Variant | Average ASR |
| --- | ---: |
| `authority` | 0.779 |
| `instruction-aware` | 0.803 |
| `original` | 0.879 |
| `query-first` | 0.889 |

### Corpus-level true ASR showed that tiny global poison rates can still matter.

Even at `0.0001%` corpus poison rate, query-targeted poison documents could still enter top-k retrieval and produce high ASR in some configurations.

Average ASR by corpus-level strategy:

| Variant | Average ASR |
| --- | ---: |
| `original` | 0.682 |
| `query-first` | 0.416 |

### The improved poison document was useful, but not universally.

The `query-first` design helped most in the candidate-pool setting, but `original` was stronger in the corpus-level true-ASR setting. This suggests that a better attack should optimize both:

- retrieval similarity, and
- generation-side credibility.

### Model behavior mattered a lot.

In corpus-level experiments, `mistral:7b` was much more vulnerable than the other two local models.

| Model | Average corpus-level ASR |
| --- | ---: |
| `llama3.1:8b` | 0.331 |
| `mistral:7b` | 0.976 |
| `qwen3.5:9b` | 0.340 |

---

## 🧭 Repository Map

```text
.
├── README.md
├── LICENSE
├── data/
│   └── dataset_manifest.csv
├── experiments/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── model_configs/
│   └── *.py
├── results/
│   ├── figures/
│   ├── tables/
│   └── raw_summaries/
└── reports/
    ├── phase1_full_report_zh.pdf
    └── phase1_results_analysis_zh.md
```

### Result Entry Points

| File | Purpose |
| --- | --- |
| `results/tables/paper_style_full_matrix_summary.csv` | Paper-style 3 dataset × 3 model reproduction summary |
| `results/tables/candidate_pool_full_matrix_summary.csv` | Candidate-pool low-rate poisoning full matrix |
| `results/tables/corpus_level_full_matrix_summary.csv` | Corpus-level true ASR full matrix |
| `results/figures/candidate_fullmatrix_asr_grid.png` | Candidate-pool ASR comparison grid |
| `results/figures/corpus_fullmatrix_asr_grid.png` | Corpus-level ASR comparison grid |
| `reports/phase1_full_report_zh.pdf` | Full Phase 1 report |

---

## 🚀 Quick Start

Clone the repository:

```bash
git clone https://github.com/isclin123/PoisonedRAG-Reproduction-Extension.git
cd PoisonedRAG-Reproduction-Extension
```

Run the lightweight tests:

```bash
cd experiments
python -m unittest tests.test_local_poison_rate_experiment tests.test_ollama_model
```

To rebuild full experiments, first prepare the required BEIR datasets locally. Raw corpora are intentionally excluded from GitHub because they are large.

---

## ⚖️ License and Attribution

This repository is a **mixed-license research artifact**, not a single-license codebase.

- PoisonedRAG-derived code is under the MIT License from the original PoisonedRAG project.
- Contriever-derived code under `experiments/src/contriever_src/` follows the Contriever license, Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- Phase 1 experiment outputs and reports are provided for academic and research use and do not relicense third-party components.

Please read:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

The practical takeaway is simple: this repository is intended for academic reading, reproduction, and non-commercial research. Commercial reuse of the complete repository may be restricted because of the Contriever-derived components.

---

## 📦 Data Policy

This public repository includes:

- experiment code,
- configuration files,
- result tables,
- generated figures,
- compact raw summaries,
- Phase 1 report artifacts.

This public repository does **not** include:

- full raw BEIR corpora,
- local model weights,
- local cache folders,
- temporary run files.

The raw corpora can be rebuilt with the dataset-preparation workflow in `experiments/prepare_dataset.py` and standard BEIR-compatible tooling.

---

## 🧩 Phase Log Template for Future Work

Use this same template for later project stages so each phase stays easy to separate.

```markdown
## Phase N: Short Phase Title

### Goal
- What this phase is trying to answer.

### Scope
- Datasets:
- Models:
- Attack variants:
- Defense variants:
- Metrics:

### What Was Done
- Bullet 1
- Bullet 2
- Bullet 3

### Main Results
| Result | Evidence |
| --- | --- |
| Finding 1 | Table or figure path |

### Key Takeaways
- Takeaway 1
- Takeaway 2

### Limitations
- Limitation 1
- Limitation 2

### Next Phase
- Next task 1
- Next task 2
```

---

## 🔮 Suggested Phase 2 Directions

- Scale from 50-query samples to larger query sets.
- Add multiple random seeds.
- Evaluate other retrievers such as BGE, E5, hybrid retrieval, or rerankers.
- Test defense ideas such as source filtering, conflict detection, clustering, and evidence-consistency checks.
- Improve poison documents by jointly optimizing retrieval rank and generation-side credibility.

---

## 🙌 Acknowledgment

This work builds on the PoisonedRAG research direction and uses a local reproduction setup to study low-rate poisoning under mostly clean-corpus assumptions.

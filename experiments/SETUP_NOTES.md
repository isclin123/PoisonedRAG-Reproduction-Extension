# PoisonedRAG Local Setup Notes

Path note after project reorganization: this runnable workspace is now under `02_reproduction_code_data/PoisonedRAG_runnable_workspace/`. Low-rate experiment outputs were moved to `03_low_rate_experiments/local_ollama_experiment_runs/`.

This repo was cloned from `https://github.com/sleeepeer/PoisonedRAG` on 2026-07-01.

## Environment

The local Conda environment is named `PoisonedRAG` and uses Python 3.10.20.

```powershell
conda create -y -n PoisonedRAG python=3.10
conda activate PoisonedRAG
python -m pip install -r requirements-poisonedrag-windows-cu117.txt
```

The upstream README was written for an older dependency stack. On this Windows CUDA machine, installing unpinned latest packages upgraded PyTorch to a CPU-only `2.x` build. The pinned requirements file keeps the setup on the paper-compatible CUDA 11.7 / PyTorch 1.13 path.

## Local Data And Models

The BEIR datasets have been downloaded and extracted under `datasets/`:

- `nq`
- `msmarco`
- `hotpotqa`

The `facebook/contriever` checkpoint has also been downloaded once through Hugging Face cache and loaded successfully on `cuda:0`.

The PoisonedRAG paper PDF is saved outside the cloned repo at:

```text
..\papers\PoisonedRAG_2402.07867.pdf
```

The previous poster paper from the email was not available in this project folder yet.

## Local Ollama Reproduction

This workspace now includes a local Ollama model provider so the reproduction can run without OpenAI, PaLM, or other hosted API keys:

- Provider implementation: `src/models/Ollama.py`
- Qwen 3.5 9B config: `model_configs/ollama_qwen3.5_config.json`
- Qwen 2.5 1.5B config: `model_configs/ollama_qwen2.5_1.5b_config.json`

The `qwen3.5:9b` config sets `think=false`. Without that option, Ollama can return content in the `thinking` field while leaving the final `response` empty for short QA prompts.

Observed local runs:

- Official-style small reproduction: `results/query_results/local_ollama/nq-ollama-qwen35-M5-paperstyle.json`
- Low poison-rate exploration: `experiments/local_ollama_qwen25_1_5b_nq10`
- Chinese reproduction report: `reports/PoisonedRAG_本机复现报告.md`

## Verification Commands

```powershell
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from src.utils import load_json, load_models; from src.models import create_model; print('repo imports ok')"
python main.py --help
python -c "from src.utils import load_beir_datasets; c,q,r=load_beir_datasets('nq','test'); print(len(c), len(q), len(r))"
python -c "from src.utils import load_models; model,c_model,tokenizer,get_emb=load_models('contriever'); model.to('cuda'); print('contriever loaded', tokenizer.__class__.__name__, next(model.parameters()).device)"
python -m unittest tests.test_local_poison_rate_experiment tests.test_ollama_model
```

Observed verification:

- `pip check`: no broken requirements.
- PyTorch: `1.13.0+cu117`, CUDA `11.7`, CUDA available.
- GPU: NVIDIA GeForce RTX 3060 Ti.
- `nq` load: `2681468` corpus docs, `3452` queries, `3452` qrels.
- Contriever load: `BertTokenizerFast`, model on `cuda:0`.

## Running Experiments On Windows

Do not use `python run.py` directly in PowerShell. The upstream `run.py` builds a Unix shell command with `nohup`, `python3`, output redirection, and background `&`.

Use `main.py` directly from an activated `PoisonedRAG` environment, for example:

```powershell
python main.py --eval_model_code contriever --eval_dataset nq --split test --query_results_dir main --model_name gpt3.5 --top_k 5 --use_truth False --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 1 --M 1 --seed 12 --name smoke-gpt35-nq
```

Before running LLM experiments, add API keys to the relevant file in `model_configs/`. Do not commit real API keys.

For local-only runs, use the Ollama configs instead:

```powershell
python main.py --eval_model_code contriever --eval_dataset nq --split test --query_results_dir local_ollama --model_config_path model_configs/ollama_qwen3.5_config.json --model_name ollama_qwen3.5 --top_k 5 --use_truth False --gpu_id 0 --attack_method LM_targeted --adv_per_query 5 --score_function dot --repeat_times 1 --M 5 --seed 12 --name nq-ollama-qwen35-M5-paperstyle
python local_poison_rate_experiment.py --dataset nq --sample_count 10 --llm_config model_configs/ollama_qwen2.5_1.5b_config.json --max_context_chars 700 --output_dir ..\..\03_low_rate_experiments\local_ollama_experiment_runs\local_ollama_qwen25_1_5b_nq10
```

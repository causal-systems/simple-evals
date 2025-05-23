import argparse
import json
import os
from collections import defaultdict

import dotenv
import pandas as pd

from . import common
from .browsecomp_eval import BrowseCompEval
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .simpleqa_eval import SimpleQAEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler
from .sampler.claude_sampler import (
    ClaudeCompletionSampler,
    CLAUDE_SYSTEM_MESSAGE_LMSYS,
)
from .sampler.grader_ablation_sampler import GraderAblationSampler

dotenv.load_dotenv()


def build_models():
    """Return the dict of model‐name → sampler."""
    return {
        # ── Reasoning models ───────────────────────────────────────────────
        "o3": ResponsesSampler(model="o3-2025-04-16", reasoning_model=True),
        "o3_high": ResponsesSampler(
            model="o3-2025-04-16", reasoning_model=True, reasoning_effort="high"
        ),
        "o3_low": ResponsesSampler(
            model="o3-2025-04-16", reasoning_model=True, reasoning_effort="low"
        ),
        # Default == medium
        "o4-mini": ResponsesSampler(model="o4-mini-2025-04-16", reasoning_model=True),
        "o4-mini_high": ResponsesSampler(
            model="o4-mini-2025-04-16", reasoning_model=True, reasoning_effort="high"
        ),
        "o4-mini_low": ResponsesSampler(
            model="o4-mini-2025-04-16", reasoning_model=True, reasoning_effort="low"
        ),
        # ── Omega ‑ 1 series ───────────────────────────────────────────────
        "o1": OChatCompletionSampler(model="o1"),
        "o1-preview": OChatCompletionSampler(model="o1-preview"),
        "o1-mini": OChatCompletionSampler(model="o1-mini"),
        # Default == medium
        "o3-mini": OChatCompletionSampler(model="o3-mini"),
        "o3-mini_high": OChatCompletionSampler(model="o3-mini", reasoning_effort="high"),
        "o3-mini_low": OChatCompletionSampler(model="o3-mini", reasoning_effort="low"),
        # ── GPT‑4.1 ────────────────────────────────────────────────────────
        "gpt-4.1": ChatCompletionSampler(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-mini": ChatCompletionSampler(
            model="gpt-4.1-mini-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-nano": ChatCompletionSampler(
            model="gpt-4.1-nano-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # ── GPT‑4o family ──────────────────────────────────────────────────
        "gpt-4o": ChatCompletionSampler(
            model="gpt-4o", system_message=OPENAI_SYSTEM_MESSAGE_API, max_tokens=2048
        ),
        "gpt-4o-mini": ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # ── GPT‑4.5 preview ────────────────────────────────────────────────
        "gpt-4.5-preview": ChatCompletionSampler(
            model="gpt-4.5-preview-2025-02-27",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # ── GPT‑4‑turbo ────────────────────────────────────────────────────
        "gpt-4-turbo-2024-04-09": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09", system_message=OPENAI_SYSTEM_MESSAGE_API
        ),
        # ── ChatGPT endpoints ──────────────────────────────────────────────
        "chatgpt-4o-latest": ChatCompletionSampler(
            model="chatgpt-4o-latest",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        ),
        "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        ),
        # ── Claude ─────────────────────────────────────────────────────────
        "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
            model="claude-3-opus-20240229", system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS
        ),
        # ── Ablation ──────────────────────────────────────────────────────
        "grader_ablation_3b": GraderAblationSampler(),
    }


def get_evals(eval_name: str, debug_mode: bool, args_examples: int | None):
    """Factory returning an evaluation object for *eval_name*."""
    num_examples = args_examples if args_examples is not None else (5 if debug_mode else None)

    match eval_name:
        case "mmlu":
            return MMLUEval(num_examples=1 if debug_mode else num_examples)
        case "math":
            return MathEval(
                equality_checker=ChatCompletionSampler(model="gpt-4-turbo-preview"),
                num_examples=num_examples,
                n_repeats=1 if debug_mode else 1,
            )
        case "gpqa":
            return GPQAEval(n_repeats=1 if debug_mode else 10, num_examples=num_examples)
        case "mgsm":
            return MGSMEval(num_examples_per_lang=10 if debug_mode else 250)
        case "drop":
            return DropEval(num_examples=10 if debug_mode else num_examples, train_samples_per_prompt=3)
        case "humaneval":
            return HumanEval(num_examples=10 if debug_mode else num_examples)
        case "simpleqa":
            return SimpleQAEval(
                grader_model=ChatCompletionSampler(model="gpt-4o"),
                num_examples=10 if debug_mode else num_examples,
            )
        case "browsecomp":
            return BrowseCompEval(
                grader_model=ChatCompletionSampler(model="gpt-4o"),
                num_examples=10 if debug_mode else num_examples,
            )
        case _:
            raise ValueError(f"Unrecognized eval type: {eval_name}")


def flatten_metrics(nested: dict[str, dict[str, dict]]) -> pd.DataFrame:
    """Flatten nested metrics into (model → columns) DataFrame."""
    rows: list[dict] = []
    for model_name, evals in nested.items():
        row: dict[str, float | str] = {"model_name": model_name}
        for eval_name, metrics in evals.items():
            for k, v in metrics.items():
                row[f"{eval_name}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows).set_index("model_name")


def main():
    parser = argparse.ArgumentParser(description="Run sampling and evaluations.")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Evaluate a single model")
    parser.add_argument("--debug", action="store_true", help="Debug / quick mode")
    parser.add_argument("--examples", type=int, help="Number of examples to override defaults")
    args = parser.parse_args()

    # ── Models ────────────────────────────────────────────────────────────
    models = build_models()
    if args.list_models:
        print("Available models:\n" + "\n".join(f" - {m}" for m in models))
        return
    if args.model:  # keep only requested model
        if args.model not in models:
            raise SystemExit(f"Model '{args.model}' not found.")
        models = {args.model: models[args.model]}

    # ── Evaluations to run (env var: EVALS=gpqa;math;… ) ──────────────────
    eval_names = [e for e in os.getenv("EVALS", "").split(";") if e]
    evals = {name: get_evals(name, args.debug, args.examples) for name in eval_names}
    print("Evaluations:", list(evals))

    debug_suffix = "_DEBUG" if args.debug else ""
    mergekey2path: dict[str, str] = {}

    # ── Execute ───────────────────────────────────────────────────────────
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)

            # HTML report ────────────────────────────────────────────────
            report_path = f"/tmp/{eval_name}_{model_name}{debug_suffix}.html"
            with open(report_path, "w") as fh:
                fh.write(common.make_report(result))
            print(f"Report → {report_path}")

            # Raw metrics ────────────────────────────────────────────────
            metrics = result.metrics | {"score": result.score}
            if isinstance(sampler, GraderAblationSampler):
                metrics |= {
                    "mean_response_length": sampler.num_tokens / max(1, sampler.num_calls)
                }
                sampler.num_calls = sampler.num_tokens = 0  # reset counters

            json_path = f"/tmp/{os.getenv('CUDA_VISIBLE_DEVICES', '')}{eval_name}_{model_name}{debug_suffix}.json"
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics → {json_path}")
            mergekey2path[f"{eval_name}_{model_name}"] = json_path

    # ── Merge results ────────────────────────────────────────────────────
    combined: defaultdict[str, dict] = defaultdict(dict)
    for key, path in mergekey2path.items():
        eval_name, model_name = key.split("_", 1)
        try:
            with open(path) as f:
                combined[model_name][eval_name] = json.load(f)
        except Exception as exc:
            print(f"⚠️  Skipping {path}: {exc}")

    # ── Persist nested JSON ──────────────────────────────────────────────
    json_output = os.getenv("OUTPUT_PATH", "/root/simple_evals_output.json")
    with open(json_output, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Nested JSON → {json_output}")

    # ── CSV / DataFrame ──────────────────────────────────────────────────
    df = flatten_metrics(combined)
    csv_output = os.getenv("CSV_OUTPUT_PATH", "/root/simple_evals_output.csv")
    df.to_csv(csv_output)
    print("\nAggregated scores (markdown):\n" + df.to_markdown())
    print(f"CSV → {csv_output}")

    return df  # convenient for REPL


if __name__ == "__main__":
    main()
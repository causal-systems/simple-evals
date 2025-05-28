"""End-to-end evaluation pipeline
================================
Downloads a dataset from S3, spins up a local inference server per checkpoint on
an available GPU and port, collects answers, grades them, verifies them, and
saves the consolidated results into the on-disk cache.

The implementation is *deliberately* kept functional and free of heavy OOP so
that individual pieces can be inspected and modified easily.  Concurrency is
handled with a simple worker pool driven by a semaphore – each worker gets a
single GPU and processes checkpoints sequentially.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Literal, Sequence

import boto3
import pandas as pd
import requests
from logging import getLogger
import vllm
from transformers import PreTrainedTokenizer, AutoTokenizer

def create_chat_tokens_promptonly(
  system_prompt: str, 
  user_prompt: str, 
  tokenizer: PreTrainedTokenizer,
):
    """
    Creates a prompt template by combining system, user, and assistant prompts with appropriate formatting.
    Args:
    system_prompt (str): The system instructions/context
    user_prompt (str): The user's input/question
    assistant_answer (str): The assistant's response
    tokenizer (Tokenizer): Tokenizer instance for processing the prompts
    Returns:
    str: The formatted prompt template combining all inputs
    """

    tokens = []
    tokens.append(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    tokens.append(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    tokens.extend(tokenizer.encode('system'))
    tokens.extend(tokenizer.encode('\n'))
    tokens.extend(tokenizer.encode(system_prompt))
    tokens.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    tokens.extend(tokenizer.encode('\n'))

    tokens.append(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    tokens.extend(tokenizer.encode('user'))
    tokens.extend(tokenizer.encode('\n'))
    tokens.extend(tokenizer.encode(user_prompt))
    tokens.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    tokens.extend(tokenizer.encode('\n'))

    tokens.append(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    tokens.extend(tokenizer.encode('assistant'))
    tokens.extend(tokenizer.encode('\n'))

    return tokens

###############################################################################
# Constants & system prompts – keep as-is unless you have a good reason.
###############################################################################
REASONING_SYSTEM_PROMPT = """A conversation between User and Assistant. 
      
The user asks a question, and the Assistant solves it.

The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. 

The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.
"""

REASONING_SUMMARY_SYSTEM_PROMPT = """A conversation between User and Assistant. 
      
The user asks a question, and the Assistant solves it.

The assistant first thinks about the reasoning process in the mind, writes a summary of the reasoning process, and then provides the user with the answer. 
The summary must contain all of the key reasoning steps that led to the answer.
The reasoning process, summary, and answer are enclosed within <think> </think>, <summary> </summary>, and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<summary> summary here </summary>
<answer> answer here </answer>."""

###############################################################################
# Helpers
###############################################################################
logger = getLogger(__name__)
logger.setLevel("INFO")

def _parse_s3_uri(uri: str) -> tuple[str, str]:
    match = re.match(r"s3://([^/]+)/(.+)", uri)
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return match.group(1), match.group(2)


BASE_CACHE_DIR = Path(__file__).parent / ".cache"
BASE_CACHE_DIR.mkdir(exist_ok=True)

def download_from_s3_cached(s3_uri: str, cache_dir: Path | str | None = None) -> Path:
    cache_dir = Path(cache_dir or BASE_CACHE_DIR / "s3").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    bucket, key = _parse_s3_uri(s3_uri)
    local_path = cache_dir / Path(key).name
    if local_path.exists():
        return local_path
    logger.info("Downloading %s → %s", s3_uri, local_path)
    boto3.client("s3").download_file(bucket, key, str(local_path))
    return local_path

def main(
    checkpoint_dir: str,
    dataset_s3_uri: str,
    model_type: Literal["reasoning", "reasoning-summary"],
    stage: Literal["model", "grader", "verifier"]
) -> None:

    print(f"running {checkpoint_dir=}, {dataset_s3_uri=}, {model_type=}, {stage=}")

    if not os.path.exists(Path(checkpoint_dir) / "tokenizer.json"):
        req = requests.get("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer.json")
        with open(Path(checkpoint_dir) / "tokenizer.json", "wb") as f:
            f.write(req.content)
    
    if not os.path.exists(Path(checkpoint_dir) / "tokenizer_config.json"):
        req = requests.get("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json")
        with open(Path(checkpoint_dir) / "tokenizer_config.json", "wb") as f:
            f.write(req.content)

    if stage == "model" and not os.path.exists("outputs/pipeline/" + str(checkpoint_dir).replace("/", "_") + "_model" + ".parquet"):
        ds_path = download_from_s3_cached(dataset_s3_uri)
        df = pd.read_parquet(ds_path, engine="pyarrow")[:20]
        questions = [msg[1]["content"] for msg in df["prompt"]]
        reference_answers = list(df["reward_model"].apply(lambda x: x["ground_truth"]))
        sources = list(df["data_source"])
        llm = vllm.LLM(model=checkpoint_dir)

        system_prompt = REASONING_SYSTEM_PROMPT if model_type == "reasoning" else REASONING_SUMMARY_SYSTEM_PROMPT
        dialogs = [
            [
                dict(role="system", content=system_prompt),
                dict(role="user", content=q)
            ]
            for q in questions
        ]
        outputs = llm.chat(dialogs, sampling_params=vllm.SamplingParams(temperature=0, max_tokens=8192))
        model_outputs = [x.outputs[0].text for x in outputs]
        model_answers = []
        for model_output in model_outputs:
            m = re.search(r"<answer>(.*)<\/answer>", model_output, re.S)
            if m:
                model_answers.append(m.group(1))
            else:
                model_answers.append("")
        df = pd.DataFrame(data={
            "question": questions,
            "model_output": model_outputs,
            "model_answer": model_answers,
            "reference_answer": reference_answers,
            "source": sources
        })
        os.makedirs("outputs/pipeline", exist_ok=True)
        df.to_parquet("outputs/pipeline/" + str(checkpoint_dir).replace("/", "_") + "_model" + ".parquet")

    elif stage == "grader" and not os.path.exists("outputs/pipeline/" + str(checkpoint_dir).replace("/", "_") + "_grader" + ".parquet"):
        df = pd.read_parquet("outputs/pipeline/" + str(checkpoint_dir).replace("/", "_") + "_model" + ".parquet", engine="pyarrow")
        questions = df["question"]
        model_answers = df["model_answer"]
        reference_answers = df["reference_answer"]
        sources = df["source"]
        llm = vllm.LLM(model="GeneralReasoning/beaglefinal")
        tokenizer = AutoTokenizer.from_pretrained("GeneralReasoning/beaglefinal")
        system_prompt = (
            "You are an inference-time checker that compares a reference answer with a model answer, "
            "and returns a yes/no answer corresponding to whether the model answer is correct."
        )
        # dialogs = [
        #     [
        #         dict(role="system", content=system_prompt),
        #         dict(role="user", content=(
        #             f"Question: {question}\n\n"
        #             f"Reference Answer: {reference_answer}\n\n"
        #             f"Model Answer: {model_answer}"
        #         ))
        #     ]
        #     for question, reference_answer, model_answer in zip(questions, reference_answers, model_answers)
        # ]
        # outputs = llm.chat(dialogs, sampling_params=vllm.SamplingParams(temperature=0, max_tokens=8192))

        dialog_tokens = [
            {'prompt_token_ids': create_chat_tokens_promptonly(system_prompt, (
                f"Question: {question}\n\n"
                f"Reference Answer: {reference_answer}\n\n"
                f"Model Answer: {model_answer}"
            ), tokenizer)}
            for question, reference_answer, model_answer in zip(questions, reference_answers, model_answers)
        ]
        # llm_engine.generate(
        #     {"prompt_token_ids": prompt_token_ids}, 
        #     sampling_params,
        #     request_id
        # )
        outputs = llm.generate(dialog_tokens, sampling_params=vllm.SamplingParams(temperature=0, max_tokens=8192))

        grader_outputs = [x.outputs[0].text for x in outputs]
        grader_scores: list[float] = []
        for x in grader_outputs:
            if '<|decision_start|>' in x:
                decision_text = x.split('<|decision_start|>')[1].split('<|decision_end|>')[0].strip()
                score = 1.0 if decision_text == 'Yes' else 0.0
            else:
                score = 0.0
            grader_scores.append(score)
        df["grader_score"] = grader_scores
        df["grader_output"] = grader_outputs
        os.makedirs("outputs/pipeline", exist_ok=True)
        df.to_parquet("outputs/pipeline/" + str(checkpoint_dir).replace("/", "_") + "_grader" + ".parquet")
        
    elif stage == "verifier" and not os.path.exists("outputs/pipeline/" + str(checkpoint_dir).replace("/", "_") + "_final" + ".parquet"):
        df = pd.read_parquet("outputs/pipeline/" + str(checkpoint_dir).replace("/", "_") + "_grader" + ".parquet", engine="pyarrow")
        questions = df["question"]
        reference_answers = df["reference_answer"]
        sources = df["source"]
        model_outputs = df["model_output"]
        llm = vllm.LLM(model="causal-systems/holmes")
        system_prompt = """You are an inference-time verifier that checks the correctness of a model answer, and returns a yes/no answer coresponding to correctness

The verifier process and final yes/no answer are enclosed within <|verification_start|> <|verification_end|> and <|decision_start|> <|decision_end|> tags respectively.
i.e <|verification_start|> verification process here <|verification_end|> <|decision_start|> answer here <|decision_end|>.

When verifying the model's answer, you should think step by step within the <|verification_start|> <|verification_end|> tags.

The final answer should be enclosed within <|decision_start|> <|decision_end|> tags."""
        verifier_inputs = []
        for model_output in model_outputs:
            m = re.search(r"<answer>(.*)<\/answer>", model_output, re.S)
            if m:
                answer_text = m.group(1)
            else:
                answer_text = ""

            m = re.search(r"<summary>(.*)<\/summary>", model_output, re.S)
            if m:
                summary_text = m.group(1)
            else:
                summary_text = ""
            
            if model_type == "reasoning":
                verifier_input = answer_text
            else:
                verifier_input = f"{summary_text}\n\n{answer_text}"

            verifier_inputs.append(verifier_input)

        dialogs = [
            [
                dict(role="system", content=system_prompt),
                dict(role="user", content=f"question: {question}\n\nmodel answer: {verifier_input}\n\n")
            ]
            for question, verifier_input in zip(questions, verifier_inputs)
        ]
        outputs = llm.chat(dialogs, sampling_params=vllm.SamplingParams(temperature=0, max_tokens=8192))
        verifier_outputs = [x.outputs[0].text for x in outputs]
        verifier_scores: list[float] = []
        for x in verifier_outputs:
            if '<|decision_start|>' in x:
                decision_text = x.split('<|decision_start|>')[1].split('<|decision_end|>')[0].strip()
                score = 1 if decision_text == 'Yes' else 0
            else:
                score = 0
            verifier_scores.append(score)

        df["verifier_scores"] = verifier_scores
        df["verifier_output"] = verifier_outputs
        os.makedirs("outputs/pipeline", exist_ok=True)
        df.to_parquet("outputs/pipeline/" + str(checkpoint_dir).replace("/", "_") + "_final" + ".parquet")

if __name__ == "__main__":
    import fire
    fire.Fire(main)

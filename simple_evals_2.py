import gc
import os
from pathlib import Path
import re
from typing import Callable, Literal, cast
from datasets import load_dataset
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import pandas as pd
from transformers import PreTrainedTokenizer

def run_supergqpa(
    model: LLM,
    prefix: list[dict],
    parse_answer: Callable[[str], str],
    sampling_params: SamplingParams = SamplingParams(temperature=0, max_tokens=8192, skip_special_tokens=False)
) -> pd.DataFrame:
    ds = load_dataset("m-a-p/SuperGPQA", split="train")
    df = cast(pd.DataFrame, ds.to_pandas())
    # adapted from https://github.com/SuperGPQA/SuperGPQA/blob/e31e8bf3e7089aef808be5ddecafd4538dee6e41/infer/models/anthropic_api.py#L63
    dialogs: list[list[dict]] = []

    for row in df.to_dict("records"):
        user_message = f"""Answer the following multiple choice question. There is only one correct answer.

Question: {row['question']}
Answers:"""
        answer_choices: list[str] = row["options"]
        for i, x in enumerate(answer_choices):
            user_message += "\n" + chr(65 + i) + ". " + x

        dialogs.append([
            *prefix,
            {
                "role": "user",
                "content": user_message
            }
        ])

    outputs = model.chat(dialogs, sampling_params=sampling_params)
    model_outputs = [x.outputs[0].text for x in outputs]
    model_answers = [parse_answer(x) for x in model_outputs]

    df["model_output"] = model_outputs
    df["model_answer"] = model_answers

    return df

def get_prefix(model_type: str) -> list[dict]:
    if model_type == "gr-reasoning":
        return [
            {
                "role": "system",
                "content": """A conversation between User and Assistant. 
      
The user asks a question, and the Assistant solves it.

The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. 

The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.
"""
            }
        ]
    elif model_type == "gr-reasoning-summary":
        return [
            {
                "role": "system",
                "content": """A conversation between User and Assistant. 
      
The user asks a question, and the Assistant solves it.

The assistant first thinks about the reasoning process in the mind, writes a summary of the reasoning process, and then provides the user with the answer. 
The summary must contain all of the key reasoning steps that led to the answer.
The reasoning process, summary, and answer are enclosed within <think> </think>, <summary> </summary>, and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<summary> summary here </summary>
<answer> answer here </answer>."""
            }
        ]
    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")

def parse_html_answer(s: str) -> str:
    m = re.search(r"<answer>(.*)<\/answer>", s, re.S)
    if m is None:
        return ""
    return m.group(1)

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


def grade_gr_beagle(grader: LLM, df: pd.DataFrame) -> None:
    system_prompt = (
        "You are an inference-time checker that compares a reference answer with a model answer, "
        "and returns a yes/no answer corresponding to whether the model answer is correct."
    )

    dialog_tokens = [
        {'prompt_token_ids': create_chat_tokens_promptonly(system_prompt, (
            f"Question: {question}\n\n"
            f"Reference Answer: {reference_answer}\n\n"
            f"Model Answer: {model_answer}"
        ), tokenizer)}
        for question, reference_answer, model_answer in zip(questions, reference_answers, model_answers)
    ]
    outputs = grader.generate(dialog_tokens, sampling_params=vllm.SamplingParams(temperature=0, max_tokens=8192, skip_special_tokens=False))

    grader_outputs = [x.outputs[0].text for x in outputs]
    grader_scores: list[float] = []
    for x in grader_outputs:
        if '<|decision_start|>' in x:
            decision_text = x.split('<|decision_start|>')[1].split('<|decision_end|>')[0].strip()
            score = 1.0 if decision_text == 'Yes' else 0.0
        else:
            score = 0.0
        grader_scores.append(score)

    df["grader_output"] = grader_outputs
    df["grader_score"] = grader_scores

def run_evals(
    model: str,
    model_type: str,
    evals: tuple[str]
) -> list[pd.DataFrame]:
    
    llm = LLM(model)
    outputs: list[pd.DataFrame] = []
    prefix = get_prefix(model_type)

    for e in evals:
        if e == "supergpqa":
            outputs.append(run_supergqpa(llm, prefix, parse_html_answer))
        else:
            raise NotImplementedError(f"Unsupported eval: {e}")
        
    destroy_model_parallel()
    del llm.llm_engine.driver_worker
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()

    llm = LLM("GeneralReasoning/beaglefinal")
    for output in outputs:
        grade_gr_beagle(llm, output)

    return outputs

def main(
    model: str,
    model_type: str,
    evals: tuple[str],
    output_dir: str,
    output_format: Literal["jsonl", "parquet"] = "parquet"
) -> None:
    
    outputs = run_evals(model, model_type, evals)
    d = Path(output_dir)
    os.makedirs(d, exist_ok=True)

    
    results: list[dict] = []

    for eval_name, output in zip(evals, outputs):
        if output_format == "jsonl":
            output.to_json(d / f"{eval_name}.jsonl", lines=True)
        elif output_format == "parquet":
            output.to_parquet(d / f"{eval_name}.parquet", engine="pyarrow")
        else:
            raise NotImplementedError(f"Unsupported output format: {output_format}")
        
        score = sum(output["grader_score"]) / len(output["grader_score"])
        results.append({ "name": eval_name, "score": score })

    results_df = pd.DataFrame(results)
    print(results_df.to_markdown())

    if output_format == "jsonl":
        results_df.to_json(d / f"results.jsonl", lines=True)
    elif output_format == "parquet":
        results_df.to_parquet(d / f"results.parquet", engine="pyarrow")
    else:
        raise NotImplementedError(f"Unsupported output format: {output_format}")
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)
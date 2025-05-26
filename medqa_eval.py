import random
import re
from typing import Literal

import pandas
from datasets import load_dataset

from . import common
from .common import ANSWER_PATTERN, HTML_JINJA, grade
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

def doc_to_text(doc: dict) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    return f"Please answer with the letter corresponding to the correct answer.\n\nQuestion: {doc['sent1']}\n{answers}Answer:"


def doc_to_target(doc: dict) -> int:
    return doc["label"]

class MedQAEval(Eval):
    def __init__(
        self,
        num_examples: int | None = None
    ):
        ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf")
        df = ds["test"].to_pandas()
        self.examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            rng = random.Random(0)
            self.examples = rng.sample(self.examples, num_examples)

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=doc_to_text(row), role="user")
            ]
            response_text = sampler(prompt_messages)
            # score = float(check_equality(self.equality_checker, row["Answer"], extracted_answer))
            correct_answer = ["A", "B", "C", "D"][row["label"]]
            score = grade(prompt_messages[0]["content"], correct_answer, str(response_text))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=response_text,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)

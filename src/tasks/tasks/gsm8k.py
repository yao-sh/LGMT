"""
"Training Verifiers to Solve Math Word Problems"
https://arxiv.org/abs/2110.14168

@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

NOTE: See the official implementation of the task: 
    https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py
for how to make use of the dataset's calculator annotations in your language
model's sample/generation function.
"""

import json
import re
from best_download import download_file
from pathlib import Path
from tasks.base import Task, rf
from tasks.metrics import mean

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


class GradeSchoolMath8K(Task):
    VERSION = 0
    DATASET_PATH = Path('data/gsm8k')

    def download(self):
        if self.DATASET_PATH.exists():
            return
        Path.mkdir(self.DATASET_PATH, parents=True)
        base_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
        splits = [
            {"name": "train", "checksum": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"},
            {"name": "test", "checksum": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"},
        ]
        for split in splits:
            file = self.DATASET_PATH / f"{split['name']}.jsonl"
            download_file(f"{base_url}/{split['name']}.jsonl", str(file), split["checksum"])

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _load_docs(self, file):
        return (json.loads(line) for line in open(file).read().splitlines())

    def training_docs(self):
        return self._load_docs(self.DATASET_PATH / "train.jsonl")

    def validation_docs(self):
        raise NotImplementedError

    def test_docs(self):
        return self._load_docs(self.DATASET_PATH / "test.jsonl")

    def doc_to_text(self, doc):
        return "Question: " + doc['question'] + '\nAnswer:'

    def doc_to_target(self, doc):
        return " " + doc['answer']

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # NOTE: The paper implements "verifiers" that assign a score to multiple 
        # solutions and output the highest ranked solution.
        completion = rf.greedy_until(ctx, ['\n'])
        return completion 

    def _extract_answer(self, completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS

    def _is_correct(self, completion, answer):
        gold = self._extract_answer(answer)
        assert gold != INVALID_ANS, "No ground truth answer found in the document."
        return self._extract_answer(completion) == gold 

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        completion = results[0]
        answer = doc["answer"]
        return {
            "acc": self._is_correct(completion, answer)
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "acc": True
        }

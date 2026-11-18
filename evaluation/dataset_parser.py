import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CUADAnswer:
    text: str
    answer_start: int

@dataclass
class CUADQuestion:
    id: str
    question: str
    category: str
    is_impossible: bool
    answers: List[CUADAnswer] = field(default_factory=list)

    @property
    def is_answerable(self) -> bool:
        return not self.is_impossible and len(self.answers) > 0

@dataclass
class CUADContract:
    title: str
    context: str
    questions: List[CUADQuestion] = field(default_factory=list)

    @property
    def answerable_questions(self) -> List[CUADQuestion]:
        return [q for q in self.questions if q.is_answerable]

    @property
    def unanswerable_questions(self) -> List[CUADQuestion]:
        return [q for q in self.questions if not q.is_answerable]

def parse_category_from_id(qa_id: str) -> str:
    parts = qa_id.split("__")
    if len(parts) >= 2:
        return parts[-1].strip()
    return qa_id.strip()

def load_cuad_dataset(dataset_path: str) -> List[CUADContract]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CUAD dataset file not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items = data.get("data", [])
    contracts: List[CUADContract] = []

    for item in raw_items:
        title = item.get("title", "").strip()
        paragraphs = item.get("paragraphs", [])
        if not paragraphs:
            continue

        p0 = paragraphs[0]
        context = p0.get("context", "")
        qas = p0.get("qas", [])

        questions: List[CUADQuestion] = []
        for qa in qas:
            qa_id = qa.get("id", "")
            q_text = qa.get("question", "")
            is_impossible = qa.get("is_impossible", False)
            category = parse_category_from_id(qa_id)

            answers: List[CUADAnswer] = []
            for ans in qa.get("answers", []):
                ans_text = ans.get("text", "")
                ans_start = ans.get("answer_start", -1)
                if ans_start >= 0 and ans_text:
                    answers.append(CUADAnswer(text=ans_text, answer_start=ans_start))

            questions.append(
                CUADQuestion(
                    id=qa_id,
                    question=q_text,
                    category=category,
                    is_impossible=is_impossible,
                    answers=answers,
                )
            )

        contracts.append(
            CUADContract(
                title=title,
                context=context,
                questions=questions,
            )
        )

    return contracts

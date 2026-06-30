from dataclasses import dataclass
from typing import Callable


@dataclass
class Question:
    id: str
    system_prompt: str
    user_template: str
    validate: Callable[[dict], bool] = lambda d: True


class QuestionRegistry:
    def __init__(self):
        self._questions: dict[str, Question] = {}

    def register(self, question: Question) -> None:
        self._questions[question.id] = question

    def get(self, qid: str) -> Question:
        return self._questions[qid]

    def list_ids(self) -> list[str]:
        return list(self._questions.keys())

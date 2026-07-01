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


import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from tqdm import tqdm


@dataclass
class AnalyzeReport:
    question_id: str
    processed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class BatchAnalyzer:
    store: object
    llm: object
    registry: QuestionRegistry
    model: str = "gemma3:4b"
    max_workers: int = 4
    flush_every: int = 20

    def run(self, question_id: str) -> AnalyzeReport:
        question = self.registry.get(question_id)
        report = AnalyzeReport(question_id=question_id)
        pending_ids = self.store.entries_missing_analysis(question_id, self.model)
        if not pending_ids:
            return report

        df = self.store.entries_to_pandas().set_index("id")
        buffer: list[dict] = []

        def _one(entry_id: str) -> dict:
            text = df.loc[entry_id, "text"]
            user_msg = question.user_template.format(text=text)
            parsed = self.llm.complete_json(
                system=question.system_prompt, user=user_msg
            )
            ok = parsed is not None and question.validate(parsed)
            if ok:
                result_json = json.dumps(parsed)
            elif parsed is not None:
                result_json = json.dumps({"_invalid": parsed})
            else:
                result_json = json.dumps({"_error": "no_json"})
            return {
                "entry_id": entry_id,
                "question_id": question_id,
                "question_text": user_msg,
                "result_json": result_json,
                "parsed_ok": bool(ok),
                "model": self.model,
                "created_at": datetime.now(),
            }

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_one, eid): eid for eid in pending_ids}
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"analyze[{question_id}]",
            ):
                row = fut.result()
                buffer.append(row)
                if row["parsed_ok"]:
                    report.processed += 1
                else:
                    report.failed += 1
                if len(buffer) >= self.flush_every:
                    self.store.add_analyses(buffer)
                    buffer = []

        if buffer:
            self.store.add_analyses(buffer)
        return report

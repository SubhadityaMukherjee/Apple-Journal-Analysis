import pytest

from journal.llm import OllamaLLM


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, model: str, messages: list, format=None):
        self.calls += 1
        return {"message": {"content": self.responses.pop(0)}}


def test_complete_json_returns_parsed_dict():
    client = FakeClient(['{"level":"negative","score":2}'])
    llm = OllamaLLM(model="gemma3:4b", client=client)
    result = llm.complete_json(system="sys", user="usr")
    assert result == {"level": "negative", "score": 2}
    assert client.calls == 1


def test_complete_json_strips_code_fences():
    client = FakeClient(['```json\n{"x": 1}\n```'])
    llm = OllamaLLM(model="m", client=client)
    assert llm.complete_json("s", "u") == {"x": 1}


def test_complete_json_retries_on_invalid_then_succeeds():
    client = FakeClient(["not json", '{"x": 1}'])
    llm = OllamaLLM(model="m", client=client)
    assert llm.complete_json("s", "u") == {"x": 1}
    assert client.calls == 2


def test_complete_json_returns_none_after_max_retries():
    client = FakeClient(["no", "no", "no"])
    llm = OllamaLLM(model="m", client=client, max_retries=3)
    assert llm.complete_json("s", "u") is None
    assert client.calls == 3

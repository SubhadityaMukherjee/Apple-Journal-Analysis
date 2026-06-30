"""Built-in question definitions."""

from ..analyze import QuestionRegistry
from .sentiment import SENTIMENT
from .topics import TOPICS


def default_registry() -> QuestionRegistry:
    reg = QuestionRegistry()
    reg.register(SENTIMENT)
    reg.register(TOPICS)
    return reg

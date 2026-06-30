"""Built-in question definitions."""

from ..analyze import QuestionRegistry
from .sentiment import SENTIMENT


def default_registry() -> QuestionRegistry:
    reg = QuestionRegistry()
    reg.register(SENTIMENT)
    return reg

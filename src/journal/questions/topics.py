from ..analyze import Question

SYSTEM = (
    "You are analyzing a personal journal entry. Identify the most prominent topics. "
    "Respond with ONLY a JSON object, no other text."
)

USER_TEMPLATE = """Identify the top 3 topics in this journal entry.
Topics must be short phrases (1-3 words), lowercase, and concrete (not "life" or "feelings").

Respond with ONLY a JSON object of the form:
{{"topics": ["topic_one", "topic_two", "topic_three"]}}

Journal entry:
{text}"""


def _validate(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    topics = d.get("topics")
    if not isinstance(topics, list):
        return False
    if len(topics) != 3:
        return False
    if not all(isinstance(t, str) and t.strip() for t in topics):
        return False
    return True


TOPICS = Question(
    id="topics",
    system_prompt=SYSTEM,
    user_template=USER_TEMPLATE,
    validate=_validate,
)

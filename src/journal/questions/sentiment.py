from ..analyze import Question

SENTIMENT_LEVELS = {
    "extreme_negative",
    "negative",
    "neutral",
    "positive",
    "extreme_positive",
}

SYSTEM = (
    "You are analyzing a personal journal entry. Classify the overall emotional tone. "
    "Respond with ONLY a JSON object, no other text."
)

USER_TEMPLATE = """Respond with ONLY a JSON object of the form:
{{"level": "<one of: extreme_negative, negative, neutral, positive, extreme_positive>", "score": <integer 1-5>, "confidence": <float 0.0-1.0>}}

Scale:
1 / extreme_negative: anguish, crisis language, overwhelming distress
2 / negative: frustrated, sad, upset, but contained
3 / neutral: factual, no strong affect
4 / positive: happy, grateful, content
5 / extreme_positive: ecstatic, overjoyed, deep fulfillment

Journal entry:
{text}"""


def _validate(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    level = d.get("level")
    score = d.get("score")
    confidence = d.get("confidence")
    if level not in SENTIMENT_LEVELS:
        return False
    if not isinstance(score, int) or not (1 <= score <= 5):
        return False
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        return False
    return True


SENTIMENT = Question(
    id="sentiment",
    system_prompt=SYSTEM,
    user_template=USER_TEMPLATE,
    validate=_validate,
)

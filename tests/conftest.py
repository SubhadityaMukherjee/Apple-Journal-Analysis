from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_html_path() -> Path:
    return FIXTURES / "2020-06-29.html"

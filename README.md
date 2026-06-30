# journal_stata

Local analysis pipeline for Apple Journal HTML exports.

See `docs/superpowers/specs/2026-06-30-journal-analysis-design.md` for the design.

## Setup

```bash
uv sync
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

## Use

Open notebooks in `notebooks/`. The pipeline modules live in `src/journal/`.

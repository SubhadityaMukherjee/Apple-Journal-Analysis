from dataclasses import dataclass

import pandas as pd


@dataclass
class QueryAnswer:
    answer: str
    sources: pd.DataFrame


def answer_question(store, embedder, llm, question: str, k: int = 10) -> QueryAnswer:
    qvec = embedder.embed(question)
    sources = store.search_entries(query_vec=qvec, k=k)
    context = "\n\n---\n\n".join(
        f"[{i}] {row['text']}" for i, (_, row) in enumerate(sources.iterrows())
    )

    system = (
        "You answer questions about a user's personal journal. "
        "Use the provided excerpts. Cite excerpts by their [N] index when relevant."
    )
    user = f"Question: {question}\n\nExcerpts:\n{context}"
    answer = llm.complete(system=system, user=user)
    return QueryAnswer(answer=answer, sources=sources)

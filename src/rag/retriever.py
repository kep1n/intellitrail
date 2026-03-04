from langsmith import traceable
from src.rag.index import load_index


@traceable
def retrieve_gear_context(
    query: str,
    k: int = 5,
) -> str:
    """Search the Pinecone index for passages relevant to the query.

    Returns top-k passages concatenated with separator lines.
    Raises FileNotFoundError (from load_index) if index does not exist.

    Intended query: the verdict + weather conditions summary built inside
    rag_enrichment node in Plan 04.
    """
    vectorstore = load_index()
    docs = vectorstore.similarity_search(query, k=k)

    if not docs:
        return "No relevant passages found in the alpine safety documents."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Passage {i} — {source}, p.{page}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)

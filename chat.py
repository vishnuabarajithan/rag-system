"""
Interactive RAG tester — ingest a document, then ask it questions.

Runs Qdrant fully in-process (no Docker, no server) by setting
QDRANT_LOCAL_PATH before any project module is imported. Data persists
in ./qdrant_local so you can re-run and keep previously ingested docs.

Usage:
    python chat.py path/to/document.pdf
    python chat.py path/to/notes.txt --collection mydocs
    python chat.py            # skip ingest, just query what's already stored
"""
import argparse
import os
from pathlib import Path

# Must be set BEFORE importing config.settings (done transitively below).
os.environ.setdefault("QDRANT_LOCAL_PATH", str(Path(__file__).parent / "qdrant_local"))

from ingestion.pipeline import IngestionPipeline
from agents.rag_agent import rag_graph


def ingest(path: Path, collection: str) -> None:
    print(f"Ingesting {path.name} ...")
    count = IngestionPipeline().ingest_file(
        filename=path.name,
        content=path.read_bytes(),
        collection=collection,
    )
    print(f"  -> stored {count} chunk(s) in collection '{collection}'\n")


def ask(question: str, collection: str) -> None:
    result = rag_graph.invoke({
        "user_query":       question,
        "session_id":       "cli",
        "collection_names": [collection],
        "iteration_count":  0,
        "messages":         [],
    })

    print("\n" + "=" * 70)
    print(result.get("final_answer", "(no answer)"))
    print("=" * 70)

    citations = result.get("citations", [])
    if citations:
        print("\nSources:")
        for c in citations:
            label = c.get("source_file", "?")
            page = c.get("page_num")
            loc = f" p.{page}" if page else ""
            print(f"  {c.get('ref', '-')} {label}{loc}")

    print(
        f"\n(intent={result.get('query_intent')}, "
        f"retrieval={result.get('retrieval_quality')}, "
        f"confidence={result.get('confidence_score', 0):.2f})\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive RAG tester")
    parser.add_argument("document", nargs="?", help="File to ingest (pdf/txt/md/xlsx/image)")
    parser.add_argument("--collection", default="default", help="Collection name")
    args = parser.parse_args()

    if args.document:
        doc = Path(args.document)
        if not doc.exists():
            raise SystemExit(f"File not found: {doc}")
        ingest(doc, args.collection)

    print("Ask questions about your document. Type 'exit' or Ctrl+C to quit.\n")
    try:
        while True:
            q = input("Question> ").strip()
            if q.lower() in {"exit", "quit", ""}:
                if q.lower() in {"exit", "quit"}:
                    break
                continue
            ask(q, args.collection)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()

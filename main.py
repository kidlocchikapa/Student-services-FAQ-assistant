"""
RAG Assistant - CLI Interface
==============================
Interactive CLI for the RAG pipeline.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _bootstrap_local_venv() -> None:
    """Relaunch with the project's virtualenv interpreter when available."""
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / "venv" / "Scripts" / "python.exe"

    if Path(sys.executable).resolve() == venv_python.resolve():
        return

    if not venv_python.exists():
        return

    result = subprocess.run([str(venv_python), __file__, *sys.argv[1:]])
    raise SystemExit(result.returncode)


_bootstrap_local_venv()
os.environ.setdefault("USER_AGENT", "student-services-faq-assistant/1.0")

from src.pipeline import RAGPipeline


def _print_interactive_help() -> None:
    print("Commands:")
    print("  teach   Add a new FAQ entry and refresh the knowledge base")
    print("  refresh Rebuild the vector index from the current data files")
    print("  help    Show this help message")
    print("  quit    Exit the assistant\n")


def _teach_interactively(pipeline: RAGPipeline) -> None:
    print("\nTeach mode: add a new UNIMA FAQ entry.")
    question = input("New question: ").strip()
    if not question:
        print("Teaching cancelled: question cannot be empty.\n")
        return

    answer = input("New answer: ").strip()
    if not answer:
        print("Teaching cancelled: answer cannot be empty.\n")
        return

    target_path = pipeline.add_faq_entry(question, answer)
    print(f"\nKnowledge saved to: {target_path}")
    print("Vector index refreshed. You can ask that question now.\n")


def interactive_mode(pipeline: RAGPipeline):
    """Run the RAG assistant in interactive mode."""
    print("=" * 50)
    print("RAG Assistant - Interactive Mode")
    print("=" * 50)
    print("Type 'help' for commands or 'quit' to stop\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if query.lower() == "help":
                _print_interactive_help()
                continue

            if query.lower() == "refresh":
                print("\nRefreshing knowledge base...")
                pipeline.refresh_knowledge()
                print("Knowledge base refreshed.\n")
                continue

            if query.lower() == "teach":
                _teach_interactively(pipeline)
                continue

            if not query:
                continue

            response = pipeline.query(query, return_sources=False)

            print(f"\nAssistant: {response['answer']}")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def demo_mode(pipeline: RAGPipeline):
    """Run with predefined demo questions."""
    demo_questions = [
        "What is the main topic of the documents?",
        "Summarize the key points.",
        "What are the main conclusions?"
    ]

    print("=" * 50)
    print("RAG Assistant - Demo Mode")
    print("=" * 50)

    for question in demo_questions:
        print(f"\nQ: {question}")
        response = pipeline.query(question)
        print(f"A: {response['answer']}")


def main():
    parser = argparse.ArgumentParser(description="RAG Assistant CLI")
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo"],
        default="interactive",
        help="Mode to run the assistant"
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing documents"
    )
    parser.add_argument(
        "--embedder",
        default="huggingface",
        choices=["huggingface", "openai", "ollama", "local"],
        help="Embedder provider"
    )
    parser.add_argument(
        "--embedder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedder model name"
    )
    parser.add_argument(
        "--llm",
        default="ollama",
        choices=["ollama", "huggingface", "openai"],
        help="LLM provider"
    )
    parser.add_argument(
        "--llm-model",
        default="phi3:latest",
        help="LLM model name"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text splitting"
    )

    args = parser.parse_args()

    print("Initializing RAG Pipeline...")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Embedder: {args.embedder} ({args.embedder_model})")
    print(f"  LLM: {args.llm} ({args.llm_model})")
    print(f"  Retrieval K: {args.k}")
    print(f"  Chunk size: {args.chunk_size}")

    pipeline = RAGPipeline(
        data_dir=args.data_dir,
        embedder_provider=args.embedder,
        embedder_model=args.embedder_model,
        llm_provider=args.llm,
        llm_model=args.llm_model,
        retrieval_k=args.k,
        chunk_size=args.chunk_size
    )

    pipeline.load_and_index()

    if args.mode == "interactive":
        interactive_mode(pipeline)
    else:
        demo_mode(pipeline)


if __name__ == "__main__":
    main()

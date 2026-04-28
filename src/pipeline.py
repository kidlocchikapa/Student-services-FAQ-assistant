"""
RAG Pipeline Orchestration
===========================
Main pipeline that ties together all RAG components.
"""

import hashlib
import json
import logging
import re
from typing import List, Optional, Dict
from pathlib import Path
from langchain_community.vectorstores import Chroma
from .loader import load_documents, chunk_documents
from .embedder import get_embedder
from .retriever import create_vectorstore, get_retriever
from .generator import get_llm, create_rag_prompt, create_qa_chain, generate_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUT_OF_SCOPE_MESSAGE = (
    "I'm here to answer University of Malawi (UNIMA) student services FAQs only. "
    "I don't have information about that topic in my current UNIMA knowledge base."
)
INDEX_SCHEMA_VERSION = "faq-pairs-v1"


class RAGPipeline:
    """
    Main RAG Pipeline class.

    Students MUST modify this class:
    - Add initialization options
    - Implement caching
    - Add evaluation hooks
    - Customize preprocessing/postprocessing
    """

    def __init__(
        self,
        data_dir: str = "data/",
        embedder_provider: str = "ollama", #can be changed
        embedder_model: str = "nomic-embed-text", # can be changed
        llm_provider: str = "ollama",
        llm_model: str = "phi3:latest", # can be changed
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        vectorstore_type: str = "chroma",
        persist_dir: Optional[str] = "vectorstore"
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.vectorstore_type = vectorstore_type
        self.persist_dir = persist_dir

        # Initialize components - students can modify configurations
        logger.info("Initializing embedder...")
        self.embedder = get_embedder(
            provider=embedder_provider,
            model_name=embedder_model
        )

        logger.info("Initializing LLM...")
        self.llm = get_llm(
            provider=llm_provider,
            model_name=llm_model
        )

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def refresh_knowledge(self) -> None:
        """Rebuild the vector index from the latest knowledge base files."""
        self.load_and_index(force_rebuild=True)

    def add_faq_entry(
        self,
        question: str,
        answer: str,
        target_file: Optional[str] = None,
    ) -> Path:
        """Append a new FAQ entry to a text knowledge-base file."""
        clean_question = " ".join(question.strip().split())
        clean_answer = " ".join(answer.strip().split())

        if not clean_question:
            raise ValueError("Question cannot be empty.")
        if not clean_answer:
            raise ValueError("Answer cannot be empty.")

        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        target_path = data_path / (target_file or "user_updates.txt")
        with target_path.open("a", encoding="utf-8") as handle:
            if target_path.stat().st_size > 0:
                handle.write("\n\n")
            handle.write(f"Q: {clean_question}\n")
            handle.write(f"A: {clean_answer}\n")

        logger.info("Added new FAQ entry to %s", target_path)
        self.refresh_knowledge()
        return target_path

    def _build_index_fingerprint(self) -> Dict[str, object]:
        data_path = Path(self.data_dir)
        source_files = []
        for file_path in sorted(data_path.glob("*")):
            if file_path.is_file():
                digest = hashlib.md5(file_path.read_bytes()).hexdigest()
                source_files.append({"name": file_path.name, "md5": digest})

        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files": source_files,
        }

    def _fingerprint_matches(self, persist_path: Path) -> bool:
        fingerprint_path = persist_path / "index_fingerprint.json"
        if not fingerprint_path.exists():
            return False

        try:
            saved = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        return saved == self._build_index_fingerprint()

    def _save_fingerprint(self, persist_path: Path) -> None:
        persist_path.mkdir(parents=True, exist_ok=True)
        fingerprint_path = persist_path / "index_fingerprint.json"
        fingerprint_path.write_text(
            json.dumps(self._build_index_fingerprint(), indent=2),
            encoding="utf-8",
        )

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z]{3,}", text.lower())
            if token not in {
                "the", "and", "for", "are", "with", "that", "this", "what",
                "when", "where", "which", "does", "have", "from", "your",
                "about", "into", "they", "their", "them", "would", "could",
                "should", "there", "here", "please", "hello", "name", "thanks",
            }
        }

    def _looks_in_scope(self, question: str, retrieved_docs: List) -> bool:
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return True

        scope_terms = {
            "unima", "university", "malawi", "student", "students", "campus",
            "admission", "apply", "application", "programs", "course", "courses",
            "accommodation", "semester", "academic", "online", "learning",
            "facilities", "library", "libraries", "ict", "contact", "services",
            "education", "faculty",
        }

        if query_tokens & scope_terms:
            return True

        doc_tokens = set()
        for doc in retrieved_docs:
            doc_tokens.update(self._tokenize(doc.page_content))
            question_text = doc.metadata.get("question")
            if question_text:
                doc_tokens.update(self._tokenize(question_text))

        overlap = query_tokens & doc_tokens
        return len(overlap) >= 1

    def load_and_index(self, force_rebuild: bool = False):
        """
        Load documents and create vector index.

        Students MUST modify:
        - Add incremental indexing
        - Implement document update logic
        """
        # Check if vectorstore exists
        persist_path = Path(self.persist_dir) if self.persist_dir else None
        has_current_index = (
            persist_path
            and persist_path.exists()
            and self._fingerprint_matches(persist_path)
        )

        if has_current_index and not force_rebuild:
            logger.info("Loading existing vectorstore...")
            # Load from persist_dir
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedder
             )
            logger.info("Vectorstore loaded from disk.")
        else:
            if persist_path and persist_path.exists() and not force_rebuild:
                logger.info("Existing vectorstore is stale. Rebuilding index...")

            logger.info("Loading and chunking documents...")
            documents = load_documents(self.data_dir)

            if not documents:
                raise RuntimeError(
                    f"No documents found in '{self.data_dir}'."
                )
            
            chunks = chunk_documents(
                documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

            logger.info("Creating vector index...")
            self.vectorstore = create_vectorstore(
                chunks,
                self.embedder,
                db_type=self.vectorstore_type,
                persist_dir=self.persist_dir
            )
            if persist_path:
                self._save_fingerprint(persist_path)
            logger.info("Vector index created")

        logger.info("Setting up retriever...")
        self.retriever = get_retriever(
            self.vectorstore,
            k=self.retrieval_k
        )

        logger.info("Creating QA chain...")
        prompt = create_rag_prompt()
        self.qa_chain = create_qa_chain(self.llm, self.retriever, prompt)

    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG pipeline.

        Students MUST modify:
        - Add query preprocessing
        - Implement response postprocessing
        - Add confidence scoring
        - Implement fallback logic
        """
        if self.qa_chain is None:
            raise RuntimeError("Pipeline not initialized. Call load_and_index() first.")

        retrieved_docs = self.retriever.invoke(question)
        if not retrieved_docs:
            return {"answer": OUT_OF_SCOPE_MESSAGE, "sources": []} if return_sources else {"answer": OUT_OF_SCOPE_MESSAGE}

        if not self._looks_in_scope(question, retrieved_docs):
            logger.info("Query rejected as out of scope: %s", question)
            response = {"answer": OUT_OF_SCOPE_MESSAGE}
            if return_sources:
                response["sources"] = []
            return response

        response = generate_response(self.qa_chain, question, return_sources)
        return response

    def evaluate(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate the pipeline on test queries.

        Students MUST implement this:
        - Calculate precision/recall
        - Measure hallucination rates
        - Add human evaluation metrics
        """
        results = []
        for item in test_queries:
            query = item["question"]
            expected = item.get("expected_answer", "")

            response = self.query(query, return_sources=True)
            results.append({
                "query": query,
                "expected": expected,
                "answer": response["answer"],
                "sources": response.get("sources", [])
            })

        #evaluation metrics
        logger.info(f"Evaluated {len(results)} queries.")
        return {"results": results}

def main():
    """Main entry point for testing."""
    pipeline = RAGPipeline(
        data_dir="data/",
        embedder_provider="ollama", # can be changed
        llm_provider="ollama",
        llm_model="phi3:latest", # can be changed
        retrieval_k=4,
        chunk_size=500 # can be changed
    )

    pipeline.load_and_index()

    test_questions = [
    "What programs does UNIMA offer?",
    "How do I apply to UNIMA?",
    "Is accommodation available on campus?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        response = pipeline.query(q)
        print(f"A: {response['answer']}")


if __name__ == "__main__":
    main()

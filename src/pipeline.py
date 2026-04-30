import hashlib
import json
import logging
import re
from typing import List, Optional, Dict, Sequence
from pathlib import Path
from langchain_community.vectorstores import Chroma
from .loader import load_documents, chunk_documents, load_web_documents
from .embedder import get_embedder
from .retriever import create_vectorstore, get_retriever
from .generator import get_llm, create_rag_prompt, create_qa_chain, generate_response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUT_OF_SCOPE_MESSAGE = (
    "I want to help, but I do not have that information in my current UNIMA knowledge base yet. "
    "If you want, ask me another UNIMA student services question, or check with the university office for the most accurate update."
)
TUITION_CLARIFICATION_MESSAGE = (
    "UNIMA tuition fees vary by study level and student category. Tell me:\n\n"
    "* undergraduate or postgraduate\n"
    "* Malawian national or international student\n\n"
    "and I'll help check the fee structure for you."
)
CONVERSATIONAL_RESPONSES = {
    "thanks": "You're welcome. If you have another question about UNIMA student services, I'm here to help.",
    "thank you": "You're welcome. If you have another question about UNIMA student services, I'm here to help.",
    "thankyou": "You're welcome. If you have another question about UNIMA student services, I'm here to help.",
    "please": "Sure. Ask me any question you have about UNIMA student services.",
    "okay": "Alright. Let me know if you want help with any UNIMA student services question.",
    "ok": "Alright. Let me know if you want help with any UNIMA student services question.",
    "hello": "Hello. I'm here to help with UNIMA student services questions.",
    "hi": "Hello. I'm here to help with UNIMA student services questions.",
    "hey": "Hello. I'm here to help with UNIMA student services questions.",
    "good morning": "Good morning. I'm here to help with UNIMA student services questions.",
    "good afternoon": "Good afternoon. I'm here to help with UNIMA student services questions.",
    "good evening": "Good evening. I'm here to help with UNIMA student services questions.",
    "good night": "Good night. I'm here whenever you need help with UNIMA student services questions.",
    "goodnight": "Good night. I'm here whenever you need help with UNIMA student services questions.",
    "sorry": "No problem. You can ask me any UNIMA student services question.",
    "who are you": "I'm your UNIMA student services assistant. I can help answer questions from the current UNIMA knowledge base.",
    "what can you do": "I can help answer questions about UNIMA student services, including fees, applications, accommodation, and other information in the knowledge base.",
    "help me": "I'm here to help. Ask me a question about UNIMA student services, and I'll do my best to answer.",
    "bye": "Goodbye. Feel free to come back with any UNIMA student services question.",
    "see you": "See you next time. I'll be here if you need help with any UNIMA student services question.",
    "see you next time": "See you next time. I'll be here if you need help with any UNIMA student services question.",
    "see you later": "See you later. I'll be here if you need help with any UNIMA student services question.",
    "later": "See you later. I'll be here if you need help with any UNIMA student services question.",
    "take care": "Take care. I'll be here if you need help with any UNIMA student services question.",
}
CONVERSATIONAL_ALIASES = {
    "tnx": "thanks",
    "thx": "thanks",
    "thank u": "thank you",
    "thanx": "thanks",
    "helo": "hello",
    "helloo": "hello",
    "hie": "hi",
    "hiya": "hi",
    "okk": "ok",
    "k": "ok",
    "pls": "please",
    "plz": "please",
    "sory": "sorry",
    "srry": "sorry",
    "gud morning": "good morning",
    "gud afternoon": "good afternoon",
    "gud evening": "good evening",
    "goodnyt": "good night",
    "gn": "good night",
    "c u": "see you",
    "cu": "see you",
    "see ya": "see you",
    "whats your name": "who are you",
    "what is your name": "who are you",
    "who r you": "who are you",
    "wat can you do": "what can you do",
    "wht can you do": "what can you do",
    "help": "help me",
}
INDEX_SCHEMA_VERSION = "faq-pairs-v1"
DEFAULT_WEB_FALLBACK_URLS = [
    "https://www.unima.ac.mw/",
]


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
        persist_dir: Optional[str] = "vectorstore",
        enable_web_fallback: bool = False,
        web_fallback_urls: Optional[Sequence[str]] = None,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.vectorstore_type = vectorstore_type
        self.persist_dir = persist_dir
        self.enable_web_fallback = enable_web_fallback
        self.web_fallback_urls = list(web_fallback_urls or DEFAULT_WEB_FALLBACK_URLS)

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
        self.web_vectorstore = None
        self.web_retriever = None
        self.qa_chain = None

    def _normalize_question(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return normalized.rstrip("?.!")

    def _get_conversational_response(self, question: str) -> Optional[str]:
        normalized_question = self._normalize_question(question)
        normalized_question = CONVERSATIONAL_ALIASES.get(normalized_question, normalized_question)
        return CONVERSATIONAL_RESPONSES.get(normalized_question)

    def _extract_answer_from_doc(self, doc) -> Optional[str]:
        match = re.search(r"Answer:\s*(.*)", doc.page_content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_fee_faq_entries(self) -> List[Dict[str, str]]:
        fee_path = Path(self.data_dir) / "fees_Structure.txt"
        if not fee_path.exists():
            return []

        text = fee_path.read_text(encoding="utf-8")
        matches = re.findall(
            r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\n\s*Q:|\Z)",
            text,
            flags=re.DOTALL,
        )
        return [
            {
                "question": " ".join(question.split()),
                "answer": answer.strip(),
            }
            for question, answer in matches
        ]

    def _try_fee_file_match(self, question: str) -> Optional[str]:
        tokens = self._tokenize(question)
        fee_terms = {"fee", "fees", "tuition", "payment", "payments", "cost", "costs"}
        if not tokens & fee_terms:
            return None

        generic_terms = fee_terms | {
            "unima", "university", "malawi", "request", "student", "details",
            "available", "structure", "clarify", "missing", "category",
            "necessary", "paid", "pay", "programme", "program", "course",
            "malawian", "international", "government", "self", "sponsored",
            "use", "how", "much",
        }
        query_terms = tokens - generic_terms
        if "masters" in query_terms:
            query_terms.add("master")
        if not query_terms:
            return None

        best_answer = None
        best_score = 0
        for entry in self._parse_fee_faq_entries():
            entry_question = entry["question"].lower()
            if "assistant" in entry_question or "what fee information is available" in entry_question:
                continue
            entry_terms = self._tokenize(f"{entry['question']} {entry['answer']}")
            if "masters" in entry_terms:
                entry_terms.add("master")
            score = len(query_terms & entry_terms)
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

        required_score = 1 if len(query_terms) <= 2 else 2
        if best_score >= required_score:
            return best_answer

        return None

    def _try_fast_faq_match(self, question: str, retrieved_docs: List) -> Optional[str]:
        normalized_question = self._normalize_question(question)
        for doc in retrieved_docs:
            source_question = doc.metadata.get("question")
            if not source_question:
                continue
            if self._normalize_question(source_question) == normalized_question:
                return self._extract_answer_from_doc(doc)
        return None

    def _is_broad_fee_question(self, question: str) -> bool:
        normalized = self._normalize_question(question)
        fee_terms = {"fee", "fees", "tuition", "payment", "payments", "cost", "costs"}
        detail_terms = {
            "programme", "program", "course", "undergraduate", "postgraduate",
            "masters", "master", "phd", "dba", "mba", "bsc", "bachelor",
            "bachelors", "degree", "education", "science", "computer",
            "business", "management", "government", "self", "sponsored",
            "international", "malawian", "year",
        }
        query_tokens = self._tokenize(normalized)
        has_fee_term = bool(query_tokens & fee_terms)
        has_detail_term = bool(query_tokens & detail_terms)
        return has_fee_term and not has_detail_term

    def _wants_fresh_information(self, question: str) -> bool:
        freshness_terms = {
            "latest", "current", "now", "today", "recent", "updated", "update",
            "new", "currently",
        }
        return bool(self._tokenize(question) & freshness_terms)

    def _merge_documents(self, primary_docs: List, secondary_docs: List) -> List:
        merged = []
        seen = set()
        for doc in [*primary_docs, *secondary_docs]:
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("question"),
                doc.page_content[:200],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged

    def _get_web_documents(self, question: str) -> List:
        if not self.web_retriever:
            return []
        try:
            return self.web_retriever.invoke(question)
        except Exception as exc:
            logger.warning("Web fallback retrieval failed: %s", exc)
            return []

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
            "education", "faculty", "graduation", "congregation", "ceremony",
            "graduand", "calendar",
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

        self.web_vectorstore = None
        self.web_retriever = None
        if self.enable_web_fallback and self.web_fallback_urls:
            try:
                logger.info("Loading official web fallback sources...")
                web_documents = load_web_documents(self.web_fallback_urls)
                if web_documents:
                    web_chunks = chunk_documents(
                        web_documents,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    self.web_vectorstore = create_vectorstore(
                        web_chunks,
                        self.embedder,
                        db_type=self.vectorstore_type,
                        persist_dir=None,
                    )
                    self.web_retriever = get_retriever(
                        self.web_vectorstore,
                        k=self.retrieval_k,
                    )
                    logger.info("Official web fallback loaded with %s chunks.", len(web_chunks))
            except Exception as exc:
                logger.warning("Unable to load official web fallback sources: %s", exc)

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

        conversational_response = self._get_conversational_response(question)
        if conversational_response:
            response = {"answer": conversational_response}
            if return_sources:
                response["sources"] = []
            return response

        local_docs = self.retriever.invoke(question)
        web_docs = []

        should_try_web = self.enable_web_fallback and (
            not local_docs or self._wants_fresh_information(question)
        )
        if should_try_web:
            web_docs = self._get_web_documents(question)

        retrieved_docs = self._merge_documents(local_docs, web_docs)
        if not retrieved_docs:
            return {"answer": OUT_OF_SCOPE_MESSAGE, "sources": []} if return_sources else {"answer": OUT_OF_SCOPE_MESSAGE}

        if not self._looks_in_scope(question, retrieved_docs):
            logger.info("Query rejected as out of scope: %s", question)
            response = {"answer": OUT_OF_SCOPE_MESSAGE}
            if return_sources:
                response["sources"] = []
            return response

        if self._is_broad_fee_question(question):
            response = {"answer": TUITION_CLARIFICATION_MESSAGE}
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                    }
                    for doc in retrieved_docs
                ]
            return response

        exact_answer = self._try_fast_faq_match(question, retrieved_docs)
        if exact_answer:
            response = {"answer": exact_answer}
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                    }
                    for doc in retrieved_docs
                ]
            return response

        fee_answer = self._try_fee_file_match(question)
        if fee_answer:
            response = {"answer": fee_answer}
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                    }
                    for doc in retrieved_docs
                ]
            return response

        response = generate_response(
            self.qa_chain,
            question,
            retrieved_docs,
            return_sources,
        )
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

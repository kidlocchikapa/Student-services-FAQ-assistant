"""
RAG Pipeline Orchestration
===========================
Main pipeline that ties together all RAG components.
"""

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
    "If you want, ask me another UNIMA student services question, or check with the university office."
)

TUITION_CLARIFICATION_MESSAGE = (
    "UNIMA tuition fees vary by study level and student category. Tell me:\n"
    "- undergraduate or postgraduate\n"
    "- Malawian or international student\n"
)

CONVERSATIONAL_RESPONSES = {
    "hello": "Hello 👋 I'm here to help with UNIMA questions.",
    "hi": "Hi 👋 Ask me anything about UNIMA student services.",
    "thanks": "You're welcome 👍",
    "thank you": "You're welcome 👍",
}


class RAGPipeline:

    def __init__(
        self,
        data_dir: str = "data/",
        embedder_provider: str = "ollama",
        embedder_model: str = "nomic-embed-text",
        llm_provider: str = "ollama",
        llm_model: str = "phi3:latest",   # ✅ FIXED (comma added here)
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
        self.web_fallback_urls = list(web_fallback_urls or [])

        logger.info("Loading embedder...")
        self.embedder = get_embedder(
            provider=embedder_provider,
            model_name=embedder_model
        )

        logger.info("Loading LLM...")
        self.llm = get_llm(
            provider=llm_provider,
            model_name=llm_model
        )

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    # -------------------------
    # BASIC CLEANING
    # -------------------------
    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _chat_response(self, question: str) -> Optional[str]:
        q = self._normalize(question)
        return CONVERSATIONAL_RESPONSES.get(q)

    # -------------------------
    # LOAD & INDEX
    # -------------------------
    def load_and_index(self):
        logger.info("Loading documents...")

        docs = load_documents(self.data_dir)
        chunks = chunk_documents(docs, self.chunk_size, self.chunk_overlap)

        logger.info(f"Chunks created: {len(chunks)}")

        self.vectorstore = create_vectorstore(
            chunks,
            self.embedder,
            db_type=self.vectorstore_type,
            persist_dir=self.persist_dir
        )

        self.retriever = get_retriever(self.vectorstore, k=self.retrieval_k)

        prompt = create_rag_prompt()
        self.qa_chain = create_qa_chain(self.llm, self.retriever, prompt)

        logger.info("Indexing complete ✅")

    # -------------------------
    # QUERY
    # -------------------------
    def query(self, question: str, return_sources: bool = True) -> Dict:

        if self.qa_chain is None:
            raise RuntimeError("Run load_and_index() first")

        # 1. chat response first
        chat = self._chat_response(question)
        if chat:
            return {"answer": chat, "sources": []}

        # 2. retrieve docs
        docs = self.retriever.invoke(question)

        if not docs:
            return {"answer": OUT_OF_SCOPE_MESSAGE, "sources": []}

        # 3. generate answer
        result = generate_response(
            self.qa_chain,
            question,
            docs,
            return_sources
        )

        return result


# -------------------------
# RUN TEST
# -------------------------
def main():
    pipeline = RAGPipeline(
        data_dir="data/",
        llm_model="phi3:latest",
        retrieval_k=4
    )

    pipeline.load_and_index()

    questions = [
        "What programs does UNIMA offer?",
        "How do I apply?",
        "How can I get a room on campus?"
    ]

    for q in questions:
        print("\nQ:", q)
        print("A:", pipeline.query(q)["answer"])


if __name__ == "__main__":
    main()

"""
LLM Generator Module
====================
Students: customize the prompt and LLM configuration.
"""

import logging
import subprocess
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


logger = logging.getLogger(__name__)


def _resolve_ollama_model(model_name: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        fallback = "phi3:latest"
        logger.warning(
            "Unable to inspect local Ollama models: %s. Falling back to '%s'.",
            exc,
            fallback,
        )
        return fallback

    installed = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            installed.append(parts[0])

    if not installed:
        return model_name

    model_aliases = {model_name}
    if ":" in model_name:
        model_aliases.add(model_name.split(":", 1)[0])
    else:
        model_aliases.add(f"{model_name}:latest")

    if any(name in installed for name in model_aliases):
        return model_name

    fallback = "phi3:latest" if "phi3:latest" in installed else installed[0]
    logger.warning(
        "Ollama model '%s' is not installed. Falling back to '%s'.",
        model_name,
        fallback,
    )
    return fallback


def get_llm(
    provider: str = "ollama",
    model_name: str = "phi3:latest",
    temperature: float = 0.4,
    **kwargs,
):
    """
    Get an LLM for generation.

    Temperature is set low (0.4) intentionally because this is a FAQ
    assistant, so we want accurate, consistent answers grounded in the
    retrieved context, not creative or variable responses.
    """
    if provider == "ollama":
        from langchain_community.llms import Ollama

        resolved_model = _resolve_ollama_model(model_name)
        return Ollama(model=resolved_model, temperature=temperature)
    if provider == "huggingface":
        from langchain_community.llms import HuggingFaceHub

        return HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={"temperature": temperature},
        )

    raise ValueError(f"Unknown provider: {provider}")


def create_rag_prompt(
    system_message: Optional[str] = None,
    template: Optional[str] = None,
) -> PromptTemplate:
    """RAG prompt template for the UNIMA Student Services assistant."""
    if system_message is None:
        system_message = (
            "You are a helpful student services assistant for the University of Malawi (UNIMA). "
            "Answer student questions using ONLY the information provided in the context below. "
            "If a question is not about UNIMA or student services, say that you only answer UNIMA student services FAQs. "
            "If the context does not contain enough information to answer the question, "
            "say: 'I don't have that information in my current knowledge base. "
            "Please contact the UNIMA student services office directly for assistance.' "
            "Do not make up information. Keep answers clear, concise, and friendly."
        )

    if template is None:
        template = f"""{system_message}:
Context:
{{context}}

Student Question: {{question}}

Answer:"""

    return PromptTemplate(template=template, input_variables=["context", "question"])


def create_qa_chain(llm, retriever=None, prompt: Optional[PromptTemplate] = None):
    """
    Create a lightweight QA config.

    Retrieval is handled in the pipeline so we avoid retrieving the same
    documents twice for one user query.
    """
    if prompt is None:
        prompt = create_rag_prompt()

    return {"llm": llm, "prompt": prompt}


def _format_context(source_documents: List[Document]) -> str:
    context_blocks = []
    for index, doc in enumerate(source_documents, start=1):
        source = doc.metadata.get("source", "unknown")
        question = doc.metadata.get("question")
        header = f"[Source {index}] {source}"
        if question:
            header += f" | Question: {question}"
        context_blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(context_blocks)


def generate_response(
    qa_chain,
    query: str,
    source_documents: List[Document],
    return_sources: bool = True,
) -> Dict:
    """
    Generate a response using the RAG pipeline.

    The pipeline supplies the retrieved documents directly so this step only
    formats context and calls the language model once.
    """
    query = query.strip()
    if query and not query.endswith("?"):
        query = query + "?"

    llm = qa_chain["llm"]
    prompt = qa_chain["prompt"]
    formatted_prompt = prompt.format(
        context=_format_context(source_documents),
        question=query,
    )
    raw_result = llm.invoke(formatted_prompt)
    answer = raw_result.strip() if isinstance(raw_result, str) else str(raw_result).strip()

    response = {"answer": answer}

    if return_sources:
        response["sources"] = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
            }
            for doc in source_documents
        ]

    return response


if __name__ == "__main__":
    prompt = create_rag_prompt()
    print("Default prompt template:")
    print(prompt.template)

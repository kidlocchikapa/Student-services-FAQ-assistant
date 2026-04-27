"""
LLM Generator Module
====================
Students: customize the prompt and LLM configuration.
"""

import logging
import subprocess
from typing import Optional, Dict

from langchain_classic.chains import RetrievalQA
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
        logger.warning("Unable to inspect local Ollama models: %s", exc)
        return model_name

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
    model_name: str = "llama3.2", # change the model if necessary
    temperature: float = 0.4, # grounding the model, reducing randomness
    **kwargs
):
    """
    Get an LLM for generation.

    Temperature is set low (0.4) intentionally — this is a FAQ assistant,
    so we want accurate, consistent answers grounded in the retrieved context,
    not creative or variable responses.

    Supported providers:
    - ollama: local inference (phi3, llama3, mistral)
    """
    if provider == "ollama":
        from langchain_community.llms import Ollama

        resolved_model = _resolve_ollama_model(model_name)
        return Ollama(model=resolved_model, temperature=temperature)
    elif provider == "huggingface":
        from langchain_community.llms import HuggingFaceHub

        return HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={"temperature": temperature}
        )
    else:
        raise ValueError(f"Unknown provider: {provider}") # shows which provider is being used in case of failure


def create_rag_prompt(
    system_message: Optional[str] = None,
    template: Optional[str] = None
) -> PromptTemplate:
    """
    RAG prompt template for the UNIMA Student Services assistant.

    modify:
    - Custom system message for different scenario
    - Added instruction formatting
    - Included few-shot examples
    """
    if system_message is None:
        system_message = (
            "You are a helpful student services assistant for the University of Malawi (UNIMA). "
            "Answer student questions using ONLY the information provided in the context below. "
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

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return prompt


def create_qa_chain(llm, retriever, prompt: Optional[PromptTemplate] = None):
    """
    A RetrievalQA chain.

    Uses chain_type="stuff" — this is appropriate here because:
    - Our FAQ document is small and chunks fit easily in the context window
    - "stuff" sends all retrieved chunks to the LLM in one call (simpler, faster)
    - For larger document sets, switch to "map_reduce" or "refine"
    """
    if prompt is None:
        prompt = create_rag_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


def generate_response(
    qa_chain,
    query: str,
    return_sources: bool = True
) -> Dict:
    """
    Generate a response using the RAG pipeline.

    Preprocesses the query slightly (strip + ensure it ends with '?')
    so the LLM receives a clean, well-formed question.

    Returns a dict with:
    - answer: the LLM's response string
    - sources: list of source chunks used (if return_sources=True
    """
    # query cleanup
    query = query.strip()
    if query and not query.endswith("?"):
        query = query + "?"

    result = qa_chain.invoke({"query": query})

    response = {
        "answer": result["result"].strip()
    }

    if return_sources and "source_documents" in result:
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
        response["sources"] = sources

    return response


if __name__ == "__main__":
    # Test prompt creation
    prompt = create_rag_prompt()
    print("Default prompt template:")
    print(prompt.template)

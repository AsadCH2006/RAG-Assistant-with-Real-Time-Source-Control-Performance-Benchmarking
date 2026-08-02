import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "groq"):
        self.persist_directory = persist_directory
        self.llm_provider = llm_provider

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB Vector Store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

        # Initialize Primary LLM
        self.llm = self._get_llm_instance(self.llm_provider)

        # System Prompt Constraints
        template = (
            "You are a helpful, accurate, and concise AI Assistant.\n"
            "Answer the user's question using ONLY the provided context snippets below.\n"
            "If the answer cannot be found in the context, explicitly state:\n"
            "\"I cannot find relevant information in the uploaded documents to answer this question.\"\n\n"
            "Context Snippets:\n{context}\n\n"
            "User Question: {question}\n\n"
            "Answer:"
        )
        self.prompt_template = ChatPromptTemplate.from_template(template)

    def _get_llm_instance(self, provider: str):
        """Initializes LLM instance based on provider with fallback capabilities."""
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_retries=1,
                    timeout=10
                )
            except Exception:
                pass

        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                max_retries=2
            )
        
        raise ValueError("No valid API Key found.")

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Retrieves contexts and generates a grounded answer."""
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)

        context_blocks = []
        sources = []

        for idx, (doc, score) in enumerate(results):
            context_blocks.append(doc.page_content)
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", idx),
                "score": float(score),
                "content": doc.page_content[:300] + "..."
            })

        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        chain = self.prompt_template | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({"context": context_str, "question": query})
        except Exception as primary_error:
            if os.getenv("OPENAI_API_KEY") and self.llm_provider != "openai":
                from langchain_openai import ChatOpenAI
                fallback_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
                fallback_chain = self.prompt_template | fallback_llm | StrOutputParser()
                answer = fallback_chain.invoke({"context": context_str, "question": query})
            else:
                raise primary_error

        return {
            "answer": answer,
            "sources": sources
        }

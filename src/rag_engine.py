import os
import httpx
from typing import List, Dict, Any
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "gemini"):
        self.persist_directory = persist_directory
        self.llm_provider = llm_provider

        # 1. Local HuggingFace Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. Vector DB Storage
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

        # 3. Prompt Template
        template = (
            "You are a helpful AI Assistant.\n"
            "Answer the user's question using ONLY the provided document context below.\n"
            "If the user asks to describe, summarize, or highlight key aspects of the document/resume, "
            "provide a comprehensive summary based on the context.\n"
            "If the information is missing from the context, state 'I cannot find relevant information in the uploaded documents.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        self.prompt_template = ChatPromptTemplate.from_template(template)

    def _get_api_key(self, key_name: str) -> str:
        """Retrieves key from Streamlit secrets or OS environment variables."""
        if hasattr(st, "secrets") and key_name in st.secrets:
            return st.secrets[key_name]
        return os.getenv(key_name, "")

    def _get_gemini_llm(self):
        gemini_key = self._get_api_key("GEMINI_API_KEY")
        if not gemini_key:
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=0.1,
            max_retries=1
        )

    def _get_groq_llm(self):
        groq_key = self._get_api_key("GROQ_API_KEY")
        if not groq_key:
            return None
        from langchain_groq import ChatGroq
        custom_http_client = httpx.Client(timeout=15.0, follow_redirects=True)
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=groq_key,
            http_client=custom_http_client,
            max_retries=1
        )

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        try:
            docs = self.vector_store.similarity_search(query, k=top_k)
        except Exception:
            docs = []

        context_blocks = []
        sources = []

        for idx, doc in enumerate(docs):
            context_blocks.append(doc.page_content)
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", idx),
                "score": 1.0,
                "content": doc.page_content[:300] + "..."
            })

        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        # Attempt Gemini first
        gemini_llm = self._get_gemini_llm()
        if gemini_llm:
            try:
                chain = self.prompt_template | gemini_llm | StrOutputParser()
                answer = chain.invoke({"context": context_str, "question": query})
                return {"answer": answer, "sources": sources}
            except Exception as e:
                st.error(f"Gemini API Error: {str(e)}")

        # Fallback to Groq
        groq_llm = self._get_groq_llm()
        if groq_llm:
            try:
                chain = self.prompt_template | groq_llm | StrOutputParser()
                answer = chain.invoke({"context": context_str, "question": query})
                return {"answer": answer, "sources": sources}
            except Exception as e:
                st.error(f"Groq API Error: {str(e)}")

        return {
            "answer": "⚠️ API Keys missing or invalid. Please check Secrets in Streamlit Cloud Settings.",
            "sources": sources
        }

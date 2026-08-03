import os
import requests
from typing import Dict, Any
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "groq"):
        self.persist_directory = persist_directory

        # 1. HuggingFace Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. Chroma Vector Store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

    def _call_groq(self, prompt: str) -> str:
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from Streamlit Secrets.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "StreamlitRAG/1.0"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Answer strictly using the provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        response = requests.post(url, headers=headers, json=payload, timeout=20)
        
        if response.status_code != 200:
            raise RuntimeError(f"Groq API returned HTTP {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        try:
            docs = self.vector_store.similarity_search(query, k=top_k)
        except Exception:
            docs = []

        context_blocks = [doc.page_content for doc in docs]
        sources = [
            {
                "filename": doc.metadata.get("filename", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", idx),
                "score": 1.0,
                "content": doc.page_content[:300] + "..."
            }
            for idx, doc in enumerate(docs)
        ]

        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        prompt = (
            f"Answer the question using ONLY the document context below.\n"
            f"If the answer cannot be found, state 'I cannot find relevant information in the uploaded documents.'\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        try:
            answer = self._call_groq(prompt)
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {"answer": f"⚠️ Groq Execution Error: {str(e)}", "sources": sources}

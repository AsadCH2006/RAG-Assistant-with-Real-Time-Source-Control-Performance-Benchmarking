import os
import requests
from typing import List, Dict, Any
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "groq"):
        self.persist_directory = persist_directory

        # 1. Local HuggingFace Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. Chroma Vector DB
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

    def _call_groq_api(self, prompt: str) -> str:
        """Direct REST API call to Groq to bypass SDK transport issues on Streamlit Cloud."""
        groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")).strip()
        
        if not groq_key:
            raise ValueError("GROQ_API_KEY is missing from Streamlit Secrets.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI Assistant. Answer questions strictly using the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        # Send request with custom session & timeout settings
        session = requests.Session()
        response = session.post(url, headers=headers, json=payload, timeout=25)
        
        if response.status_code != 200:
            raise RuntimeError(f"Groq API Error ({response.status_code}): {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        # Perform Vector Search
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

        full_prompt = (
            f"Answer the question using ONLY the context provided below.\n"
            f"If the question asks for an overview or summary of the document, provide a clear breakdown.\n"
            f"If the context contains no relevant details, state 'I cannot find relevant information in the uploaded documents.'\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        try:
            answer = self._call_groq_api(full_prompt)
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {
                "answer": f"⚠️ Groq Execution Error: {str(e)}",
                "sources": sources
            }

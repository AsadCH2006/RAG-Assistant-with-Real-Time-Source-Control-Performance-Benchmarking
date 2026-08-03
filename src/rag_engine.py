import os
import requests
from typing import Dict, Any
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "groq"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._sync_vector_store()

    def _sync_vector_store(self):
        """Re-syncs ChromaDB instance across possible collection names."""
        # Try primary collection name first
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )
        
        # Fallback to default collection if primary is empty
        try:
            raw_data = self.vector_store._collection.get()
            if not raw_data or not raw_data.get("documents"):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        except Exception:
            pass

    def _call_groq(self, prompt: str) -> str:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
        api_key = api_key.strip()
        
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from Streamlit Cloud Secrets or .env file.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "StreamlitRAG/1.0"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are an AI assistant. Answer accurately based on the provided document contexts."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        response = requests.post(url, headers=headers, json=payload, timeout=25)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API HTTP {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        self._sync_vector_store()

        try:
            raw_data = self.vector_store._collection.get()
            raw_docs = raw_data.get("documents", []) if raw_data else []
            metadatas = raw_data.get("metadatas", []) if raw_data else []
        except Exception:
            raw_docs = []
            metadatas = []

        if not raw_docs:
            return {
                "answer": "No active document found in vector storage. Please re-upload your document in 'Real-Time Source Control'.",
                "sources": []
            }

        sources = []
        context_blocks = []

        broad_keywords = ["person", "resume", "document", "tell me", "summary", "signify", "about", "overview", "indexed"]
        is_broad_query = any(kw in query.lower() for kw in broad_keywords)

        if is_broad_query or len(raw_docs) <= top_k:
            for idx, text in enumerate(raw_docs[:top_k]):
                meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
                context_blocks.append(text)
                sources.append({
                    "filename": meta.get("filename", "Uploaded Document"),
                    "chunk_index": meta.get("chunk_index", idx),
                    "score": 1.0,
                    "content": text[:300] + "..."
                })
        else:
            docs = self.vector_store.similarity_search(query, k=top_k)
            for idx, doc in enumerate(docs):
                context_blocks.append(doc.page_content)
                sources.append({
                    "filename": doc.metadata.get("filename", "Uploaded Document"),
                    "chunk_index": doc.metadata.get("chunk_index", idx),
                    "score": 1.0,
                    "content": doc.page_content[:300] + "..."
                })

        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = (
            f"Question: {query}\n\n"
            f"Document Context:\n{context_str}\n\n"
            f"Instructions:\n"
            f"- Answer the user's question clearly using ONLY the provided document context.\n"
            f"- If the query asks for an overview, summary, or what the document signifies, analyze the context and present a structured summary.\n"
            f"- If information is completely missing, state 'I cannot find relevant information in the uploaded documents.'\n\n"
            f"Answer:"
        )

        answer = self._call_groq(prompt)
        return {"answer": answer, "sources": sources}

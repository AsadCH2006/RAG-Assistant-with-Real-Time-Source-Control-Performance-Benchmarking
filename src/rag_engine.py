import os
import requests
from typing import List, Dict, Any
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db"):
        self.persist_directory = persist_directory

        # Local HuggingFace Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Chroma Store Initializer
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

    def add_documents(self, documents: List[Document]):
        """Clear old collection and store newly chunked document."""
        try:
            # Delete old documents in vector DB to prevent stale state
            existing_ids = self.vector_store.get()["ids"]
            if existing_ids:
                self.vector_store.delete(ids=existing_ids)
        except Exception:
            pass

        # Add new chunks to database
        self.vector_store.add_documents(documents)

    def _call_groq(self, prompt: str) -> str:
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY missing from Streamlit Cloud Secrets.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a professional AI Assistant. Answer strictly based on context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        response = requests.post(url, headers=headers, json=payload, timeout=25)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API HTTP Error {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    def generate_response(self, query: str) -> Dict[str, Any]:
        # Re-fetch active vector store data
        all_data = self.vector_store.get()
        all_docs = all_data.get("documents", [])

        if not all_docs:
            return {
                "answer": "No uploaded document found in active memory. Please upload your document first.",
                "sources": []
            }

        # Vector search attempt
        docs = self.vector_store.similarity_search(query, k=5)
        
        # If vector distance fails on open-ended queries ("tell me about...", "what does this signify"), fallback to raw chunks
        if not docs or any(w in query.lower() for w in ["tell me", "summary", "about", "resume", "signify", "document"]):
            context_blocks = all_docs[:5]
        else:
            context_blocks = [d.page_content for d in docs]

        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = (
            f"Here is the content extracted from an uploaded document:\n\n"
            f"--- CONTEXT START ---\n{context_str}\n--- CONTEXT END ---\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"- Answer the user's question in detail using the context above.\n"
            f"- If the question asks what the document signifies or asks for an overview, summarize the person/document clearly.\n\n"
            f"Answer:"
        )

        try:
            answer = self._call_groq(prompt)
            return {"answer": answer, "sources": context_blocks}
        except Exception as e:
            return {"answer": f"⚠️ LLM Error: {str(e)}", "sources": []}

import os
import requests
from typing import Dict, Any
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
                {"role": "system", "content": "You are a helpful AI assistant. Summarize or answer clearly based on the provided document context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API returned HTTP {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    def generate_response(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        # Always re-sync collection to catch freshly uploaded documents
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

        context_blocks = []
        sources = []

        try:
            # Check raw collection data
            raw_data = self.vector_store._collection.get()
            raw_docs = raw_data.get("documents", []) if raw_data else []

            # If documents exist, retrieve them (via similarity or direct dump for broad queries)
            if raw_docs:
                docs = self.vector_store.similarity_search(query, k=top_k)
                
                # If similarity search missed due to query phrasing, pull raw document chunks
                if not docs or any(w in query.lower() for w in ["person", "resume", "document", "tell me", "summary", "signify", "about"]):
                    for idx, txt in enumerate(raw_docs[:top_k]):
                        context_blocks.append(txt)
                        sources.append({
                            "filename": "Uploaded Document",
                            "chunk_index": idx,
                            "score": 1.0,
                            "content": txt[:200] + "..."
                        })
                else:
                    for idx, doc in enumerate(docs):
                        context_blocks.append(doc.page_content)
                        sources.append({
                            "filename": doc.metadata.get("filename", "Uploaded Document"),
                            "chunk_index": doc.metadata.get("chunk_index", idx),
                            "score": 1.0,
                            "content": doc.page_content[:300] + "..."
                        })
        except Exception:
            context_blocks = []
            sources = []

        context_str = "\n\n---\n\n".join(context_blocks).strip()

        if not context_str:
            return {
                "answer": "I cannot find relevant information in the uploaded documents. Please re-upload your document.",
                "sources": []
            }

        prompt = (
            f"You are given text extracted from an uploaded document.\n"
            f"Question: {query}\n\n"
            f"Document Context:\n{context_str}\n\n"
            f"Instructions:\n"
            f"- Answer the question comprehensively based on the document context provided above.\n"
            f"- If the question asks what the document signifies or tells about a person, provide a clear, professional summary.\n"
            f"- Only if the document is completely empty or unrelated, state 'I cannot find relevant information in the uploaded documents.'\n\n"
            f"Answer:"
        )

        try:
            answer = self._call_groq(prompt)
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {"answer": f"⚠️ Groq Execution Error: {str(e)}", "sources": sources}

import os
import httpx
from typing import List, Dict, Any
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "groq"):
        self.persist_directory = persist_directory

        # 1. HuggingFace Local Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. Vector DB (Chroma)
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

        # 3. Prompt Template
        template = (
            "You are a helpful AI Assistant.\n"
            "Answer the question using ONLY the document context provided below.\n"
            "If the question asks for an overview or summary of the document, provide a clean, structured summary from the context.\n"
            "If the context contains no relevant details to answer, state 'I cannot find relevant information in the uploaded documents.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        self.prompt_template = ChatPromptTemplate.from_template(template)

    def _get_groq_llm(self):
        # Fetch key from Streamlit secrets or environment
        groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        
        if not groq_key:
            return None

        # Custom HTTP Client configured to handle Streamlit Cloud outbound network limits
        custom_http_client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            verify=True
        )

        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=groq_key,
            http_client=custom_http_client,
            max_retries=2
        )

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        llm = self._get_groq_llm()
        if not llm:
            return {
                "answer": "⚠️ Missing Groq API Key: Please add GROQ_API_KEY to Streamlit Cloud Secrets.",
                "sources": []
            }

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

        try:
            chain = self.prompt_template | llm | StrOutputParser()
            answer = chain.invoke({"context": context_str, "question": query})
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {
                "answer": f"⚠️ Groq Execution Error: {str(e)}",
                "sources": sources
            }

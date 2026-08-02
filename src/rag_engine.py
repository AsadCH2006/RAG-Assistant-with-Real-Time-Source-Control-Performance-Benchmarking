import os
import httpx
from typing import List, Dict, Any
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "gemini"):
        self.persist_directory = persist_directory

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

        # 3. Retrieve Key from Secrets
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        
        if not api_key or not api_key.startswith("AIzaSy"):
            raise ValueError("GEMINI_API_KEY is missing or invalid in Streamlit Secrets. Must start with 'AIzaSy'.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )

        # 4. Prompt Template
        template = (
            "You are a helpful AI Assistant.\n"
            "Answer the question using ONLY the document context provided below.\n"
            "If the question asks for an overview or summary of the document, provide a clean breakdown of the key content.\n"
            "If the question cannot be answered from the context, state 'I cannot find relevant information in the uploaded documents.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        self.prompt_template = ChatPromptTemplate.from_template(template)

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

        chain = self.prompt_template | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": query})

        return {
            "answer": answer,
            "sources": sources
        }

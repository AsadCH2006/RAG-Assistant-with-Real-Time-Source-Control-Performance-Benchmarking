import os
import httpx
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "groq"):
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

        # 3. Groq Setup with explicit HTTP client to bypass network drops
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        # Create custom httpx client to prevent connection timeouts on Cloud
        custom_http_client = httpx.Client(timeout=30.0, follow_redirects=True)

        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=groq_api_key,
            http_client=custom_http_client
        )

        # 4. Strict Grounded Prompt
        template = (
            "You are a helpful AI Assistant.\n"
            "Answer the question using ONLY the context provided below.\n"
            "If the answer is not contained in the context, say 'I cannot find relevant information in the uploaded documents.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        self.prompt_template = ChatPromptTemplate.from_template(template)

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
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

        answer = chain.invoke({"context": context_str, "question": query})

        return {
            "answer": answer,
            "sources": sources
        }

import os
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb


class RAGEngine:
    """
    Handles retrieval from ChromaDB using HuggingFace local embeddings
    and generates grounded answers using Groq or OpenAI LLMs.
    """
    def __init__(
        self,
        persist_directory: str = "./vector_db",
        collection_name: str = "rag_knowledge_base",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "groq",
        model_name: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.llm_provider = llm_provider.lower()
        
        # 1. Initialize HuggingFace Local Embeddings (Matches ingestion)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # 2. Connect to ChromaDB Vector Store
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        
        # 3. Initialize LLM Provider
        if self.llm_provider == "groq":
            groq_model = model_name or "llama-3.3-70b-versatile"
            self.llm = ChatGroq(
                model_name=groq_model,
                temperature=temperature,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        elif self.llm_provider == "openai":
            openai_model = model_name or "gpt-4o-mini"
            self.llm = ChatOpenAI(
                model_name=openai_model,
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: '{llm_provider}'. Use 'groq' or 'openai'.")

        # 4. Define Strict Grounded RAG Prompt Template
        self.system_prompt = (
            "You are an expert AI Assistant specializing in document question-answering.\n"
            "Answer the question strictly based on the retrieved context below.\n"
            "If the context does not contain enough information to answer the question, "
            "state clearly: 'I cannot find the answer to this question in the uploaded documents.'\n"
            "Do not fabricate or infer any information outside the provided context.\n\n"
            "--- Context ---\n"
            "{context}\n"
            "---------------"
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}")
        ])

    def retrieve_relevant_chunks(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant text chunks and metadata from ChromaDB."""
        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=top_k
        )
        
        sources = []
        for doc, score in results:
            sources.append({
                "content": doc.page_content,
                "filename": doc.metadata.get("source_filename", "Unknown File"),
                "chunk_index": doc.metadata.get("chunk_index", "N/A"),
                "score": float(score)
            })
        return sources

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Performs full RAG pipeline: Context Retrieval -> Prompt Formatting -> LLM Generation."""
        # Step 1: Retrieve context
        sources = self.retrieve_relevant_chunks(query=query, top_k=top_k)
        
        if not sources:
            return {
                "answer": "No relevant context found in the active knowledge base documents.",
                "sources": []
            }
        
        # Step 2: Combine context blocks
        context_str = "\n\n".join(
            [f"[Source: {s['filename']} | Chunk {s['chunk_index']}]\n{s['content']}" for s in sources]
        )
        
        # Step 3: Run LangChain LCEL Chain
        chain = self.prompt_template | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": query})
        
        return {
            "answer": answer,
            "sources": sources
        }
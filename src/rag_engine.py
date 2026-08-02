import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "groq"):
        self.persist_directory = persist_directory
        self.llm_provider = llm_provider

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB Vector Store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

        # Initialize Primary LLM
        self.llm = self._get_llm_instance(self.llm_provider)

        # System Prompt Constraints
        self.prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful, accurate, and concise AI Assistant.\n"
            "Answer the user's question using ONLY the provided context snippets below.\n"
            "If the answer cannot be found in the context, explicitly state:\n"
            "\"I cannot find relevant information in the uploaded documents to answer this question.\"\n\n"
            "Context Snippets:\n{context}\n\n"
            "User Question: {question}\n\n"
            "Answer:"
        )

    def _get_llm_instance(self, provider: str):
        """Initializes LLM instance based on provider with fallback capabilities."""
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_retries=1,
                    timeout=10
                )
            except Exception:
                pass

        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                max_retries=2
            )
        
        raise ValueError("No valid API Key (GROQ_API_KEY or OPENAI_API_KEY) found.")

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Retrieves contexts and generates a grounded answer with error handling."""
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

        try:
            answer = chain.invoke({"context": context_str, "question": query})
        except Exception as primary_error:
            if os.getenv("OPENAI_API_KEY") and self.llm_provider != "openai":
                from langchain_openai import ChatOpenAI
                fallback_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
                fallback_chain = self.prompt_template | fallback_llm | StrOutputParser()
                answer = fallback_chain.invoke({"context": context_str, "question": query})
            else:
                raise primary_error

        return {
            "answer": answer,
            "sources": sources
        }import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_db", llm_provider: str = "groq"):
        self.persist_directory = persist_directory
        self.llm_provider = llm_provider

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB Vector Store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )

        # Initialize Primary and Fallback LLMs
        self.llm = self._get_llm_instance(self.llm_provider)

        # System Prompt Constraints
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful, accurate, and concise AI Assistant.
Answer the user's question using ONLY the provided context snippets below.
If the answer cannot be found in the context, explicitly state:
"I cannot find relevant information in the uploaded documents to answer this question."

Context Snippets:
{context}

User Question: {question}

Answer:
""")

    def _get_llm_instance(self, provider: str):
        """Initializes LLM instance based on provider with fallback capabilities."""
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_retries=1,
                    timeout=10
                )
            except Exception:
                pass  # Fall through to OpenAI if Groq fails to load

        # Default / Fallback to OpenAI
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.1,
                max_retries=2
            )
        else:
            raise ValueError("No valid API Key (GROQ_API_KEY or OPENAI_API_KEY) found.")

    def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Retrieves contexts and generates a grounded answer with error handling."""
        # 1. Similarity Search with Scores
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

        # 2. Build Generation Chain
        chain = self.prompt_template | self.llm | StrOutputParser()

        # 3. Safe Invocation with OpenAI Fallback
        try:
            answer = chain.invoke({"context": context_str, "question": query})
        except Exception as primary_error:
            # If Groq fails due to connection error, attempt OpenAI
            if os.getenv("OPENAI_API_KEY") and self.llm_provider != "openai":
                from langchain_openai import ChatOpenAI
                fallback_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
                fallback_chain = self.prompt_template | fallback_llm | StrOutputParser()
                answer = fallback_chain.invoke({"context": context_str, "question": query})
            else:
                raise primary_error

        return {
            "answer": answer,
            "sources": sources
        }import os
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

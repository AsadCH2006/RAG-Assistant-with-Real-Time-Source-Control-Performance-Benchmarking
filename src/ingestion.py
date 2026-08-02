import os
import uuid
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb


class RAGVectorManager:
    """
    Manages document loading, metadata tagging, vector ingestion, 
    and dynamic vector deletion in ChromaDB using free local embeddings.
    """
    def __init__(
        self,
        persist_directory: str = "./vector_db",
        collection_name: str = "rag_knowledge_base",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 1. Initialize Free Local Embedding Model (No API key needed)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # 2. Instantiate Direct Chroma Native Client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # 3. Instantiate LangChain VectorStore Wrapper
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Loads PDF, TXT, or DOCX documents based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return loader.load()

    def process_and_add_document(
        self, 
        file_path: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Loads document, chunks it, attaches standard + custom metadata, 
        and adds embeddings to ChromaDB.
        """
        filename = os.path.basename(file_path)
        print(f"📥 Loading '{filename}'...")
        
        # Load Raw Documents
        raw_docs = self.load_document(file_path)
        
        # Chunking Logic
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(raw_docs)
        
        # Generate Chunk Unique IDs & Tag Metadata
        chunk_ids = []
        for index, chunk in enumerate(chunks):
            # Formulate standard metadata required for dynamic management
            chunk.metadata["source_filename"] = filename
            chunk.metadata["file_path"] = file_path
            chunk.metadata["chunk_index"] = index
            
            if custom_metadata:
                chunk.metadata.update(custom_metadata)
                
            # Create persistent unique chunk UUID
            chunk_id = f"{filename}_chunk_{index}_{uuid.uuid4().hex[:6]}"
            chunk_ids.append(chunk_id)

        # Ingest into VectorDB
        print(f"⚙️ Generating embeddings for {len(chunks)} chunks...")
        self.vector_store.add_documents(documents=chunks, ids=chunk_ids)
        print(f"✅ Successfully indexed '{filename}'.")
        
        return chunk_ids

    def delete_document_by_filename(self, filename: str) -> bool:
        """
        Removes ALL vector embeddings and metadata matching a specific filename from ChromaDB.
        """
        print(f"🗑️ Deleting all vectors for source document: '{filename}'...")
        
        # Access Chroma collection directly to execute metadata filtering
        collection = self.chroma_client.get_collection(name=self.collection_name)
        
        # Execute delete operation filtering by metadata 'source_filename'
        collection.delete(where={"source_filename": filename})
        print(f"✅ Removed all chunks associated with '{filename}'.")
        return True

    def list_indexed_files(self) -> List[str]:
        """Returns a list of unique filenames currently indexed in the vector store."""
        collection = self.chroma_client.get_collection(name=self.collection_name)
        results = collection.get(include=["metadatas"])
        
        if not results or not results["metadatas"]:
            return []
            
        filenames = {meta.get("source_filename") for meta in results["metadatas"] if meta and "source_filename" in meta}
        return list(filenames)
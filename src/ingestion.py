import os
import shutil
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RAGVectorManager:
    def __init__(self, persist_directory: str = "./vector_db", collection_name: str = "rag_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Local HuggingFace Embedding Model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Text Splitter Strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize ChromaDB Vector Store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Loads PDF, TXT, or DOCX files into LangChain Document format."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: '{ext}'")

        documents = loader.load()
        
        # Attach filename metadata to every document page/chunk
        for doc in documents:
            doc.metadata["filename"] = path.name

        return documents

    def process_and_add_document(self, file_path: str) -> int:
        """Processes a file, splits into chunks, and stores in ChromaDB."""
        # 1. Load document text
        raw_documents = self.load_document(file_path)

        # 2. Chunk text into smaller segments
        chunks = self.text_splitter.split_documents(raw_documents)

        # 3. Add explicit chunk index to metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx

        # 4. Insert chunks into ChromaDB
        self.vector_store.add_documents(chunks)
        
        return len(chunks)

    def list_indexed_files(self) -> List[str]:
        """Returns a list of unique filenames currently indexed in the vector database."""
        try:
            raw_data = self.vector_store._collection.get()
            metadatas = raw_data.get("metadatas", []) if raw_data else []
            
            filenames = set()
            for meta in metadatas:
                if meta and "filename" in meta:
                    filenames.add(meta["filename"])
            return sorted(list(filenames))
        except Exception:
            return []

    def delete_document_by_filename(self, filename: str) -> bool:
        """Removes all vector embeddings belonging to a specific file."""
        try:
            raw_data = self.vector_store._collection.get()
            if not raw_data:
                return False

            ids = raw_data.get("ids", [])
            metadatas = raw_data.get("metadatas", [])

            ids_to_delete = [
                doc_id for doc_id, meta in zip(ids, metadatas)
                if meta and meta.get("filename") == filename
            ]

            if ids_to_delete:
                self.vector_store.delete(ids=ids_to_delete)
                return True
            return False
        except Exception as e:
            print(f"Error deleting document {filename}: {str(e)}")
            return False

    def clear_database(self) -> None:
        """Completely purges the ChromaDB persistent directory."""
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            
            # Re-initialize vector store
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        except Exception as e:
            print(f"Error purging database: {str(e)}")

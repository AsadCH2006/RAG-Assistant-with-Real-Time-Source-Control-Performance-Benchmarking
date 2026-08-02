import os
import sys
import time
from pathlib import Path

# Explicitly add root directory to Python path for Streamlit Cloud compatibility
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import custom RAG modules from src/
from src.ingestion import RAGVectorManager
from src.rag_engine import RAGEngine

# Force load local .env if available
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic RAG Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Sidebar Configuration & Secrets Fallback
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ Control Panel")
st.sidebar.caption("RAG Control & Performance Monitor")
st.sidebar.markdown("---")

st.sidebar.subheader("🔑 API Credentials")

# Fetch keys from environment or Streamlit Secrets (Cloud deployment)
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

# Fallback UI inputs if missing
user_openai_key = st.sidebar.text_input(
    "OpenAI Key", 
    value=openai_api_key, 
    type="password",
    help="Required for OpenAI LLM or default verification."
)

user_groq_key = st.sidebar.text_input(
    "Groq Key", 
    value=groq_api_key, 
    type="password",
    help="Recommended for ultra-fast Llama-3 inference."
)

if user_openai_key:
    os.environ["OPENAI_API_KEY"] = user_openai_key
if user_groq_key:
    os.environ["GROQ_API_KEY"] = user_groq_key

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Retrieval Parameters")
top_k = st.sidebar.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=4)

st.sidebar.markdown("---")
st.sidebar.caption("Status: **Active System Online 🟢**")

# Halt execution if no OpenAI key is set
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Please provide an OpenAI API Key in the sidebar or via Streamlit Secrets / .env file to initialize the application.")
    st.stop()

# -----------------------------------------------------------------------------
# System Initialization (Session State)
# -----------------------------------------------------------------------------
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = RAGVectorManager(persist_directory="./vector_db")

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(
        persist_directory="./vector_db",
        llm_provider="groq" if os.getenv("GROQ_API_KEY") else "openai"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latency_logs" not in st.session_state:
    st.session_state.latency_logs = []

UPLOAD_DIR = ROOT_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Main Application UI & Navigation Tabs
# -----------------------------------------------------------------------------
st.title("🧠 Dynamic RAG Knowledge Base")

tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📂 Real-Time Source Control", "📊 Performance Benchmarking"])

# -----------------------------------------------------------------------------
# TAB 1: Chat Assistant
# -----------------------------------------------------------------------------
with tab1:
    st.caption("Ask questions grounded strictly in your active vector documents.")

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Cited Sources"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**File:** `{src['filename']}` | **Chunk:** `{src['chunk_index']}` | "
                            f"**Score:** `{src['score']:.4f}`"
                        )
                        st.caption(src["content"])

    # User Query Input
    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base & generating response..."):
                start_time = time.time()
                response = st.session_state.rag_engine.generate_response(query=prompt, top_k=top_k)
                elapsed_time = round(time.time() - start_time, 2)

                st.markdown(response["answer"])

                if response["sources"]:
                    with st.expander("📚 View Cited Sources"):
                        for src in response["sources"]:
                            st.markdown(
                                f"**File:** `{src['filename']}` | **Chunk:** `{src['chunk_index']}` | "
                                f"**Score:** `{src['score']:.4f}`"
                            )
                            st.caption(src["content"])

                # Log Telemetry Metrics
                st.session_state.latency_logs.append({
                    "Timestamp": time.strftime("%H:%M:%S"),
                    "Query": prompt,
                    "Latency (s)": elapsed_time,
                    "Chunks Retrieved": len(response["sources"])
                })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"]
        })

# -----------------------------------------------------------------------------
# TAB 2: Dynamic Source Manager
# -----------------------------------------------------------------------------
with tab2:
    st.header("Document Management")
    
    col1, col2 = st.columns([1, 1])

    # Upload Section
    with col1:
        st.subheader("Upload New Document")
        uploaded_file = st.file_uploader(
            "Choose a file (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"]
        )

        if uploaded_file is not None:
            if st.button("Process & Index Document", type="primary"):
                file_path = UPLOAD_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Indexing '{uploaded_file.name}' into ChromaDB..."):
                    st.session_state.vector_manager.process_and_add_document(str(file_path))
                    st.success(f"Indexed `{uploaded_file.name}` successfully!")

                if file_path.exists():
                    os.remove(file_path)
                st.rerun()

    # Active Vector Files & Deletion
    with col2:
        st.subheader("Active Knowledge Base Documents")
        indexed_files = st.session_state.vector_manager.list_indexed_files()

        if not indexed_files:
            st.info("No documents currently indexed in ChromaDB.")
        else:
            for filename in indexed_files:
                f_col1, f_col2 = st.columns([3, 1])
                f_col1.write(f"📄 **{filename}**")
                
                if f_col2.button("Delete", key=f"del_{filename}"):
                    with st.spinner(f"Purging vectors for '{filename}'..."):
                        st.session_state.vector_manager.delete_document_by_filename(filename)
                        st.success(f"Deleted `{filename}` vectors.")
                    st.rerun()

# -----------------------------------------------------------------------------
# TAB 3: Performance Benchmarking
# -----------------------------------------------------------------------------
with tab3:
    st.header("Performance & Telemetry Dashboard")

    if not st.session_state.latency_logs:
        st.info("No query metrics recorded yet. Ask a question in the Chat Assistant tab to view live performance telemetry.")
    else:
        df_logs = pd.DataFrame(st.session_state.latency_logs)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Queries Executed", len(df_logs))
        m_col2.metric("Average Latency", f"{df_logs['Latency (s)'].mean():.2f}s")
        m_col3.metric("Max Latency", f"{df_logs['Latency (s)'].max():.2f}s")

        st.subheader("Query Execution History")
        st.dataframe(df_logs, use_container_width=True)

        st.subheader("Latency Trend")
        st.line_chart(df_logs, x="Timestamp", y="Latency (s)")

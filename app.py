import os
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. Environment & Path Setup
# -----------------------------------------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

st.set_page_config(
    page_title="RAG Assistant | Real-Time Source Control & Benchmarking",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables Early
if "benchmark_logs" not in st.session_state:
    st.session_state.benchmark_logs = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# 2. Custom CSS Theme Styling
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #c9d1d9;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #58a6ff !important;
    }
    .stMetric {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid #30363d;
        padding: 12px;
        border-radius: 8px;
    }
    .stButton>button {
        border-radius: 6px;
        border: 1px solid #30363d;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #58a6ff;
        color: #58a6ff;
    }
    .streamlit-expanderHeader {
        background-color: rgba(22, 27, 34, 0.8) !important;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Sidebar API Key Configuration & Live Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Control Panel")
    st.caption("RAG Control & Performance Monitor")
    st.divider()

    st.subheader("🔑 API Credentials")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    if not openai_api_key:
        openai_api_key = st.text_input("OpenAI Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    if not groq_api_key:
        groq_api_key = st.text_input("Groq Key", type="password")
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

    st.divider()
    
    st.subheader("⚙️ Retrieval Parameters")
    top_k = st.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=4)
    
    st.divider()
    st.caption("Status: **Active System Online** 🟢")

# Block execution if missing crucial key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Please provide an **OpenAI API Key** in the sidebar or via `.env` file to initialize the application.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. Lazy-load RAG Modules
# -----------------------------------------------------------------------------
from src.ingestion import RAGVectorManager
from src.rag_engine import RAGEngine

if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = RAGVectorManager(persist_directory="./vector_db")

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(
        persist_directory="./vector_db",
        llm_provider="groq" if os.getenv("GROQ_API_KEY") else "openai"
    )

UPLOAD_DIR = Path("./temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# 5. Header Section & Synchronized Dashboard Metrics
# -----------------------------------------------------------------------------
st.title("🤖 RAG Assistant with Real-Time Source Control & Performance Benchmarking")
st.markdown("Dynamic knowledge base query assistant powered by local sentence embeddings and vector analytics.")

# Synchronized Metrics Calculation
indexed_files = st.session_state.vector_manager.list_indexed_files()
total_queries = len(st.session_state.benchmark_logs)

if total_queries > 0:
    df_metrics = pd.DataFrame(st.session_state.benchmark_logs)
    avg_latency = f"{df_metrics['latency_sec'].mean():.3f}s"
else:
    avg_latency = "N/A"

col1, col2, col3 = st.columns(3)
col1.metric("Indexed Documents", len(indexed_files))
col2.metric("Total Queries Executed", total_queries)
col3.metric("Avg Response Time", avg_latency)

st.divider()

# -----------------------------------------------------------------------------
# 6. Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📂 Real-Time Source Control", "📊 Performance Benchmarking"])

# -----------------------------------------------------------------------------
# TAB 1: Chat Assistant
# -----------------------------------------------------------------------------
with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Cited Sources"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**File:** `{src['filename']}` | **Chunk:** `{src['chunk_index']}` | "
                            f"**Similarity Score:** `{src['score']:.4f}`"
                        )
                        st.caption(src["content"])

    if prompt := st.chat_input("Ask a question about your active documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving vector contexts & benchmarking response..."):
                start_time = time.time()
                response = st.session_state.rag_engine.generate_response(query=prompt, top_k=top_k)
                latency = time.time() - start_time
                
                st.markdown(response["answer"])
                
                if response["sources"]:
                    with st.expander("📚 View Cited Sources"):
                        for src in response["sources"]:
                            st.markdown(
                                f"**File:** `{src['filename']}` | **Chunk:** `{src['chunk_index']}` | "
                                f"**Similarity Score:** `{src['score']:.4f}`"
                            )
                            st.caption(src["content"])

        st.session_state.benchmark_logs.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "query": prompt,
            "latency_sec": round(latency, 3),
            "retrieved_chunks": len(response["sources"]),
            "top_score": round(response["sources"][0]["score"], 4) if response["sources"] else 0.0
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"]
        })
        st.rerun()

# -----------------------------------------------------------------------------
# TAB 2: Real-Time Source Control
# -----------------------------------------------------------------------------
with tab2:
    st.header("📂 Real-Time Knowledge Base Management")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Add Document")
        uploaded_file = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
        
        if uploaded_file is not None:
            if st.button("⚡ Index Document", type="primary"):
                file_path = UPLOAD_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Splitting & embedding '{uploaded_file.name}'..."):
                    st.session_state.vector_manager.process_and_add_document(str(file_path))
                    st.success(f"Indexed `{uploaded_file.name}` into ChromaDB!")

                if file_path.exists():
                    os.remove(file_path)
                st.rerun()

    with c2:
        st.subheader("Active Knowledge Base Index")
        current_files = st.session_state.vector_manager.list_indexed_files()
        
        if not current_files:
            st.info("No documents indexed in the vector store.")
        else:
            for fname in current_files:
                fc1, fc2 = st.columns([3, 1])
                fc1.write(f"📄 **{fname}**")
                if fc2.button("Remove", key=f"del_{fname}"):
                    st.session_state.vector_manager.delete_document_by_filename(fname)
                    st.success(f"Removed vectors for `{fname}`")
                    st.rerun()

# -----------------------------------------------------------------------------
# TAB 3: Performance Benchmarking
# -----------------------------------------------------------------------------
with tab3:
    st.header("📊 Real-Time Performance Benchmarking")
    
    if not st.session_state.benchmark_logs:
        st.info("No benchmarking telemetry gathered yet. Submit questions in the Chat tab to log latency metrics.")
    else:
        df_bench = pd.DataFrame(st.session_state.benchmark_logs)
        
        b_col1, b_col2, b_col3 = st.columns(3)
        b_col1.metric("Min Latency", f"{df_bench['latency_sec'].min():.3f}s")
        b_col2.metric("Max Latency", f"{df_bench['latency_sec'].max():.3f}s")
        b_col3.metric("Avg Similarity Score", f"{df_bench['top_score'].mean():.4f}")

        st.subheader("Latency Analytics History")
        st.line_chart(df_bench.set_index("timestamp")["latency_sec"])

        st.subheader("Detailed Query Benchmark Logs")
        st.dataframe(df_bench, use_container_width=True)
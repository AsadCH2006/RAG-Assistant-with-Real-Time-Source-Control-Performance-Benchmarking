# 🤖 RAG Assistant with Real-Time Source Control & Performance Benchmarking

A high-performance, modular Retrieval-Augmented Generation (RAG) web application built using **Streamlit**, **LangChain**, **ChromaDB**, and **Groq / OpenAI**.

This application enables users to upload custom documents (PDF, TXT, DOCX), process and embed them locally, query the knowledge base in real-time with grounded responses, dynamically delete vector collections, and monitor end-to-end system latency and similarity scoring via built-in performance telemetry.

---

## 🌟 Key Features

* **⚡ Real-Time Source Control:** Dynamically upload new documents, inspect currently indexed files in ChromaDB, and delete vector embeddings on the fly without restarting the app.
* **🛡️ Grounded & Hallucination-Free QA:** Strict system prompt constraints ensure the assistant responds *only* using information present in the retrieved chunks.
* **🔍 Full Citation & Transparency:** Expandable source viewers display exact document sources, chunk indices, and vector similarity scores for every response.
* **📊 Performance Benchmarking & Telemetry:** Track latency (min/max/average) across queries, monitor similarity score trends, and inspect detailed execution logs in real-time.
* **💰 Zero-Cost Local Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace for fast, 100% free vector embeddings—eliminating OpenAI quota limits.
* **🚀 Multi-LLM Support:** Seamless support for high-speed inference via **Groq** (`llama-3.3-70b-versatile`) or **OpenAI** (`gpt-4o-mini`).
* **🎨 Modern UI/UX:** Styled dark theme with glassmorphism panels, interactive metrics, and multi-tab layout.

---

## 📂 Project Structure

```text
├── .env                  # API Credentials & Environment Variables
├── app.py                # Main Streamlit Dashboard & UI Application
├── requirements.txt      # Python Dependencies
├── src/
│   ├── ingestion.py      # Document Loader, Text Splitter & ChromaDB Vector Manager
│   └── rag_engine.py     # Context Retrieval & LLM Generation Pipeline
└── vector_db/            # Persistent Local Vector Database Storage (ChromaDB)
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/rag-assistant.git
cd rag-assistant
```

### 3. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Create a `.env` file in the root directory of your project:

```env
OPENAI_API_KEY=sk-proj-your-openai-key-here
GROQ_API_KEY=gsk_your-groq-key-here
```

> **Note:**
> * `OPENAI_API_KEY` is validated upon launch.
> * `GROQ_API_KEY` is recommended for ultra-fast, low-latency LLM inference.
> * Keys can also be added directly via the **Control Panel Sidebar** at runtime.

---

## 🚀 Running the Application

Launch the Streamlit web app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥️ How to Use

1. **Upload Documents:** Navigate to the **📂 Real-Time Source Control** tab, choose a file (`.pdf`, `.txt`, `.docx`), and click **Index Document**.
2. **Query Knowledge Base:** Switch to the **💬 Chat Assistant** tab and ask questions grounded in your document context.
3. **Inspect Citations:** Click on **📚 View Cited Sources** below any response to review exact chunk citations and vector similarity metrics.
4. **Monitor Performance:** Check the **📊 Performance Benchmarking** tab to analyze query latency history and similarity scoring trends.
5. **Manage Vectors:** Delete individual documents in the Source Control tab to immediately purge their embeddings from ChromaDB.

---

## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` ([HuggingFace](https://huggingface.co/))
* **LLM Engine:** [Groq API](https://groq.com/) / [OpenAI API](https://openai.com/)
* **Data Processing:** Pandas, PyPDF, Docx2txt

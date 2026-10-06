import os
import time
import tempfile
import datetime
import html

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: modern dark theme ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #111827 50%, #0f172a 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 30, 0.95) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
    backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c4b5fd !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    padding: 0;
}

.app-subheader {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 4px;
    margin-bottom: 1.5rem;
}

/* ── Welcome card ── */
.welcome-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 600px;
}

.welcome-card h2 {
    color: #e2e8f0;
    font-size: 1.5rem;
    margin-bottom: 1rem;
}

.welcome-card p {
    color: #94a3b8;
    line-height: 1.6;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.user-bubble {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1)) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 1rem 1.2rem !important;
    margin-left: 2rem !important;
}

/* AI bubble */
[data-testid="stChatMessage"]:not([data-testid*="user"]) {
    background: rgba(30, 30, 50, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 1rem 1.2rem !important;
    margin-right: 2rem !important;
    backdrop-filter: blur(8px);
}

/* ── Source chunk cards ── */
.source-card {
    background: rgba(20, 20, 40, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}

.source-card .source-num {
    color: #6366f1;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

/* ── Doc card in sidebar ── */
.doc-card {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    margin: 0.3rem 0;
    font-size: 0.85rem;
    color: #c4b5fd;
    cursor: pointer;
}

.doc-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    background: rgba(99, 102, 241, 0.15);
}

/* ── Status badge ── */
.status-ready {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    color: #6ee7b7;
    font-size: 0.82rem;
}

.status-idle {
    background: rgba(100, 116, 139, 0.1);
    border: 1px solid rgba(100, 116, 139, 0.2);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    color: #94a3b8;
    font-size: 0.82rem;
}

/* ── Divider ── */
hr {
    border-color: rgba(99, 102, 241, 0.15) !important;
}

/* ── Input box ── */
[data-testid="stChatInput"] {
    background: rgba(20, 20, 40, 0.9) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    backdrop-filter: blur(8px);
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99, 102, 241, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(20, 20, 40, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 10px !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(20, 20, 40, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Toast/progress ── */
[data-testid="stProgress"] > div {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border-radius: 4px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(20, 20, 40, 0.5) !important;
    border: 1.5px dashed rgba(99, 102, 241, 0.35) !important;
    border-radius: 12px !important;
}

/* ── Timestamp ── */
.msg-timestamp {
    font-size: 0.7rem;
    color: #475569;
    margin-top: 4px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }
</style>
""", unsafe_allow_html=True)


# ── Session-state init ─────────────────────────────────────────────────────────
defaults = {
    "pipeline": None,
    "chat_history": [],          # [(user, ai), ...]
    "messages": [],              # [{"role", "content", "sources", "time"}, ...]
    "doc_ready": False,
    "doc_name": "",
    "chunk_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Pipeline (cached across reruns) ───────────────────────────────────────────
@st.cache_resource(show_spinner="🔌 Connecting to vector database…")
def get_pipeline():
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline()


def ts():
    """Current time string for message timestamps."""
    return datetime.datetime.now().strftime("%H:%M")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / branding
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2rem;">🧠</div>
        <div style="color:#c4b5fd; font-weight:700; font-size:1.1rem; letter-spacing:0.5px;">DocChat AI</div>
        <div style="color:#475569; font-size:0.75rem;">Powered by Groq + ChromaDB</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Upload section ─────────────────────────────────────────────────────────
    st.markdown("**📂 Upload Document**")
    uploaded_file = st.file_uploader(
        label="PDF or DOCX",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        process_btn = st.button(
            "⚡ Process & Index",
            use_container_width=True,
            type="primary",
        )

        if process_btn:
            progress = st.progress(0, text="Reading file…")
            try:
                pipeline = get_pipeline()
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                progress.progress(20, text="Extracting text…")
                time.sleep(0.2)

                progress.progress(40, text="Chunking document…")
                # ingest does chunking + embedding + storing
                chunk_count = pipeline.ingest(tmp_path, filename=uploaded_file.name)
                os.unlink(tmp_path)

                progress.progress(90, text="Saving to ChromaDB…")
                time.sleep(0.2)
                progress.progress(100, text="Done!")
                time.sleep(0.4)
                progress.empty()

                st.session_state.pipeline = pipeline
                pipeline.ingested_filename = None
                st.session_state.doc_ready = False
                st.session_state.doc_name = ""
                st.session_state.chunk_count = 0
                st.session_state.chat_history = []
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": (
                        f"✅ **{uploaded_file.name}** is indexed!\n\n"
                        f"I've stored **{chunk_count} chunks** in ChromaDB. "
                        "Ask a question to search across all stored documents, or select a document in the sidebar."
                    ),
                    "sources": [],
                    "time": ts(),
                }]
                st.rerun()

            except Exception as e:
                progress.empty()
                st.error(f"❌ {e}")

    st.divider()

    # ── Stored documents ───────────────────────────────────────────────────────
    st.markdown("**🗂️ Stored Documents**")
    try:
        pipeline = get_pipeline()
        if not st.session_state.doc_ready:
            pipeline.ingested_filename = None
        st.session_state.pipeline = pipeline
        stored = pipeline.previously_ingested_files()
    except Exception:
        stored = []

    if stored:
        if st.button(
            "🔎 Search all stored documents",
            key="search_all_documents",
            use_container_width=True,
            type="primary" if not st.session_state.doc_ready else "secondary",
        ):
            pipeline = get_pipeline()
            pipeline.ingested_filename = None
            st.session_state.pipeline = pipeline
            st.session_state.doc_ready = False
            st.session_state.doc_name = ""
            st.session_state.chunk_count = 0
            st.session_state.chat_history = []
            st.session_state.messages = [{
                "role": "assistant",
                "content": f"🔎 Searching across all **{len(stored)} stored documents**. Ask me anything!",
                "sources": [],
                "time": ts(),
            }]
            st.rerun()

        for fname in stored:
            is_active = fname == st.session_state.doc_name
            icon = "🟢" if is_active else "📄"
            if st.button(
                f"{icon} {fname}",
                key=f"doc_{fname}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                pipeline = get_pipeline()
                pipeline.ingested_filename = fname
                pipeline.chunk_count = pipeline.vectorstore.chunk_count(fname)
                st.session_state.pipeline = pipeline
                st.session_state.doc_ready = True
                st.session_state.doc_name = fname
                st.session_state.chunk_count = pipeline.chunk_count
                st.session_state.chat_history = []
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": f"📂 Loaded **{fname}** from database ({pipeline.chunk_count} chunks). Ask me anything!",
                    "sources": [],
                    "time": ts(),
                }]
                st.rerun()
    else:
        st.markdown('<div class="status-idle">No documents stored yet.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Status ─────────────────────────────────────────────────────────────────
    if st.session_state.doc_ready:
        st.markdown(f"""
        <div class="status-ready">
            🟢 <strong>Active</strong><br>
            📄 {st.session_state.doc_name}<br>
            🔢 {st.session_state.chunk_count} chunks indexed
        </div>
        """, unsafe_allow_html=True)
    elif stored:
        st.markdown(f"""
        <div class="status-ready">
            🔎 <strong>Searching all documents</strong><br>
            📚 {len(stored)} documents available
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-idle">⏳ No document selected</div>', unsafe_allow_html=True)

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────────
    st.markdown("**⚙️ Settings**")
    top_k = st.slider("Sources to retrieve", 3, 12, 7, help="More sources = richer, more detailed long-form explanation")
    show_sources = st.toggle("Show source chunks", value=True, help="Display document excerpts used to answer")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset", use_container_width=True, help="Reset document selection"):
            st.session_state.doc_ready = False
            st.session_state.doc_name = ""
            st.session_state.messages = []
            st.session_state.chat_history = []
            if st.session_state.pipeline:
                st.session_state.pipeline.ingested_filename = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<p class="app-header">🧠 DocChat AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subheader">Chat with your documents — powered by RAG + Groq</p>', unsafe_allow_html=True)

with col_h2:
    if st.session_state.doc_ready:
        st.markdown(f"""
        <div style="text-align:right; padding-top:0.5rem;">
            <span style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
                         border-radius:20px; padding:4px 12px; color:#6ee7b7; font-size:0.8rem;">
                🟢 {st.session_state.doc_name}
            </span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Welcome screen ─────────────────────────────────────────────────────────────
if not st.session_state.doc_ready and not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:3rem; margin-bottom:1rem;">🤖</div>
        <h2>Welcome to DocChat AI</h2>
        <p>Upload a <strong style="color:#c4b5fd">PDF or DOCX</strong> from the sidebar,
           select a previously indexed document, or ask a question to search all stored documents.<br><br>
           Answers use retrieved document content, and source results identify the file each passage came from —
           no hallucinations, just facts.</p>
        <br>
        <div style="display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;">
            <div style="color:#6366f1; font-size:0.85rem;">📤 Upload document</div>
            <div style="color:#8b5cf6; font-size:0.85rem;">⚡ Auto-index</div>
            <div style="color:#a78bfa; font-size:0.85rem;">💬 Chat instantly</div>
            <div style="color:#c4b5fd; font-size:0.85rem;">🔍 Source citations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

        # Timestamp
        if msg.get("time"):
            st.markdown(f'<div class="msg-timestamp">{msg["time"]}</div>', unsafe_allow_html=True)

        # Source chunks (only for assistant messages)
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source chunk(s) used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    content = src.get("content", "")
                    preview = html.escape(content[:300] + "…" if len(content) > 300 else content)
                    filename = html.escape(str(src.get("filename", st.session_state.doc_name or "Unknown file")))
                    page_number = src.get("page_number")
                    page_label = f"page {page_number}" if page_number is not None else "page not indexed"
                    rerank_score = src.get("rerank_score")
                    score_label = f" · reranker score: {rerank_score:.2f}" if rerank_score is not None else ""
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-num">Source {i} · {filename} · {page_label}{score_label}</div>
                        {preview}
                    </div>
                    """, unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
prompt = st.chat_input(
    placeholder=(
        "Ask anything about your document…"
        if st.session_state.doc_ready
        else "Ask across all stored documents…" if stored
        else "Upload a document first…"
    ),
    disabled=not st.session_state.doc_ready and not stored,
)

if prompt:
    # Show user message
    now = ts()
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": [], "time": now})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.markdown(f'<div class="msg-timestamp">{now}</div>', unsafe_allow_html=True)

    # Stream AI response
    with st.chat_message("assistant", avatar="🧠"):
        try:
            pipeline = st.session_state.pipeline

            # Stream tokens
            answer_placeholder = st.empty()
            response_text = answer_placeholder.write_stream(
                pipeline.stream(
                    query=prompt,
                    chat_history=st.session_state.chat_history,
                    top_k=top_k,
                )
            )
            citation_error = pipeline.get_last_citation_error()
            if citation_error:
                response_text = (
                    "I couldn't provide a verified answer because the response "
                    "did not cite valid retrieved sources. Please try rephrasing "
                    "the question."
                )
                answer_placeholder.empty()
                answer_placeholder.warning(
                    f"⚠️ Answer withheld: {citation_error}"
                )
            sources = pipeline.get_last_sources()
            now_ai = ts()

            st.markdown(f'<div class="msg-timestamp">{now_ai}</div>', unsafe_allow_html=True)

            # Sources inline
            if show_sources and sources:
                with st.expander(f"📎 {len(sources)} source chunk(s) used", expanded=False):
                    for i, src in enumerate(sources, 1):
                        content = src.get("content", "")
                        preview = html.escape(content[:300] + "…" if len(content) > 300 else content)
                        filename = html.escape(str(src.get("filename", st.session_state.doc_name or "Unknown file")))
                        page_number = src.get("page_number")
                        page_label = f"page {page_number}" if page_number is not None else "page not indexed"
                        rerank_score = src.get("rerank_score")
                        score_label = f" · reranker score: {rerank_score:.2f}" if rerank_score is not None else ""
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-num">Source {i} · {filename} · {page_label}{score_label}</div>
                            {preview}
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            response_text = f"❌ Error: {e}"
            sources = []
            now_ai = ts()
            st.error(response_text)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
        "time": now_ai,
    })
    st.session_state.chat_history = (
        st.session_state.chat_history + [(prompt, response_text)]
    )[-10:]

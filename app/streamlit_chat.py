import streamlit as st
import requests
import os
import json

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom dark theme CSS ----------
st.markdown("""
<style>
    /* Overall page background */
    .stApp {
        background-color: #0f0f0f;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2a2a2a;
    }

    /* Main title area */
    .main-header {
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #ececec;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #8e8ea0;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #2a2a2a;
    }

    /* Citation pills */
    .citation-pill {
        display: inline-flex;
        align-items: center;
        background-color: #202020;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px 4px 4px 0;
        font-size: 0.82rem;
        color: #c9c9c9;
    }
    .citation-pill .page-num {
        font-weight: 600;
        color: #6ea8ff;
        margin-right: 6px;
    }
    .citation-pill .score-bar-bg {
        display: inline-block;
        width: 40px;
        height: 5px;
        background-color: #333;
        border-radius: 3px;
        margin-left: 8px;
        overflow: hidden;
    }
    .citation-pill .score-bar-fill {
        display: block;
        height: 100%;
        background-color: #6ea8ff;
        border-radius: 3px;
    }

    /* Sidebar example question buttons */
    div[data-testid="stSidebar"] .stButton button {
        background-color: #202020;
        color: #d0d0d0;
        border: 1px solid #333;
        border-radius: 8px;
        text-align: left;
        font-size: 0.85rem;
        padding: 0.5rem 0.8rem;
        width: 100%;
    }
    div[data-testid="stSidebar"] .stButton button:hover {
        background-color: #2a2a2a;
        border-color: #6ea8ff;
        color: #ffffff;
    }

    /* Stat cards in sidebar */
    .stat-card {
        background-color: #202020;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
    }
    .stat-card .stat-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ececec;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #8e8ea0;
    }
</style>
""", unsafe_allow_html=True)


def render_citations(citations):
    """Renders citations as styled pills. Uses relative rank (not raw
    RRF score, which is a tiny unbounded number and not meaningfully
    a 0-1 confidence) to fill the mini bar -- rank 1 = full bar."""
    max_rank = len(citations)
    html = ""
    for c in citations:
        rank = c["number"]
        pct = int(100 * (max_rank - rank + 1) / max_rank)
        html += f"""
        <span class="citation-pill">
            <span class="page-num">[{c['number']}]</span>
            {c['source_file']}, p.{c['page_number']}
            <span class="score-bar-bg"><span class="score-bar-fill" style="width:{pct}%"></span></span>
        </span>
        """
    st.markdown(html, unsafe_allow_html=True)

def get_health():
    """Fetch corpus stats for the sidebar. Fails quietly if API is down --
    the sidebar shouldn't crash the whole app just because the backend
    isn't running yet."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def ask_stream(question, top_k):
    """Calls the streaming endpoint and returns (full_text, citations)
    while rendering the typing effect live into a placeholder."""
    response = requests.post(
        f"{API_URL}/ask/stream",
        json={"question": question, "top_k": top_k},
        stream=True,
        timeout=60
    )
    response.raise_for_status()

    placeholder = st.empty()
    full_text = ""
    citation_marker_found = False
    buffer = ""

    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if not chunk:
            continue
        buffer += chunk

        if "[[CITATIONS]]" in buffer and not citation_marker_found:
            answer_part, _, citations_json = buffer.partition("[[CITATIONS]]")
            full_text = answer_part.strip()
            placeholder.markdown(full_text)
            citation_marker_found = True
            buffer = citations_json
        elif not citation_marker_found:
            full_text = buffer
            placeholder.markdown(full_text + "▌")

    placeholder.markdown(full_text)

    citations = []
    if citation_marker_found:
        try:
            citations = json.loads(buffer.strip())
        except json.JSONDecodeError:
            citations = []

    return full_text, citations


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📊 Corpus Info")
    health = get_health()
    if health:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{health['chunks_indexed']}</div>
            <div class="stat-label">Chunks indexed</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="stat-card"><div class="stat-value">✅</div><div class="stat-label">API connected</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stat-card"><div class="stat-value">⚠️</div><div class="stat-label">API not reachable</div></div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks retrieved (top-k)", min_value=3, max_value=10, value=5)

    st.markdown("### 💡 Try asking")
    example_questions = [
        "What was Apple's total net sales for fiscal year 2024?",
        "What are Apple's main risk factors?",
        "Who is Apple's CEO?",
        "Does Apple's 10-K mention cybersecurity risks?",
    ]
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main header ----------
st.markdown("""
<div class="main-header">
    <h1>📄 RAG Document Q&A Assistant</h1>
    <p>Ask questions about the indexed document corpus — every answer is grounded with page citations.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Render chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            render_citations(msg["citations"])

# ---------- Handle input (typed or clicked example) ----------
user_question = st.chat_input("Ask a question about the document...")
if st.session_state.pending_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            full_text, citations = ask_stream(user_question, top_k)
            if citations:
                render_citations(citations)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "citations": citations
            })
        except requests.exceptions.ConnectionError:
            error_msg = "⚠️ Could not connect to the API. Make sure the FastAPI server is running (`uvicorn api.main:app --reload`)."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except requests.exceptions.HTTPError as e:
            error_msg = f"⚠️ API returned an error: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
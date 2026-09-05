import streamlit as st
import requests
import os
import json

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A Assistant")
st.caption("Ask questions about the indexed document corpus — answers are grounded with source citations.")

# session_state persists data across Streamlit's re-runs (which happen
# on every interaction). Without this, chat history would reset every
# time you sent a new message, since Streamlit re-executes this whole
# script from top to bottom each time.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**[{c['number']}]** {c['source_file']}, page {c['page_number']} "
                        f"(similarity: {c['score']:.3f})"
                    )

# Chat input box -- this returns the typed text only when the user
# submits, otherwise returns None
user_question = st.chat_input("Ask a question about the document...")

if user_question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Call the FastAPI streaming endpoint
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/ask/stream",
                json={"question": user_question, "top_k": 5},
                stream=True,   # tells `requests` not to buffer the whole response first
                timeout=60
            )
            response.raise_for_status()

            full_text = ""
            citations = []
            placeholder = st.empty()  # a spot in the UI we can keep overwriting as tokens arrive
            citation_marker_found = False
            buffer = ""

            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk

                if "[[CITATIONS]]" in buffer and not citation_marker_found:
                    # Split off the answer text from the citations JSON that follows
                    answer_part, _, citations_json = buffer.partition("[[CITATIONS]]")
                    full_text = answer_part.strip()
                    placeholder.markdown(full_text)
                    citation_marker_found = True
                    buffer = citations_json  # keep accumulating remaining JSON chunks
                elif not citation_marker_found:
                    full_text = buffer
                    placeholder.markdown(full_text + "▌")  # cursor-like effect while streaming

            if citation_marker_found:
                try:
                    citations = json.loads(buffer.strip())
                except json.JSONDecodeError:
                    citations = []

            placeholder.markdown(full_text)  # final render, no cursor

            if citations:
                with st.expander("Sources"):
                    for c in citations:
                        st.markdown(
                            f"**[{c['number']}]** {c['source_file']}, page {c['page_number']} "
                            f"(similarity: {c['score']:.3f})"
                        )

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
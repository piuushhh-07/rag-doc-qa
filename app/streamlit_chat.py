import streamlit as st
import requests
import os
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

    # Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": user_question, "top_k": 5},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                st.markdown(data["answer"])
                if data["citations"]:
                    with st.expander("Sources"):
                        for c in data["citations"]:
                            st.markdown(
                                f"**[{c['number']}]** {c['source_file']}, page {c['page_number']} "
                                f"(similarity: {c['score']:.3f})"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "citations": data["citations"]
                })

            except requests.exceptions.ConnectionError:
                error_msg = "⚠️ Could not connect to the API. Make sure the FastAPI server is running (`uvicorn api.main:app --reload`)."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.HTTPError as e:
                error_msg = f"⚠️ API returned an error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
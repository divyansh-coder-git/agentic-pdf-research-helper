import streamlit as st
from data_ingestion import process_pdf
from graph import build_agent, format_chat_history

st.set_page_config(
    page_title="Chat My PDF",
    page_icon="🤖",
    layout="centered",
)

st.title("Agentic PDF Research Assistant")
st.write("Upload PDF File in the sidebar to start")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if st.button("Reset / Reprocess"):
        st.session_state.vectorstore = None
        st.session_state.agent = None
        st.session_state.chunks = None
        st.session_state.messages = []
        st.rerun()

with st.sidebar:
    if st.session_state.vectorstore is not None:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF.... (chunking + embedding)"):
        vectorstore, chunks = process_pdf(uploaded_file)

        # app = build_agent(vectorstore, chunks)
        # with open("graph.png", "wb") as f:
        #     f.write(app.get_graph().draw_mermaid_png())

        st.session_state.vectorstore = vectorstore
        st.session_state.agent = build_agent(vectorstore, chunks)
        st.session_state.chunks = chunks

        st.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}")

if st.session_state.vectorstore is not None:
    st.success("PDF is successfully converted to a vector base, please enter your query")


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.vectorstore is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("View sources"):
                    for doc in msg["sources"]:
                        page = doc.metadata.get("page", "unknown")
                        st.caption(f"Page {page}")
                        st.text(doc.page_content[:200] + "...")

    user_question = st.chat_input("Ask a question about the PDF...")

    if user_question:
        # Show the user's message immediately
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        chat_history_text = format_chat_history(st.session_state.messages[:-1])
        # Run it through your LangGraph agent
        with st.spinner("Thinking..."):
            result = st.session_state.agent.invoke({"question": user_question, "chat_history": chat_history_text}) # type: ignore
            answer = result["answer"]
            source_docs = result.get("docs", [])

        # Show and store the assistant's reply
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_docs,
        })
        with st.chat_message("assistant"):
            st.write(answer)
            if source_docs:
                with st.expander("View sources"):
                    for doc in source_docs:
                        page = doc.metadata.get("page", "unknown")
                        st.caption(f"Page {page}")
                        st.text(doc.page_content[:200] + "...")
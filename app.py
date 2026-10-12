import streamlit as st
from rag import load_text, chunk_text, build_index, retrieve, answer_question

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Chat with your Document")

# Session state keeps data between Streamlit reruns
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.history = []

# ----- Sidebar: upload & process -----
with st.sidebar:
    st.header("1. Upload document")
    uploaded = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])

    if uploaded and st.button("Process document"):
        with st.status("Processing document...", expanded=True) as status:
            status.write("Reading the document...")
            text = load_text(uploaded)
            if not text.strip():
                status.update(label="No readable text found", state="error")
                st.error("No text found (scanned PDFs need OCR).")
            else:
                status.write("Splitting the document into chunks...")
                chunks = chunk_text(text)
                status.write(
                    "Loading the embedding model and building the search index. "
                    "The first run may download the model."
                )
                index = build_index(chunks)

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.history = []
                st.success(f"Done! {len(chunks)} chunks indexed.")
                status.update(
                    label=f"Document processed: {len(chunks)} chunks indexed",
                    state="complete",
                    expanded=False,
                )

# ----- Main: ask questions -----
if st.session_state.index is None:
    st.info("Upload a document in the sidebar and click **Process document**.")
else:
    for q, a in st.session_state.history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").markdown(a)

    question = st.chat_input("Ask a question about your document...")
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            retrieved = retrieve(question, st.session_state.index, st.session_state.chunks)
            answer = answer_question(question, retrieved)
        st.chat_message("assistant").markdown(answer)

        if retrieved:
            with st.expander("Source passages used"):
                for i, (chunk, score) in enumerate(retrieved, 1):
                    st.markdown(f"**Source {i}** (similarity: {score:.2f})")
                    st.write(chunk)
        else:
            st.info(
                "No sufficiently relevant passages were found. Try rephrasing "
                "the question using terms from the document."
            )

        st.session_state.history.append((question, answer))
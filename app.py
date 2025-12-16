import streamlit as st
import os
import uuid

from src.rag_pipeline import build_vector_store_from_pdfs, load_vector_store
from src.llm_groq import generate_answer

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 Chat with Your Documents (RAG from Scratch)")

# --------------------------------------------------
# Create folders
# --------------------------------------------------
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# --------------------------------------------------
# Initialize session ID (multi-user safe)
# --------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --------------------------------------------------
# Initialize chat memory
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# Sidebar - Upload PDFs
# --------------------------------------------------
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    pdf_paths = []

    for file in uploaded_files:
        file_path = os.path.join("temp_uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        pdf_paths.append(file_path)

    if st.sidebar.button("📥 Process Documents"):
        with st.spinner("Processing documents..."):
            build_vector_store_from_pdfs(
                pdf_paths,
                st.session_state.session_id
            )
        st.sidebar.success("Documents processed successfully!")

# --------------------------------------------------
# Display chat history
# --------------------------------------------------
st.subheader("💬 Conversation")

for chat in st.session_state.chat_history:
    st.markdown(f"**🧑 User:** {chat['question']}")
    st.markdown(f"**🤖 Assistant:** {chat['answer']}")
    st.markdown("---")

# --------------------------------------------------
# Ask question
# --------------------------------------------------
query = st.text_input("Ask a question")

if query:
    try:
        embedder, vector_store = load_vector_store(
            st.session_state.session_id
        )
    except Exception:
        st.error("Please upload and process documents first.")
        st.stop()

    # Retrieve context
    query_embedding = embedder.embed_query(query)
    contexts = vector_store.search(query_embedding, top_k=4)
    context_text = "\n\n".join(contexts)

    # Prepare memory
    recent_history = st.session_state.chat_history[-3:]
    history_text = ""
    for h in recent_history:
        history_text += f"User: {h['question']}\nAssistant: {h['answer']}\n"

    # Generate answer
    with st.spinner("Generating answer..."):
        answer = generate_answer(
            context=f"{history_text}\n\n{context_text}",
            question=query
        )

    # Save chat
    st.session_state.chat_history.append({
        "question": query,
        "answer": answer
    })

    # Display response
    st.markdown(f"**🧑 User:** {query}")
    st.markdown(f"**🤖 Assistant:** {answer}")

    with st.expander("📚 Retrieved Context"):
        for i, ctx in enumerate(contexts):
            st.markdown(f"**Chunk {i+1}**")
            st.write(ctx)



* Built a **multi-user, memory-enabled RAG system** using FAISS and Groq LLaMA-3.1.
* Developed **PDF ingestion, chunking, embedding, and retrieval pipeline** from scratch without LangChain.
* Implemented **session-based chat memory** for conversational context in real-time Q&A.
* Deployed the app on **Streamlit Cloud** with free public access for interactive document querying.
* Enhanced retrieval accuracy using **hybrid FAISS + TF-IDF reranking** for context-aware answers.

## 🔗 Live Demo

[Streamlit App Link](https://vpgn36rutgzatfm3skw5l9.streamlit.app/)



# 📄 RAG from Scratch – PDF Q&A App

A **multi-user, memory-enabled, LangChain-free Retrieval-Augmented Generation (RAG) system** built with **Streamlit**, **FAISS**, **Sentence Transformers**, and **Groq LLaMA-3.1**.
Users can **upload PDFs** and ask questions in a **chat-like interface**, receiving **accurate, context-aware answers**.

---

## 🔹 Features

* **📂 Upload Multiple PDFs:** Users can upload one or more documents at a time.
* **🤖 Real-time Q&A:** Retrieves relevant content from PDFs and generates answers using Groq’s LLaMA-3.1 model.
* **💬 Conversational Memory:** Maintains the last 3–5 interactions for context-aware follow-ups.
* **🔒 Multi-User Safe:** Each session has a unique FAISS vector store, preventing data leakage between users.
* **📚 Retrieved Context:** Users can expand and view the chunks used to generate answers.
* **⚡ Hybrid Retrieval:** FAISS vector search + TF-IDF keyword reranking for higher answer accuracy.
* **🌐 Web-based:** Fully deployed on **Streamlit Community Cloud**, free public URL access.

---

## 🛠 Tech Stack

* **Frontend / Interface:** Streamlit
* **PDF Processing:** PyPDF2 / PyPDF
* **Embeddings:** Sentence Transformers (MiniLM)
* **Vector Store:** FAISS (vector similarity search)
* **LLM:** Groq LLaMA-3.1
* **Environment Management:** Python, `.env` for API keys

---

## 📂 Project Structure

```
rag_from_scratch/
│
├── temp_uploads/       # runtime folder for uploaded PDFs (ignored in Git)
├── embeddings/         # session-specific FAISS stores (ignored in Git)
├── src/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── llm_groq.py
├── app.py              # Streamlit app
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  How it Works

1. **PDF Upload:** User uploads documents via sidebar.
2. **Text Extraction:** Extract text from PDFs using `PyPDF`.
3. **Chunking:** Text is split into manageable chunks (300–500 tokens).
4. **Embedding:** Chunks are converted to embeddings using MiniLM.
5. **Vector Store:** Embeddings are stored in **FAISS** for fast similarity search.
6. **Query:** User enters a question.
7. **Retrieval:** Retrieve top chunks using FAISS + TF-IDF keyword reranking.
8. **Answer Generation:** Groq LLaMA-3.1 generates answer using retrieved chunks + chat history.
9. **Chat Memory:** Stores last 3–5 Q&A pairs per session for follow-up questions.

---

## 🚀 Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/Nupur-lab/RAG.git
cd RAG
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set Groq API Key**

* Create `.env` (local testing):

```env
GROQ_API_KEY=your_groq_api_key_here
```

4. **Run locally**

```bash
streamlit run app.py
```

5. **Access via web** (after deploying to Streamlit Cloud)
   Public URL: 'https://vpgn36rutgzatfm3skw5l9.streamlit.app/'

---

## 📌 Notes

* FAISS vector stores are **per session** to ensure multi-user safety.
* Only the **top relevant chunks** are passed to the LLM to reduce hallucinations.
* Supports **multiple PDF uploads** simultaneously.
* Recommended **PDF size < 10MB per file** for free tier performance.

---

## 💡 Future Enhancements

* Add **image / table parsing** for richer document types.
* Integrate **summary previews** for uploaded PDFs.
* Add **feedback mechanism** to improve answer relevance.
* Use **hybrid dense + sparse search** for even higher accuracy.






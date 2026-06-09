# 📚 Universal Document AI Librarian (Conversational RAG)

A production-grade, end-to-end Conversational Retrieval-Augmented Generation (RAG) system built with **Python**, **Streamlit**, and the **Gemini 2.5** ecosystem. This application allows users to upload any PDF document, securely index its contents into an ephemeral in-memory vector space, and have an interactive, contextual conversation with the document.

---

## 🚀 Features

- **Dynamic PDF Ingestion**: Automatically processes uploaded PDFs, extracts text page-by-page, and splits text using recursive character token boundaries.
- **Context-Aware Query Rewriting**: Utilizes an LLM-based query rewriter to translate ambiguous, conversational follow-up questions (e.g., *"Was that higher than last year?"*) into standalone vector search queries based on the deep chat history.
- **Strict Guardrails & Zero Hallucinations**: Prompt-engineered system instructions that anchor the generative model entirely to the extracted document facts.
- **Isolated In-Memory Vector Storage**: Uses thread-safe, session-isolated memory vector stores to ensure strict multi-user privacy. Data vanishes when the session ends.

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Orchestration**: LangChain Core
- **LLM Engine**: Google Gemini 2.5 Flash (`gemini-2.5-flash`)
- **Embeddings**: Google Gemini 2 Embeddings (`gemini-embedding-2-preview` - 3072 dimensions)
- **PDF Processing**: pypdf

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/financial-rag-ai.git](https://github.com/YOUR_GITHUB_USERNAME/financial-rag-ai.git)
cd financial-rag-ai
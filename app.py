import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores.in_memory import InMemoryVectorStore

load_dotenv()

st.set_page_config(page_title="Universal AI Librarian", page_icon="📚", layout="wide")
st.title("📚 Universal Document AI Librarian")
st.markdown("Upload any PDF and chat with it securely. Your data stays in your session memory.")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ... (the rest of your code stays exactly the same) ...

# 2. Session Memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

# Sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        if st.session_state.vectorstore is None:
            with st.spinner("Processing and indexing document..."):
                pdf_reader = PdfReader(uploaded_file)
                documents = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(Document(page_content=text, metadata={"page": page_num + 1}))
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
                chunks = text_splitter.split_documents(documents)
                
                st.session_state.vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
                
                # Save analytics
                st.session_state.doc_stats = {
                    "pages": len(pdf_reader.pages),
                    "chunks": len(chunks)
                }
                st.session_state.messages = []
                st.success("Indexing complete!")
    
    # Dashboard
    if st.session_state.vectorstore is not None:
        st.divider()
        st.subheader("📊 Document Analytics")
        st.metric("Total Pages", st.session_state.doc_stats.get("pages", 0))
        st.metric("Vector Chunks", st.session_state.doc_stats.get("chunks", 0))
        st.metric("Embedding Dimensions", "3072")
        st.caption("Powered by Gemini-2-Preview Dense Vectors")

#  Chat 
if st.session_state.vectorstore is None:
    st.info("👈 Please upload a PDF file in the sidebar to begin chatting.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask something about your uploaded document..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            # Use Streamlit's status dropdown for a cleaner UI during thinking
            with st.status("Analyzing document...", expanded=False) as status:
                search_query = user_query
                history_string = ""
                for msg in st.session_state.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "AI"
                    history_string += f"{role}: {msg['content']}\n"

                if len(st.session_state.messages) > 1:
                    status.update(label="Rewriting query based on history...")
                    rewriter_prompt = f"""
                    Given the chat history and follow-up question, rewrite it into a descriptive standalone search query.
                    Output ONLY the query.
                    CHAT HISTORY:\n{history_string}\nFOLLOW-UP QUESTION: {user_query}\nSEARCH QUERY:
                    """
                    search_query = llm.invoke(rewriter_prompt).content.strip()

                status.update(label="Searching vector space...")
                matching_chunks = st.session_state.vectorstore.similarity_search(search_query, k=3)
                context_text = "\n\n".join([f"[Page {doc.metadata.get('page')}]: {doc.page_content}" for doc in matching_chunks])
                
                status.update(label="Context retrieved!", state="complete")

            
            final_prompt = f"""
            You are a precise document assistant. Answer using ONLY the context provided below. 
            If the answer isn't explicitly there, say "I cannot find the answer in the provided document."
            CRITICAL: You must explicitly cite the exact [Page X] for the facts you provide, based on the context text.

            CHAT HISTORY:\n{history_string}
            CONTEXT:\n{context_text}
            QUESTION: {user_query}
            """
            
            def stream_parser(stream):
                for chunk in stream:
                    yield chunk.content
                    
            response_stream = llm.stream(final_prompt)
            ai_answer = st.write_stream(stream_parser(response_stream))
            
        st.session_state.messages.append({"role": "assistant", "content": ai_answer})
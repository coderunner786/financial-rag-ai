import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores.in_memory import InMemoryVectorStore

st.set_page_config(page_title="Universal AI Librarian", page_icon="📚", layout="wide")
st.title("📚 Universal Document AI Librarian")
st.markdown("Upload any PDF and chat with it securely. Your data stays in your session memory.")

# 1. Setup the Brains
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Setup Session Memory Architecture
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 3. Sidebar for File Uploads
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        # Check if we've already processed this specific file
        if st.session_state.vectorstore is None:
            with st.spinner("Processing and indexing document..."):
                # Read bytes from the browser upload
                pdf_reader = PdfReader(uploaded_file)
                documents = []
                
                # Extract text page by page
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(Document(page_content=text, metadata={"page": page_num + 1}))
                
                # Chunk it up
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
                chunks = text_splitter.split_documents(documents)
                
                # Build an instant, private in-memory database just for this user
                st.session_state.vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
                st.success(f"Successfully loaded {len(chunks)} text blocks into memory!")
                
                # Clear past conversation if a new file is uploaded
                st.session_state.messages = []

# 4. Main Chat Interface
if st.session_state.vectorstore is None:
    st.info("👈 Please upload a PDF file in the sidebar to begin chatting.")
else:
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box
    if user_query := st.chat_input("Ask something about your uploaded document..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Reading document context..."):
                
                search_query = user_query
                history_string = ""
                for msg in st.session_state.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "AI"
                    history_string += f"{role}: {msg['content']}\n"

                # Query Rewriter (for multi-turn conversational accuracy)
                if len(st.session_state.messages) > 1:
                    rewriter_prompt = f"""
                    Given the chat history and follow-up question, rewrite it into a descriptive standalone search query.
                    Output ONLY the query.
                    CHAT HISTORY:\n{history_string}\nFOLLOW-UP QUESTION: {user_query}\nSEARCH QUERY:
                    """
                    rewrite_response = llm.invoke(rewriter_prompt)
                    search_query = rewrite_response.content.strip()

                # Search our ephemeral session database
                matching_chunks = st.session_state.vectorstore.similarity_search(search_query, k=3)
                context_text = "\n\n".join([f"[Page {doc.metadata.get('page')}]: {doc.page_content}" for doc in matching_chunks])

                # Final Answer Build
                final_prompt = f"""
                You are a precise document assistant. Answer using ONLY the context provided below. 
                If the answer isn't explicitly there, say "I cannot find the answer in the provided document."
                Cite the page number if applicable.

                CHAT HISTORY:\n{history_string}
                CONTEXT:\n{context_text}
                QUESTION: {user_query}
                """
                
                response = llm.invoke(final_prompt)
                ai_answer = response.content
                st.markdown(ai_answer)
                
        st.session_state.messages.append({"role": "assistant", "content": ai_answer})
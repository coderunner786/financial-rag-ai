import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print(" Waking up the Advanced Gemini Librarian...")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
index_name = "gemini-rag"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

chat_history = []

print(" Advanced System Ready! Query Rewriter is live.")
print("-" * 50)

while True:
    query = input("\n👤 You: ")
    if query.lower() in ['quit', 'exit', 'q']:
        print("Shutting down...")
        break
    
    search_query = query
    
    if len(chat_history) > 0:
        print(" Rewriting query based on conversation history...")
        
        # Format history for the rewriter
        history_string = ""
        for turn in chat_history:
            history_string += f"User: {turn['user']}\nAI: {turn['ai']}\n"
            
        rewriter_prompt = f"""
        Given the following chat history and a follow-up question, rewrite the follow-up question into a standalone, descriptive search query that can be used to search a financial vector database. 
        DO NOT answer the question. Just output the rewritten search query and nothing else.

        CHAT HISTORY:
        {history_string}

        FOLLOW-UP QUESTION:
        {query}

        STANDALONE SEARCH QUERY:
        """
        
        rewrite_response = llm.invoke(rewriter_prompt)
        search_query = rewrite_response.content.strip()
        print(f" Optimized Search Query for Pinecone: '{search_query}'")

    print(" Searching the Pinecone vault...")
    matching_chunks = vectorstore.similarity_search(search_query, k=3)
    context_text = "\n\n".join([doc.page_content for doc in matching_chunks])

    final_history_string = ""
    for turn in chat_history:
        final_history_string += f"User: {turn['user']}\nAI: {turn['ai']}\n"

    prompt = f"""
    You are an expert financial analyst. Answer the user's question using ONLY the context provided below. 
    Take the Chat History into account to maintain conversation continuity.
    If the answer is not in the context, do not guess. Just say "I cannot find the answer in the provided report."

    CHAT HISTORY:
    {final_history_string}

    CONTEXT:
    {context_text}

    QUESTION:
    {query}
    """

    print(" Reading the context and formulating final response...")
    response = llm.invoke(prompt)
    ai_answer = response.content
    
    print(f"\n AI: {ai_answer}")
    print("-" * 50)

    chat_history.append({"user": query, "ai": ai_answer})
import streamlit as st
import os
import tempfile
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'search_tool' not in st.session_state:
        st.session_state.search_tool = None

# Function to load PDF documents
def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Function to split documents
def split_documents(docs, chunk_size=600, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# Function to create embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Function to create FAISS vector store
def create_vector_store(splitted_docs, embedding_model):
    return FAISS.from_documents(splitted_docs, embedding_model)

# Function to create the LLM
def get_llm(api_key: str):
    return ChatGroq(
        model="gemma2-9b-it",
        api_key=api_key
    )

# Initialize search tool
def get_search_tool():
    try:
        search = DuckDuckGoSearchRun()
        return Tool(
            name="web_search",
            description="Search the web for current information",
            func=search.run
        )
    except Exception as e:
        return None

# Web search function using LangChain
def web_search_langchain(query: str, max_results: int = 3) -> str:
    try:
        if st.session_state.search_tool is None:
            st.session_state.search_tool = get_search_tool()
        
        if st.session_state.search_tool:
            results = st.session_state.search_tool.func(query)
            return results[:1000]
        else:
            return ""
    except Exception as e:
        return ""

# Enhanced prompt templates for different modes
def get_rag_prompt():
    return PromptTemplate.from_template(
        """You are an AI assistant. Use the document context to answer the question accurately.

Context: {context}
Question: {question}
Chat History: {chat_history}

Answer based on the context provided:"""
    )

def get_search_prompt():
    return PromptTemplate.from_template(
        """You are an AI assistant with web search capabilities. Use the search results to provide current information.

Search Results: {web_results}
Question: {question}
Chat History: {chat_history}

Answer based on the search results:"""
    )

def get_hybrid_prompt():
    return PromptTemplate.from_template(
        """You are an AI assistant with access to documents and web search. Provide comprehensive answers using both sources.

Document Context: {context}
Web Search Results: {web_results}
Question: {question}
Chat History: {chat_history}

Answer using both document context and search results:"""
    )

# Format chat history
def format_chat_history(chat_history: List[Dict]) -> str:
    if not chat_history:
        return "No previous conversation"
    
    formatted = []
    for chat in chat_history[-3:]:  # Last 3 exchanges
        formatted.append(f"User: {chat['user'][:100]}...")
        formatted.append(f"Assistant: {chat['assistant'][:100]}...")
    
    return "\n".join(formatted)

# Build RAG chain based on mode
def build_rag_chain(vector_store, llm, mode="rag"):
    if vector_store is None and mode in ["rag", "hybrid"]:
        retriever = None
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store else None

    def process_query(inputs):
        question = inputs["question"]
        context = ""
        web_results = ""
        chat_history = format_chat_history(st.session_state.chat_history)
        
        if mode in ["rag", "hybrid"] and retriever:
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
        
        if mode in ["search", "hybrid"]:
            web_results = web_search_langchain(question)
        
        return {
            "context": context,
            "web_results": web_results,
            "question": question,
            "chat_history": chat_history
        }

    # Select prompt based on mode
    if mode == "rag":
        prompt = get_rag_prompt()
    elif mode == "search":
        prompt = get_search_prompt()
    else:  # hybrid
        prompt = get_hybrid_prompt()
    
    rag_chain = (
        RunnableLambda(process_query)
        | prompt
        | llm
    )
    
    return rag_chain

# Process uploaded PDF
def process_pdf(uploaded_file, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        docs = load_documents(tmp_file_path)
        splitted_docs = split_documents(docs, chunk_size, chunk_overlap)
        
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = get_embedding_model()
        
        vector_store = create_vector_store(splitted_docs, st.session_state.embedding_model)
        st.session_state.vector_store = vector_store
        st.session_state.documents_loaded = True
        
        return len(docs), len(splitted_docs)
    finally:
        os.unlink(tmp_file_path)

# Main chatbot function
def chatbot_response(user_input: str, groq_api_key: str, mode: str = "rag"):
    if not groq_api_key:
        return "Please provide your Groq API key."
    
    try:
        llm = get_llm(groq_api_key)
        rag_chain = build_rag_chain(st.session_state.vector_store, llm, mode)
        
        result = rag_chain.invoke({"question": user_input})
        return result.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.markdown("# 🤖 AI Chat Assistant")
    
    # Top controls in columns
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        groq_api_key = st.text_input("🔑 API Key", type="password", placeholder="Enter Groq API key")
    
    with col2:
        mode = st.selectbox("Mode", ["RAG Only", "Search Only", "Hybrid"], index=0)
        mode_map = {"RAG Only": "rag", "Search Only": "search", "Hybrid": "hybrid"}
        selected_mode = mode_map[mode]
    
    with col3:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col4:
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **Modes:**
            - **RAG Only**: Answer from uploaded documents
            - **Search Only**: Answer from web search
            - **Hybrid**: Combine documents + web search
            
            **Usage:**
            1. Enter API key
            2. Upload PDF (for RAG/Hybrid)
            3. Select mode
            4. Start chatting
            """)
    
    # Document upload section
    if selected_mode in ["rag", "hybrid"]:
        with st.expander("📁 Document Upload", expanded=not st.session_state.documents_loaded):
            upload_col1, upload_col2 = st.columns([3, 1])
            
            with upload_col1:
                uploaded_file = st.file_uploader("Choose PDF", type="pdf", label_visibility="collapsed")
            
            with upload_col2:
                if uploaded_file:
                    if st.button("Process", type="primary"):
                        with st.spinner("Processing..."):
                            try:
                                num_docs, num_chunks = process_pdf(uploaded_file, 600, 150)
                                st.success(f"✅ {num_docs} pages → {num_chunks} chunks")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
    
    # Status bar
    status_cols = st.columns(3)
    with status_cols[0]:
        doc_status = "✅ Ready" if st.session_state.documents_loaded else "⚪ No docs"
        st.caption(f"📄 Documents: {doc_status}")
    
    with status_cols[1]:
        if st.session_state.search_tool is None and selected_mode in ["search", "hybrid"]:
            st.session_state.search_tool = get_search_tool()
        search_status = "✅ Ready" if st.session_state.search_tool else "⚪ Inactive"
        st.caption(f"🔍 Search: {search_status}")
    
    with status_cols[2]:
        st.caption(f"💬 Messages: {len(st.session_state.chat_history)}")
    
    st.divider()
    
    # Chat container
    chat_container = st.container(height=500)
    
    with chat_container:
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["user"])
                with st.chat_message("assistant"):
                    st.write(chat["assistant"])
        else:
            st.info("Start a conversation by typing a message below!")
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Validate inputs
        if not groq_api_key:
            st.error("Please enter your Groq API key")
            return
        
        if selected_mode in ["rag", "hybrid"] and not st.session_state.documents_loaded:
            st.warning("Upload a document for RAG/Hybrid mode")
            return
        
        # Add user message
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": "Thinking...",
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate response
        with st.spinner(f"Processing with {mode} mode..."):
            response = chatbot_response(user_input, groq_api_key, selected_mode)
        
        # Update last message with response
        st.session_state.chat_history[-1]["assistant"] = response
        
        st.rerun()

if __name__ == "__main__":
    main()
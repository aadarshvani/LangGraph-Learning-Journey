import streamlit as st
import os
import tempfile
from datetime import datetime
from typing import List, Dict

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'chat_history': [],
        'vector_store': None,
        'embedding_model': None,
        'documents_loaded': False,
        'search_tool': None,
        'processed_docs_info': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================================================
# DOCUMENT PROCESSING MODULE
# =============================================================================

class DocumentProcessor:
    @staticmethod
    def load_pdf(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    @staticmethod
    def split_documents(docs, chunk_size=600, chunk_overlap=150):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)
    
    @staticmethod
    def get_embedding_model():
        return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    @staticmethod
    def create_vector_store(splitted_docs, embedding_model):
        return FAISS.from_documents(splitted_docs, embedding_model)
    
    @staticmethod
    def process_uploaded_pdf(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        docs = DocumentProcessor.load_pdf(tmp_file_path)
        splitted_docs = DocumentProcessor.split_documents(docs)
        
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = DocumentProcessor.get_embedding_model()
        
        vector_store = DocumentProcessor.create_vector_store(splitted_docs, st.session_state.embedding_model)
        st.session_state.vector_store = vector_store
        st.session_state.documents_loaded = True
        st.session_state.processed_docs_info = {
            'filename': uploaded_file.name,
            'pages': len(docs),
            'chunks': len(splitted_docs)
        }
        
        os.unlink(tmp_file_path)
        return len(docs), len(splitted_docs)

# =============================================================================
# SEARCH MODULE
# =============================================================================

class SearchManager:
    @staticmethod
    def get_search_tool():
        search = DuckDuckGoSearchRun()
        return Tool(
            name="web_search",
            description="Search the web for current information",
            func=search.run
        )
    
    @staticmethod
    def web_search(query: str) -> str:
        if st.session_state.search_tool is None:
            st.session_state.search_tool = SearchManager.get_search_tool()
        
        results = st.session_state.search_tool.func(query)
        return results[:1000]

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class PromptManager:
    @staticmethod
    def get_rag_prompt():
        return PromptTemplate.from_template(
            """You are an AI assistant. Use the document context to answer the question accurately.

Context: {context}
Question: {question}
Chat History: {chat_history}

Answer based on the context provided:"""
        )
    
    @staticmethod
    def get_search_prompt():
        return PromptTemplate.from_template(
            """You are an AI assistant with web search capabilities. Use the search results to provide current information.

Search Results: {web_results}
Question: {question}
Chat History: {chat_history}

Answer based on the search results:"""
        )
    
    @staticmethod
    def get_hybrid_prompt():
        return PromptTemplate.from_template(
            """You are an AI assistant with access to documents and web search. Provide comprehensive answers using both sources.

Document Context: {context}
Web Search Results: {web_results}
Question: {question}
Chat History: {chat_history}

Answer using both document context and search results:"""
        )

# =============================================================================
# CHAT ENGINE
# =============================================================================

class ChatEngine:
    @staticmethod
    def get_llm(api_key: str):
        return ChatGroq(model="gemma2-9b-it", api_key=api_key)
    
    @staticmethod
    def format_chat_history(chat_history: List[Dict]) -> str:
        if not chat_history:
            return "No previous conversation"
        
        formatted = []
        for chat in chat_history[-3:]:
            formatted.append(f"User: {chat['user'][:100]}...")
            formatted.append(f"Assistant: {chat['assistant'][:100]}...")
        
        return "\n".join(formatted)
    
    @staticmethod
    def build_rag_chain(vector_store, llm, mode="rag"):
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store else None

        def process_query(inputs):
            question = inputs["question"]
            context = ""
            web_results = ""
            chat_history = ChatEngine.format_chat_history(st.session_state.chat_history)
            
            if mode in ["rag", "hybrid"] and retriever:
                docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs])
            
            if mode in ["search", "hybrid"]:
                web_results = SearchManager.web_search(question)
            
            return {
                "context": context,
                "web_results": web_results,
                "question": question,
                "chat_history": chat_history
            }

        # Select prompt based on mode
        if mode == "rag":
            prompt = PromptManager.get_rag_prompt()
        elif mode == "search":
            prompt = PromptManager.get_search_prompt()
        else:
            prompt = PromptManager.get_hybrid_prompt()
        
        rag_chain = (
            RunnableLambda(process_query)
            | prompt
            | llm
        )
        
        return rag_chain
    
    @staticmethod
    def get_response(user_input: str, groq_api_key: str, mode: str = "rag"):
        llm = ChatEngine.get_llm(groq_api_key)
        rag_chain = ChatEngine.build_rag_chain(st.session_state.vector_store, llm, mode)
        result = rag_chain.invoke({"question": user_input})
        return result.content

# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIComponents:
    @staticmethod
    def render_sidebar():
        """Render sidebar with API key and document upload"""
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # API Key Section
            st.subheader("🔑 API Settings")
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="Enter your Groq API key",
                help="Get your API key from Groq Console"
            )
            
            st.divider()
            
            # Document Upload Section
            st.subheader("📁 Document Upload")
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type="pdf",
                help="Upload a PDF file to enable RAG functionality"
            )
            
            if uploaded_file:
                if st.button("📤 Process Document", type="primary", use_container_width=True):
                    with st.spinner("Processing document..."):
                        DocumentProcessor.process_uploaded_pdf(uploaded_file)
                    st.success("Document processed successfully!")
                    st.rerun()
            
            # Document Status
            if st.session_state.documents_loaded and st.session_state.processed_docs_info:
                info = st.session_state.processed_docs_info
                st.success(f"✅ **{info['filename']}**")
                st.caption(f"📄 {info['pages']} pages • 🔤 {info['chunks']} chunks")
            
            st.divider()
            
            # Stats
            st.subheader("📊 Session Stats")
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("💬 Messages", len(st.session_state.chat_history))
            with stats_col2:
                doc_status = "✅" if st.session_state.documents_loaded else "❌"
                st.metric("📄 Documents", doc_status)
            
            st.markdown("")
            
            # Clear Chat Button
            if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
        
        return groq_api_key, uploaded_file
    
    @staticmethod
    def render_mode_selector():
        """Render mode selection at bottom"""
        st.markdown("---")
        
        # Center the mode selector
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown("##### 🔄 Chat Mode")
            mode = st.radio(
                "Select how you want to interact:",
                ["RAG Only", "Search Only", "Hybrid"],
                index=0,
                horizontal=True,
                help="• **RAG Only**: Answer from uploaded documents\n• **Search Only**: Answer from web search\n• **Hybrid**: Combine both sources",
                label_visibility="collapsed"
            )
            
            # Add mode descriptions
            mode_descriptions = {
                "RAG Only": "📄 Using uploaded document knowledge",
                "Search Only": "🌐 Using real-time web search",
                "Hybrid": "🔄 Using both documents and web search"
            }
            st.caption(mode_descriptions[mode])
        
        mode_map = {"RAG Only": "rag", "Search Only": "search", "Hybrid": "hybrid"}
        return mode_map[mode]
    
    @staticmethod
    def render_chat_interface():
        """Render main chat interface"""
        st.title("💬 AI Chat Assistant")
        st.markdown("*Ask questions about your documents or search the web for information*")
        
        # Add some spacing
        st.markdown("")
        
        # Chat container with better styling
        chat_container = st.container(height=450, border=True)
        
        with chat_container:
            if st.session_state.chat_history:
                for i, chat in enumerate(st.session_state.chat_history):
                    with st.chat_message("user"):
                        st.markdown(chat["user"])
                    with st.chat_message("assistant"):
                        st.markdown(chat["assistant"])
            else:
                # Welcome message with better styling
                st.markdown("""
                <div style='text-align: center; padding: 50px 20px; color: #666;'>
                    <h3>👋 Welcome to AI Chat Assistant!</h3>
                    <p>Get started by:</p>
                    <ul style='list-style: none; padding: 0;'>
                        <li>🔑 Adding your API key in the sidebar</li>
                        <li>📄 Uploading a PDF document (optional)</li>
                        <li>💬 Typing your question below</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        return chat_container
    
    @staticmethod
    def validate_inputs(groq_api_key: str, selected_mode: str) -> tuple:
        """Validate user inputs and return validation status"""
        if not groq_api_key:
            return False, "⚠️ Please enter your Groq API key in the sidebar"
        
        if selected_mode in ["rag", "hybrid"] and not st.session_state.documents_loaded:
            return False, "⚠️ Please upload and process a document for RAG/Hybrid mode"
        
        return True, ""

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Render sidebar
    groq_api_key, uploaded_file = UIComponents.render_sidebar()
    
    # Main content area
    UIComponents.render_chat_interface()
    
    # Chat input - positioned before mode selector for better flow
    user_input = st.chat_input("💭 Type your message here...", key="chat_input")
    
    # Mode selector at bottom with better spacing
    selected_mode = UIComponents.render_mode_selector()
    
    # Add some bottom spacing
    st.markdown("")
    st.markdown("")
    
    if user_input:
        # Validate inputs
        is_valid, error_message = UIComponents.validate_inputs(groq_api_key, selected_mode)
        
        if not is_valid:
            st.error(error_message)
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": "Thinking...",
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate response
        with st.spinner(f"🤔 Processing with {selected_mode.upper()} mode..."):
            response = ChatEngine.get_response(user_input, groq_api_key, selected_mode)
        
        # Update chat history with response
        st.session_state.chat_history[-1]["assistant"] = response
        
        st.rerun()

if __name__ == "__main__":
    main()
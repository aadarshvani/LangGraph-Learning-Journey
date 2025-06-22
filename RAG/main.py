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
# CUSTOM CSS STYLES - MODE SELECTOR ONLY
# =============================================================================

def load_custom_css():
    st.markdown("""
    <style>
    /* Mode selector styling */
    .mode-selector {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .mode-header {
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    
    .mode-header h3 {
        margin: 0;
        font-size: 1.8em;
        font-weight: 600;
    }
    
    .mode-options {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
    }
    
    .mode-card {
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        min-width: 150px;
        flex: 1;
        max-width: 200px;
    }
    
    .mode-card:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .mode-card.active {
        background: rgba(255,255,255,0.4);
        border-color: rgba(255,255,255,0.6);
        transform: translateY(-2px);
    }
    
    .mode-icon {
        font-size: 2.5em;
        margin-bottom: 10px;
        display: block;
    }
    
    .mode-title {
        font-weight: 600;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
    
    .mode-desc {
        font-size: 0.85em;
        opacity: 0.9;
        line-height: 1.3;
    }
    
    /* Current mode indicator */
    .current-mode-indicator {
        text-align: center; 
        margin: 15px 0; 
        padding: 10px; 
        background: rgba(255,255,255,0.1); 
        border-radius: 10px; 
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

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
        'processed_docs_info': None,
        'selected_mode': 'rag'
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
        """Render sidebar with Streamlit components only"""
        with st.sidebar:
            st.header("🤖 AI Assistant")
            st.caption("Configuration Panel")
            
            # API Key Section
            st.subheader("🔑 API Configuration")
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="Enter your Groq API key...",
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
                if st.button("📤 Process Document", key="process_btn", use_container_width=True):
                    with st.spinner("🔄 Processing document..."):
                        DocumentProcessor.process_uploaded_pdf(uploaded_file)
                    st.rerun()
            
            # Document Status
            if st.session_state.documents_loaded and st.session_state.processed_docs_info:
                info = st.session_state.processed_docs_info
                st.success(f"✅ **{info['filename']}**  \n📄 {info['pages']} pages • 🔤 {info['chunks']} chunks")
            
            st.divider()
            
            # Session Statistics
            st.subheader("📊 Session Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💬 Messages", len(st.session_state.chat_history))
            with col2:
                st.metric("📄 Documents", "✅" if st.session_state.documents_loaded else "❌")
            

            
            st.divider()
            
            # Clear Chat Button
            if st.button("🗑️ Clear Chat History", key="clear_btn", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        return groq_api_key, uploaded_file
    
    @staticmethod
    def render_chat_interface():
        """Render chat interface with Streamlit components only"""
        st.title("💬 AI Chat Assistant")
        st.caption("Ask questions about your documents or search the web for information")
        
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(chat["user"])
                with st.chat_message("assistant"):
                    st.markdown(chat["assistant"])
        else:
            st.info("👋 **Welcome to AI Chat Assistant!**  \nYour intelligent companion for document analysis and web search")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🔑 Add API Key**  \nEnter your Groq API key in the sidebar")
            with col2:
                st.markdown("**📄 Upload Document**  \nOptional: Upload a PDF for document-based queries")
            with col3:
                st.markdown("**💬 Start Chatting**  \nType your question and select your preferred mode")
    
    @staticmethod
    def render_mode_selector():
        """Render enhanced mode selector with custom styling (HTML/CSS preserved)"""

        
        # Create columns for mode selection
        col1, col2, col3 = st.columns(3)
        
        modes = [
            {"key": "rag", "title": "RAG Only", "icon": "📄", "desc": "Document-based responses"},
            {"key": "search", "title": "Search Only", "icon": "🔍", "desc": "Web search responses"},
            {"key": "hybrid", "title": "Hybrid", "icon": "🔄", "desc": "Combined approach"}
        ]
        
        cols = [col1, col2, col3]
        
        for i, mode in enumerate(modes):
            with cols[i]:
                if st.button(
                    f"{mode['icon']} {mode['title']}\n{mode['desc']}", 
                    key=f"mode_{mode['key']}",
                    use_container_width=True
                ):
                    st.session_state.selected_mode = mode['key']
        
        # Display current mode
        current_mode = st.session_state.selected_mode
        mode_names = {"rag": "RAG Only", "search": "Search Only", "hybrid": "Hybrid"}
        mode_descs = {
            "rag": "📄 Using uploaded document knowledge",
            "search": "🔍 Using real-time web search",
            "hybrid": "🔄 Using both documents and web search"
        }
        
        st.markdown(f"""
        <div class="current-mode-indicator">
            <strong>Current Mode: {mode_names[current_mode]}</strong><br>
            <small>{mode_descs[current_mode]}</small>
        </div>
        """, unsafe_allow_html=True)
        
        return current_mode
    
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
    
    # Load custom CSS (mode selector only)
    load_custom_css()
    
    initialize_session_state()
    
    # Render sidebar
    groq_api_key, uploaded_file = UIComponents.render_sidebar()
    
    # Main content area
    UIComponents.render_chat_interface()
    
    # Chat input
    user_input = st.chat_input("💭 Type your message here...", key="main_chat_input")
    
    # Mode selector (with custom HTML/CSS)
    selected_mode = UIComponents.render_mode_selector()
    
    if user_input:
        # Validate inputs
        is_valid, error_message = UIComponents.validate_inputs(groq_api_key, selected_mode)
        
        if not is_valid:
            st.error(error_message)
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": "🤔 Thinking...",
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate response
        with st.spinner(f"🚀 Processing with {selected_mode.upper()} mode..."):
            response = ChatEngine.get_response(user_input, groq_api_key, selected_mode)
        
        # Update chat history with response
        st.session_state.chat_history[-1]["assistant"] = response
        
        st.rerun()

if __name__ == "__main__":
    main()
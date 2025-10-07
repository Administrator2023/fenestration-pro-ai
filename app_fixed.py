import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Core imports for basic RAG
try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.chat_models import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

import openai
from streamlit_option_menu import option_menu

# Page config with enhanced settings
st.set_page_config(
    page_title="Fenestration Pro AI - SOTA Edition",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/administrator2023/fenestration-pro-ai',
        'Report a bug': "https://github.com/administrator2023/fenestration-pro-ai/issues",
        'About': "State-of-the-art AI for fenestration professionals"
    }
)

# Enhanced CSS with modern design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.sota-badge {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-left: 1rem;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: 2rem;
}

.assistant-message {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-right: 2rem;
}

.processing-status {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.sidebar-section {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🚀 Welcome to Fenestration Pro AI - State-of-the-Art Edition! I'm here to help answer your questions about windows, doors, and building envelope systems. Upload a PDF or ask me anything!",
            "timestamp": datetime.now()
        })
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "avg_response_time": 0
        }

initialize_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏗️ Fenestration Pro AI <span class="sota-badge">SOTA</span></h1>
    <p>Advanced RAG-Powered Document Intelligence System</p>
</div>
""", unsafe_allow_html=True)

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["💬 Chat", "📊 Analytics", "🔧 Settings"],
    icons=["chat-dots", "graph-up", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#667eea", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#667eea"},
    }
)

# Sidebar with enhanced features
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("⚙️ AI Configuration")
    
    # API Key management
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your key from platform.openai.com"
        )
    else:
        st.success("✅ API Key loaded from secrets")
    
    # Model selection
    model_choice = st.selectbox(
        "🤖 AI Model",
        ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
        help="Choose your preferred AI model"
    )
    
    # Advanced settings
    with st.expander("🔬 Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 2000, 800, 100)
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document processing section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("📄 Document Processing")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDFs for analysis"
    )
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} file(s) uploaded!")
        
        if st.button("🔄 Process Documents", type="primary"):
            if LANGCHAIN_AVAILABLE:
                process_documents(uploaded_files, api_key, chunk_size, chunk_overlap)
            else:
                st.error("LangChain not available. Please install required dependencies.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("📊 Statistics")
    
    stats = st.session_state.document_stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", stats["total_documents"])
        st.metric("Queries", stats["total_queries"])
    with col2:
        st.metric("Chunks", stats["total_chunks"])
        st.metric("Avg Response", f"{stats['avg_response_time']:.1f}s")
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_documents(uploaded_files, api_key, chunk_size, chunk_overlap):
    """Process uploaded documents with RAG pipeline"""
    if not api_key:
        st.error("Please provide an OpenAI API key")
        return
    
    with st.spinner("🔄 Processing documents with advanced RAG pipeline..."):
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            documents = []
            
            # Save and load documents
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load document
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                documents.extend(docs)
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            
            # Create conversation chain
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=model_choice,
                temperature=temperature
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                memory=memory,
                return_source_documents=True
            )
            
            # Update session state
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation_chain = conversation_chain
            st.session_state.document_stats.update({
                "total_documents": len(uploaded_files),
                "total_chunks": len(chunks)
            })
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            st.success(f"✅ Processed {len(uploaded_files)} documents into {len(chunks)} chunks!")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# Main content based on selected tab
if selected == "💬 Chat":
    # Chat interface
    if st.session_state.conversation_chain is None:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 1rem; margin: 2rem 0;">
            <h3>📄 Upload Documents to Get Started</h3>
            <p>Upload PDF documents in the sidebar to enable advanced RAG-powered conversations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history with enhanced styling
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Fenestration Pro AI</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input with enhanced processing
    if prompt := st.chat_input("Ask sophisticated questions about your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Generate response
        start_time = datetime.now()
        
        with st.spinner("🧠 Processing with advanced AI..."):
            if api_key and st.session_state.conversation_chain:
                try:
                    # Use RAG chain for response
                    result = st.session_state.conversation_chain({
                        "question": prompt
                    })
                    
                    assistant_response = result["answer"]
                    source_docs = result.get("source_documents", [])
                    
                    # Add source information
                    if source_docs:
                        assistant_response += "\n\n📚 **Sources:**\n"
                        for i, doc in enumerate(source_docs[:3]):  # Show top 3 sources
                            source_info = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "N/A")
                            assistant_response += f"- Document: {Path(source_info).name}, Page: {page}\n"
                    
                except Exception as e:
                    assistant_response = f"Error: {str(e)}. Please check your setup."
            elif api_key:
                try:
                    # Fallback to direct OpenAI API
                    openai.api_key = api_key
                    
                    response = openai.ChatCompletion.create(
                        model=model_choice,
                        messages=[
                            {"role": "system", "content": "You are an expert in fenestration, windows, doors, glazing systems, and building envelope. Provide detailed, technical answers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    assistant_response = response.choices[0].message['content']
                    
                except Exception as e:
                    assistant_response = f"Error: {str(e)}. Please check your API key."
            else:
                assistant_response = "Please upload documents and ensure your API key is configured to enable advanced RAG responses."
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Update statistics
        st.session_state.document_stats["total_queries"] += 1
        current_avg = st.session_state.document_stats["avg_response_time"]
        total_queries = st.session_state.document_stats["total_queries"]
        st.session_state.document_stats["avg_response_time"] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": assistant_response,
            "timestamp": datetime.now(),
            "response_time": response_time
        })
        
        st.rerun()

elif selected == "📊 Analytics":
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.messages:
        # Message statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>💬 Total Messages</h3>
                <h2>{}</h2>
            </div>
            """.format(len(st.session_state.messages)), unsafe_allow_html=True)
        
        with col2:
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.markdown("""
            <div class="metric-card">
                <h3>👤 User Queries</h3>
                <h2>{}</h2>
            </div>
            """.format(user_messages), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📄 Documents</h3>
                <h2>{}</h2>
            </div>
            """.format(st.session_state.document_stats["total_documents"]), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍 Chunks</h3>
                <h2>{}</h2>
            </div>
            """.format(st.session_state.document_stats["total_chunks"]), unsafe_allow_html=True)
        
        # Response time chart
        if len(st.session_state.messages) > 2:
            response_times = []
            timestamps = []
            
            for msg in st.session_state.messages:
                if msg["role"] == "assistant" and "response_time" in msg:
                    response_times.append(msg["response_time"])
                    timestamps.append(msg["timestamp"])
            
            if response_times:
                fig = px.line(
                    x=timestamps, 
                    y=response_times,
                    title="Response Time Trends",
                    labels={"x": "Time", "y": "Response Time (seconds)"}
                )
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start chatting to see analytics!")

elif selected == "🔧 Settings":
    st.header("🔧 Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI Configuration")
        st.info(f"Current Model: {model_choice}")
        st.info(f"Temperature: {temperature}")
        st.info(f"Max Tokens: {max_tokens}")
        
        if st.button("🔄 Reset Conversation"):
            st.session_state.messages = []
            st.success("Conversation reset!")
    
    with col2:
        st.subheader("📄 Document Settings")
        st.info(f"Chunk Size: {chunk_size}")
        st.info(f"Chunk Overlap: {chunk_overlap}")
        
        if st.button("🗑️ Clear Vector Store"):
            st.session_state.vectorstore = None
            st.session_state.conversation_chain = None
            st.success("Vector store cleared!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>🚀 Fenestration Pro AI - State-of-the-Art Edition</h4>
    <p>Powered by Advanced RAG • OpenAI GPT-4 • Vector Search • LangChain</p>
    <p>Features: Multi-document RAG • Conversation Memory • Source Attribution • Real-time Analytics</p>
    <a href='https://github.com/administrator2023/fenestration-pro-ai' style='color: #667eea;'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)
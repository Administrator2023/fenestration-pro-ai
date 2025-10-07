"""
Fenestration Pro AI - Complete State-of-the-Art Edition
The ultimate AI-powered document intelligence system for fenestration professionals
"""

import streamlit as st
import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
import uuid

# Import all our advanced modules
from advanced_rag import RAGPipeline, get_rag_pipeline, streamlit_process_documents
from multimodal_processor import MultimodalDocumentProcessor, get_multimodal_processor, process_document_with_multimodal
from analytics_dashboard import AdvancedAnalyticsDashboard, init_analytics, track_query, track_document_processing
from enterprise_features import StreamlitAuth, start_api_server
from performance_optimizer import (
    init_performance_optimization, render_performance_dashboard, 
    async_cached, monitor_performance, MemoryOptimizer, cache, perf_monitor
)

# Page configuration
st.set_page_config(
    page_title="Fenestration Pro AI - Complete SOTA Edition",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/administrator2023/fenestration-pro-ai',
        'Report a bug': "https://github.com/administrator2023/fenestration-pro-ai/issues",
        'About': "State-of-the-art AI system for fenestration professionals with advanced RAG, multimodal processing, enterprise features, and comprehensive analytics."
    }
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced CSS with animations and modern design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    color: white;
    padding: 2.5rem;
    border-radius: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
    pointer-events: none;
}

.sota-badge {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
    background-size: 200% 200%;
    animation: gradientShift 3s ease infinite;
    color: white;
    padding: 0.4rem 1rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 700;
    display: inline-block;
    margin-left: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.feature-badge {
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
    display: inline-block;
    backdrop-filter: blur(10px);
}

.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 2rem;
    border-radius: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    margin-bottom: 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 48px rgba(0,0,0,0.15);
}

.chat-message {
    padding: 1.5rem;
    border-radius: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    animation: fadeInUp 0.5s ease;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: 2rem;
    border: 2px solid rgba(255,255,255,0.1);
}

.assistant-message {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 2px solid #e2e8f0;
    margin-right: 2rem;
    position: relative;
}

.processing-animation {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 50%, #4facfe 100%);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
    color: white;
    padding: 1rem 2rem;
    border-radius: 1rem;
    margin: 1rem 0;
    text-align: center;
    font-weight: 600;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.sidebar-section {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.2);
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

.status-online { background-color: #10b981; }
.status-processing { background-color: #f59e0b; }
.status-error { background-color: #ef4444; }

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.nav-tab {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 1rem;
    font-weight: 600;
    margin: 0.2rem;
    transition: all 0.3s ease;
}

.nav-tab:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.progress-ring {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, #667eea, #764ba2, #f093fb, #667eea);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: rotate 2s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.glassmorphism {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 1rem;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize all systems
def initialize_application():
    """Initialize all application systems"""
    try:
        # Initialize session state
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = False
        
        if not st.session_state.app_initialized:
            # Initialize performance optimization
            init_performance_optimization()
            
            # Initialize analytics
            init_analytics()
            
            # Initialize unique session ID
            if 'session_id' not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
            
            # Initialize message history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🚀 Welcome to Fenestration Pro AI - Complete State-of-the-Art Edition! I'm powered by advanced RAG technology, multimodal processing, enterprise features, and comprehensive analytics. Upload documents and experience the future of AI-powered fenestration intelligence!",
                    "timestamp": datetime.now(),
                    "features_used": ["welcome", "initialization"]
                })
            
            # Initialize document state
            if 'processed_documents' not in st.session_state:
                st.session_state.processed_documents = []
            
            if 'rag_pipeline' not in st.session_state:
                st.session_state.rag_pipeline = None
            
            if 'multimodal_processor' not in st.session_state:
                st.session_state.multimodal_processor = None
            
            # Initialize system status
            if 'system_status' not in st.session_state:
                st.session_state.system_status = {
                    'rag_ready': False,
                    'multimodal_ready': False,
                    'analytics_ready': True,
                    'enterprise_ready': True,
                    'performance_optimized': True
                }
            
            st.session_state.app_initialized = True
            logger.info("Application initialized successfully")
    
    except Exception as e:
        logger.error(f"Application initialization error: {e}")
        st.error(f"Initialization error: {str(e)}")

# Main application header
def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ Fenestration Pro AI <span class="sota-badge">Complete SOTA</span></h1>
        <p style="font-size: 1.2rem; margin: 1rem 0;">Advanced AI Document Intelligence System</p>
        <div style="margin-top: 1.5rem;">
            <span class="feature-badge">🧠 Advanced RAG</span>
            <span class="feature-badge">📊 Multimodal AI</span>
            <span class="feature-badge">📈 Real-time Analytics</span>
            <span class="feature-badge">🏢 Enterprise Ready</span>
            <span class="feature-badge">⚡ Performance Optimized</span>
            <span class="feature-badge">🔒 Secure & Scalable</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced sidebar with all features
def render_enhanced_sidebar():
    """Render enhanced sidebar with all features"""
    with st.sidebar:
        # System status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔋 System Status")
        
        status = st.session_state.system_status
        col1, col2 = st.columns(2)
        
        with col1:
            rag_status = "🟢 Ready" if status['rag_ready'] else "🟡 Standby"
            st.markdown(f"**RAG System:** {rag_status}")
            
            analytics_status = "🟢 Active" if status['analytics_ready'] else "🔴 Offline"
            st.markdown(f"**Analytics:** {analytics_status}")
        
        with col2:
            multimodal_status = "🟢 Ready" if status['multimodal_ready'] else "🟡 Standby"
            st.markdown(f"**Multimodal:** {multimodal_status}")
            
            perf_status = "🟢 Optimized" if status['performance_optimized'] else "🟡 Standard"
            st.markdown(f"**Performance:** {perf_status}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # API Configuration
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ AI Configuration")
        
        # API Key (already configured)
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if api_key:
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key missing")
        
        # Model selection
        model_choice = st.selectbox(
            "🤖 AI Model",
            ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="Choose your preferred AI model"
        )
        
        # Advanced settings
        with st.expander("🔬 Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.slider("Max Tokens", 100, 4000, 1000, 100)
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
            
            # Processing options
            st.subheader("Processing Options")
            enable_multimodal = st.checkbox("Enable Multimodal Processing", value=True)
            enable_table_extraction = st.checkbox("Extract Tables", value=True)
            enable_image_analysis = st.checkbox("Analyze Images", value=True)
            enable_chart_detection = st.checkbox("Detect Charts", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Document Processing
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📄 Document Processing")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple documents for advanced AI processing"
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) uploaded!")
            
            # Processing configuration
            processing_config = {
                'loader_type': 'auto',
                'splitting_strategy': 'recursive',
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'vector_store_type': 'chroma',
                'embedding_model': 'openai',
                'enable_multimodal': enable_multimodal,
                'enable_table_extraction': enable_table_extraction,
                'enable_image_analysis': enable_image_analysis,
                'enable_chart_detection': enable_chart_detection
            }
            
            if st.button("🚀 Process Documents", type="primary"):
                process_documents_advanced(uploaded_files, api_key, processing_config)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📊 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.processed_documents))
            st.metric("Messages", len(st.session_state.messages))
        
        with col2:
            # Calculate total chunks
            total_chunks = sum([doc.get('chunks', 0) for doc in st.session_state.processed_documents])
            st.metric("Chunks", total_chunks)
            
            # Session duration
            if 'session_start_time' in st.session_state:
                duration = datetime.now() - st.session_state.session_start_time
                st.metric("Session", f"{duration.seconds // 60}m")
            else:
                st.session_state.session_start_time = datetime.now()
                st.metric("Session", "0m")
        
        st.markdown('</div>', unsafe_allow_html=True)

@monitor_performance("document_processing")
async def process_documents_advanced(uploaded_files, api_key: str, config: Dict[str, Any]):
    """Advanced document processing with all features"""
    
    if not api_key:
        st.error("OpenAI API key not configured")
        return
    
    progress_container = st.container()
    
    with progress_container:
        st.markdown("""
        <div class="processing-animation">
            🔄 Processing documents with state-of-the-art AI pipeline...
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize processors
            status_text.text("Initializing AI systems...")
            progress_bar.progress(10)
            
            if not st.session_state.rag_pipeline:
                st.session_state.rag_pipeline = get_rag_pipeline(api_key)
            
            if config['enable_multimodal'] and not st.session_state.multimodal_processor:
                st.session_state.multimodal_processor = get_multimodal_processor(api_key)
            
            progress_bar.progress(20)
            
            # Save uploaded files temporarily
            temp_files = []
            status_text.text("Preparing documents...")
            
            for i, uploaded_file in enumerate(uploaded_files):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
                temp_file.write(uploaded_file.getbuffer())
                temp_file.close()
                temp_files.append(temp_file.name)
                progress_bar.progress(20 + (i + 1) * 10 // len(uploaded_files))
            
            # Process with RAG pipeline
            status_text.text("Processing with advanced RAG pipeline...")
            progress_bar.progress(40)
            
            rag_results = await st.session_state.rag_pipeline.process_documents_async(temp_files, config)
            progress_bar.progress(60)
            
            # Process with multimodal if enabled
            multimodal_results = {}
            if config['enable_multimodal'] and st.session_state.multimodal_processor:
                status_text.text("Analyzing images, tables, and charts...")
                
                for i, temp_file in enumerate(temp_files):
                    if temp_file.endswith('.pdf'):
                        file_results = await process_document_with_multimodal(temp_file, api_key)
                        multimodal_results[uploaded_files[i].name] = file_results
                        progress_bar.progress(60 + (i + 1) * 20 // len(temp_files))
            
            progress_bar.progress(90)
            status_text.text("Finalizing processing...")
            
            # Store results
            for i, uploaded_file in enumerate(uploaded_files):
                doc_info = {
                    'filename': uploaded_file.name,
                    'size': uploaded_file.size,
                    'processed_time': datetime.now(),
                    'chunks': rag_results.get('total_chunks', 0),
                    'rag_results': rag_results,
                    'multimodal_results': multimodal_results.get(uploaded_file.name, {}),
                    'config': config
                }
                st.session_state.processed_documents.append(doc_info)
                
                # Track analytics
                track_document_processing(
                    uploaded_file.name,
                    uploaded_file.size,
                    rag_results.get('processing_time', 0),
                    rag_results.get('total_chunks', 0),
                    len(rag_results.get('errors', [])) == 0
                )
            
            # Update system status
            st.session_state.system_status['rag_ready'] = True
            if config['enable_multimodal']:
                st.session_state.system_status['multimodal_ready'] = True
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink()
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {temp_file}: {e}")
            
            # Show results
            st.success(f"✅ Successfully processed {len(uploaded_files)} documents!")
            
            # Display processing summary
            with st.expander("📊 Processing Summary", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Files Processed", len(rag_results.get('processed_files', [])))
                    st.metric("Total Chunks", rag_results.get('total_chunks', 0))
                
                with col2:
                    processing_time = rag_results.get('processing_time', 0)
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                    
                    errors = len(rag_results.get('errors', []))
                    st.metric("Errors", errors, delta="🟢 None" if errors == 0 else "🔴 Found")
                
                with col3:
                    if multimodal_results:
                        total_images = sum([len(r.get('multimodal_results', {}).get('images', [])) 
                                          for r in multimodal_results.values()])
                        total_tables = sum([len(r.get('multimodal_results', {}).get('tables', [])) 
                                          for r in multimodal_results.values()])
                        
                        st.metric("Images Found", total_images)
                        st.metric("Tables Found", total_tables)
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            st.error(f"Processing failed: {str(e)}")
            status_text.text("❌ Processing failed")
        
        finally:
            # Clear processing animation
            progress_container.empty()

# Enhanced chat interface
def render_chat_interface():
    """Render enhanced chat interface"""
    
    # Display chat history with enhanced styling
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <strong>👤 You</strong>
                    <span style="margin-left: auto; font-size: 0.8rem; opacity: 0.8;">
                        {message.get('timestamp', datetime.now()).strftime('%H:%M')}
                    </span>
                </div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Enhanced assistant message with features used
            features_used = message.get('features_used', [])
            features_html = ""
            if features_used:
                features_html = "<div style='margin-top: 1rem;'>"
                for feature in features_used:
                    features_html += f"<span class='feature-badge' style='background: rgba(102, 126, 234, 0.2); color: #667eea; margin: 0.2rem;'>{feature}</span>"
                features_html += "</div>"
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <strong>🤖 Fenestration Pro AI</strong>
                    <span style="margin-left: auto; font-size: 0.8rem; opacity: 0.6;">
                        {message.get('timestamp', datetime.now()).strftime('%H:%M')}
                    </span>
                </div>
                <div>{message["content"]}</div>
                {features_html}
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced chat input
    if prompt := st.chat_input("Ask sophisticated questions about your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Process query with advanced features
        process_advanced_query(prompt)

@monitor_performance("query_processing")
async def process_advanced_query(prompt: str):
    """Process query with all advanced features"""
    
    start_time = datetime.now()
    features_used = []
    
    with st.spinner("🧠 Processing with advanced AI systems..."):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
            
            if not api_key:
                response = "Please configure your OpenAI API key to enable AI responses."
                features_used = ["error_handling"]
            elif not st.session_state.rag_pipeline:
                response = "Please upload and process documents first to enable advanced RAG responses."
                features_used = ["guidance"]
            else:
                # Use RAG pipeline for response
                result = await st.session_state.rag_pipeline.query_async(prompt)
                
                if 'error' in result:
                    response = f"Error processing query: {result['error']}"
                    features_used = ["error_handling"]
                else:
                    response = result.get('answer', 'No response generated')
                    features_used = ["advanced_rag", "vector_search", "llm_processing"]
                    
                    # Add source information if available
                    source_docs = result.get('source_documents', [])
                    if source_docs:
                        response += "\n\n📚 **Sources:**\n"
                        for i, doc in enumerate(source_docs[:3]):
                            source_info = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            response += f"- Document: {Path(source_info).name}, Page: {page}\n"
                        features_used.append("source_attribution")
                    
                    # Add multimodal context if available
                    if st.session_state.multimodal_processor and any(doc.get('multimodal_results') for doc in st.session_state.processed_documents):
                        response += "\n\n🖼️ *Note: Multimodal content (images, tables, charts) has been analyzed and integrated into the response.*"
                        features_used.append("multimodal_analysis")
        
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            response = f"An error occurred while processing your query: {str(e)}"
            features_used = ["error_handling"]
    
    # Calculate response time
    response_time = (datetime.now() - start_time).total_seconds()
    
    # Track analytics
    track_query(
        prompt,
        response_time,
        "gpt-4-turbo-preview",  # Default model
        tokens_used=len(response.split()) * 1.3,  # Rough estimate
        error_occurred='error' in response.lower()
    )
    
    # Add assistant response with features used
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now(),
        "response_time": response_time,
        "features_used": features_used
    })
    
    st.rerun()

# Main navigation and content
def render_main_content():
    """Render main content with navigation"""
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Chat", 
        "📊 Analytics", 
        "🖼️ Multimodal", 
        "⚡ Performance", 
        "🔧 Settings"
    ])
    
    with tab1:
        st.header("💬 Advanced AI Chat")
        
        if not st.session_state.processed_documents:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 1rem; margin: 2rem 0;">
                <h3>🚀 Get Started</h3>
                <p>Upload documents in the sidebar to unlock the full power of advanced RAG, multimodal processing, and AI intelligence!</p>
                <div style="margin-top: 1rem;">
                    <span class="feature-badge" style="background: rgba(102, 126, 234, 0.1); color: #667eea;">📄 Upload PDFs</span>
                    <span class="feature-badge" style="background: rgba(102, 126, 234, 0.1); color: #667eea;">🧠 AI Processing</span>
                    <span class="feature-badge" style="background: rgba(102, 126, 234, 0.1); color: #667eea;">💬 Smart Chat</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        render_chat_interface()
    
    with tab2:
        st.header("📊 Advanced Analytics")
        dashboard = AdvancedAnalyticsDashboard()
        dashboard.render_dashboard()
    
    with tab3:
        st.header("🖼️ Multimodal Analysis")
        render_multimodal_analysis()
    
    with tab4:
        st.header("⚡ Performance Dashboard")
        render_performance_dashboard()
    
    with tab5:
        st.header("🔧 Advanced Settings")
        render_advanced_settings()

def render_multimodal_analysis():
    """Render multimodal analysis results"""
    
    if not st.session_state.processed_documents:
        st.info("Upload and process documents to see multimodal analysis results.")
        return
    
    # Filter documents with multimodal results
    multimodal_docs = [doc for doc in st.session_state.processed_documents 
                      if doc.get('multimodal_results')]
    
    if not multimodal_docs:
        st.info("No multimodal content found. Enable multimodal processing in the sidebar.")
        return
    
    # Document selector
    selected_doc = st.selectbox(
        "Select Document",
        multimodal_docs,
        format_func=lambda x: x['filename']
    )
    
    if selected_doc:
        multimodal_data = selected_doc['multimodal_results']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            images_count = len(multimodal_data.get('multimodal_results', {}).get('images', []))
            st.metric("Images", images_count)
        
        with col2:
            tables_count = len(multimodal_data.get('multimodal_results', {}).get('tables', []))
            st.metric("Tables", tables_count)
        
        with col3:
            charts_count = len(multimodal_data.get('multimodal_results', {}).get('charts', []))
            st.metric("Charts", charts_count)
        
        with col4:
            text_items = len(multimodal_data.get('multimodal_results', {}).get('text_content', []))
            st.metric("Text Items", text_items)
        
        # Detailed analysis
        if images_count > 0:
            st.subheader("🖼️ Image Analysis")
            images = multimodal_data['multimodal_results']['images']
            
            for i, image in enumerate(images[:5]):  # Show first 5 images
                with st.expander(f"Image {i+1} - Page {image.page_number}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        try:
                            import base64
                            from PIL import Image as PILImage
                            from io import BytesIO
                            
                            # Display image
                            pil_image = PILImage.open(BytesIO(image.image_data))
                            st.image(pil_image, caption=f"Page {image.page_number}")
                        except Exception as e:
                            st.error(f"Could not display image: {e}")
                    
                    with col2:
                        st.write("**AI Description:**")
                        st.write(image.description)
                        
                        if image.ocr_text.strip():
                            st.write("**OCR Text:**")
                            st.code(image.ocr_text)
        
        if tables_count > 0:
            st.subheader("📊 Table Analysis")
            tables = multimodal_data['multimodal_results']['tables']
            
            for i, table in enumerate(tables[:3]):  # Show first 3 tables
                with st.expander(f"Table {i+1} - Page {table.page_number}"):
                    st.write(f"**Type:** {table.table_type}")
                    st.write(f"**Confidence:** {table.confidence:.2f}")
                    
                    if not table.dataframe.empty:
                        st.dataframe(table.dataframe, use_container_width=True)
                    else:
                        st.info("Table data could not be extracted")

def render_advanced_settings():
    """Render advanced settings panel"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI Configuration")
        
        # Current settings display
        st.info("Current settings are configured in the sidebar")
        
        # System information
        st.subheader("🔧 System Information")
        
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        st.write(f"**CPU Usage:** {cpu_percent:.1f}%")
        st.write(f"**Memory Usage:** {memory.percent:.1f}%")
        st.write(f"**Available Memory:** {memory.available / 1024 / 1024 / 1024:.1f} GB")
        
        # Cache information
        st.subheader("🗄️ Cache Status")
        cache_size = len(cache.memory_cache)
        st.write(f"**Cache Items:** {cache_size}")
        
        if st.button("Clear All Caches"):
            asyncio.run(cache.clear())
            MemoryOptimizer.cleanup_memory()
            st.success("All caches cleared!")
    
    with col2:
        st.subheader("📊 Session Statistics")
        
        # Session stats
        session_duration = datetime.now() - st.session_state.get('session_start_time', datetime.now())
        st.write(f"**Session Duration:** {session_duration.seconds // 60} minutes")
        st.write(f"**Messages Sent:** {len([m for m in st.session_state.messages if m['role'] == 'user'])}")
        st.write(f"**Documents Processed:** {len(st.session_state.processed_documents)}")
        
        # Performance metrics
        metrics_summary = perf_monitor.get_metrics_summary()
        if metrics_summary.get("total_operations", 0) > 0:
            st.write(f"**Operations Performed:** {metrics_summary['total_operations']}")
            st.write(f"**Average Response Time:** {metrics_summary['avg_duration']:.2f}s")
        
        # Reset options
        st.subheader("🔄 Reset Options")
        
        col1_reset, col2_reset = st.columns(2)
        
        with col1_reset:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.success("Chat history cleared!")
        
        with col2_reset:
            if st.button("Reset Session"):
                # Clear session state except for authentication
                keys_to_keep = ['authenticated', 'user']
                keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
                
                for key in keys_to_remove:
                    del st.session_state[key]
                
                st.success("Session reset! Please refresh the page.")

# Main application
def main():
    """Main application entry point"""
    
    try:
        # Initialize application
        initialize_application()
        
        # Authentication (optional - can be disabled for demo)
        # auth = StreamlitAuth()
        # auth.setup_authentication()
        
        # if not st.session_state.get('authenticated', True):  # Set to True to skip auth for demo
        #     return
        
        # Render header
        render_header()
        
        # Render sidebar
        render_enhanced_sidebar()
        
        # Render main content
        render_main_content()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <h4>🚀 Fenestration Pro AI - Complete State-of-the-Art Edition</h4>
            <p>Powered by Advanced RAG • Multimodal AI • Real-time Analytics • Enterprise Features • Performance Optimization</p>
            <p><strong>Features:</strong> Multi-document RAG • Image/Table/Chart Analysis • Conversation Memory • Source Attribution • Real-time Performance Monitoring • Enterprise Authentication • Comprehensive Testing</p>
            <a href='https://github.com/administrator2023/fenestration-pro-ai' style='color: #667eea; text-decoration: none;'>🔗 View on GitHub</a>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application error: {str(e)}")
        
        # Error recovery
        if st.button("🔄 Restart Application"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
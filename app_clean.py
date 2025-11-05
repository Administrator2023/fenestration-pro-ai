"""
Fenestration Pro AI - Clean, Intelligent Interface
Single-purpose GUI: Upload drawings → Ask questions → Get expert answers
"""

import streamlit as st
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import json

# Intelligent document processing
try:
    from domain_qa_engine import create_qa_engine, PMResponse
    from continuous_learning_engine import LearningStats
    INTELLIGENT_ENGINE_AVAILABLE = True
except ImportError:
    INTELLIGENT_ENGINE_AVAILABLE = False
    st.error("⚠️ Intelligent engine not available. Install dependencies: pip install -r requirements.txt")

from openai import OpenAI

# ====================
# PAGE CONFIGURATION
# ====================
st.set_page_config(
    page_title="Fenestration Pro AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================
# CUSTOM CSS - CLEAN & MODERN
# ====================
st.markdown("""
<style>
    /* Clean header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Clean chat messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }

    .user-message {
        background: #f8f9fa;
        border-left-color: #667eea;
    }

    .assistant-message {
        background: #ffffff;
        border-left-color: #10b981;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .citation {
        background: #f0f9ff;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.9rem;
        border-left: 3px solid #3b82f6;
    }

    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .confidence-high { background: #d1fae5; color: #065f46; }
    .confidence-medium { background: #fef3c7; color: #92400e; }
    .confidence-low { background: #fee2e2; color: #991b1b; }

    /* Learning indicator */
    .learning-pulse {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
        margin-bottom: 1rem;
    }

    /* Clean metrics */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================
# SESSION STATE INITIALIZATION
# ====================
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'qa_engine' not in st.session_state:
    st.session_state.qa_engine = None

if 'tenant_id' not in st.session_state:
    st.session_state.tenant_id = "default_tenant"

if 'project_id' not in st.session_state:
    st.session_state.project_id = "default_project"

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

if 'learning_enabled' not in st.session_state:
    st.session_state.learning_enabled = True

# ====================
# HEADER
# ====================
st.markdown("""
<div class="main-header">
    <h1>🏗️ Fenestration Pro AI</h1>
    <p>Upload shop drawings, specs, and calculations → Ask questions → Get expert answers with citations</p>
</div>
""", unsafe_allow_html=True)

# ====================
# SIDEBAR - MINIMAL & SMART
# ====================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # API Key
    api_key = st.secrets.get("OPENAI_API_KEY", "") or st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key"
    )

    if api_key:
        st.success("✅ Connected")

    st.markdown("---")

    # Tenant/Project (minimal)
    with st.expander("📁 Project Settings", expanded=False):
        tenant_id = st.text_input("Company ID", value=st.session_state.tenant_id, help="Your company identifier")
        project_id = st.text_input("Project ID", value=st.session_state.project_id, help="Current project")

        if st.button("Update Project"):
            st.session_state.tenant_id = tenant_id
            st.session_state.project_id = project_id
            st.session_state.qa_engine = None  # Reset engine
            st.success("Project updated!")

    st.markdown("---")

    # Advanced (collapsed by default)
    with st.expander("🚀 Advanced Features", expanded=False):
        st.markdown("**Document AI (Google Cloud)**")
        use_docai = st.checkbox("Enable Document AI", value=False, help="Advanced OCR for scanned drawings")

        if use_docai:
            docai_project = st.text_input("GCP Project ID", value=st.secrets.get("DOCAI_PROJECT_ID", ""))
            docai_location = st.text_input("Location", value=st.secrets.get("DOCAI_LOCATION", "us"))
            docai_processor = st.text_input("Processor ID", value=st.secrets.get("DOCAI_PROCESSOR_ID", ""))
            st.session_state.docai_credentials = {
                'project_id': docai_project,
                'location': docai_location,
                'processor_id': docai_processor
            }

        st.markdown("**Pinecone (Vector Database)**")
        use_pinecone = st.checkbox("Enable Pinecone", value=False, help="Production-scale search")

        if use_pinecone:
            pinecone_key = st.text_input("Pinecone API Key", type="password", value=st.secrets.get("PINECONE_API_KEY", ""))
            pinecone_env = st.text_input("Environment", value=st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1-aws"))
            st.session_state.pinecone_credentials = {
                'api_key': pinecone_key,
                'environment': pinecone_env
            }

    st.markdown("---")

    # Learning stats (subtle)
    if st.session_state.qa_engine:
        try:
            stats = st.session_state.qa_engine.get_learning_stats()
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: #f0fdf4; border-radius: 8px;">
                <div class="learning-pulse"></div>
                <strong>AI Learning Active</strong><br>
                <small>{stats.total_documents} docs • {stats.patterns_discovered} patterns</small>
            </div>
            """, unsafe_allow_html=True)
        except:
            pass

# ====================
# MAIN INTERFACE
# ====================

# Initialize QA engine if API key provided
if api_key and not st.session_state.qa_engine:
    try:
        with st.spinner("🧠 Initializing intelligent engine..."):
            docai_creds = st.session_state.get('docai_credentials', {}) if st.session_state.get('use_docai') else None
            pinecone_creds = st.session_state.get('pinecone_credentials', {}) if st.session_state.get('use_pinecone') else None

            st.session_state.qa_engine = create_qa_engine(
                openai_api_key=api_key,
                tenant_id=st.session_state.tenant_id,
                docai_credentials=docai_creds,
                pinecone_credentials=pinecone_creds
            )
            st.success("✅ Engine ready!")
    except Exception as e:
        st.error(f"Failed to initialize engine: {e}")

# ====================
# FILE UPLOAD - CLEAN & SMART
# ====================
st.markdown("### 📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Drop shop drawings, specs, calculations, or images here",
    type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    accept_multiple_files=True,
    help="Supports: PDFs, images (scanned drawings, photos)",
    label_visibility="collapsed"
)

if uploaded_files and st.button("🚀 Process Documents", type="primary", use_container_width=True):
    if not api_key:
        st.error("⚠️ Please add your OpenAI API key in the sidebar")
    elif not st.session_state.qa_engine:
        st.error("⚠️ Engine not initialized")
    else:
        with st.spinner("🧠 Processing with intelligent document understanding..."):
            try:
                # Save files temporarily
                temp_dir = tempfile.mkdtemp()
                file_paths = []

                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(temp_path)

                # Process with QA engine (Admin mode)
                report = st.session_state.qa_engine.admin_ingest(
                    file_paths=file_paths,
                    project_id=st.session_state.project_id,
                    discipline="fenestration"
                )

                # Track processed files
                st.session_state.processed_files.extend([f.name for f in uploaded_files])

                # Show success with metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{report['files_processed']}</div>
                        <div class="metric-label">Files Processed</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{report['total_pages']}</div>
                        <div class="metric-label">Pages Analyzed</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{report['total_chunks']}</div>
                        <div class="metric-label">Chunks Indexed</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    entities = len(report['extracted_entities']['dimensions']) + len(report['extracted_entities']['materials'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{entities}</div>
                        <div class="metric-label">Entities Found</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Show extracted intelligence (collapsible)
                with st.expander("📊 View Extracted Data", expanded=False):
                    if report['extracted_entities']['dimensions']:
                        st.markdown("**📐 Dimensions:**")
                        st.write(", ".join(report['extracted_entities']['dimensions'][:20]))

                    if report['extracted_entities']['materials']:
                        st.markdown("**🔧 Materials:**")
                        st.write(", ".join(report['extracted_entities']['materials']))

                    if report['extracted_entities']['specs']:
                        st.markdown("**📋 Specifications:**")
                        st.json(report['extracted_entities']['specs'])

                st.success("✅ Documents processed! AI is learning from your uploads...")

                # Cleanup
                shutil.rmtree(temp_dir)

            except Exception as e:
                st.error(f"Error processing documents: {e}")
                import traceback
                st.error(traceback.format_exc())

# Show processed files (subtle)
if st.session_state.processed_files:
    with st.expander(f"📚 Knowledge Base ({len(st.session_state.processed_files)} documents)", expanded=False):
        for filename in st.session_state.processed_files:
            st.markdown(f"✅ {filename}")

# ====================
# CHAT INTERFACE - CLEAN & FOCUSED
# ====================
st.markdown("---")
st.markdown("### 💬 Ask Questions")

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Parse structured response if available
        response_data = message.get("data", {})

        answer = response_data.get("answer", content)
        citations = response_data.get("citations", [])
        confidence = response_data.get("confidence", 0.0)
        assumptions = response_data.get("assumptions", [])
        followups = response_data.get("followups", [])

        # Confidence badge
        if confidence >= 0.8:
            conf_class = "confidence-high"
            conf_text = f"High Confidence ({confidence:.0%})"
        elif confidence >= 0.6:
            conf_class = "confidence-medium"
            conf_text = f"Medium Confidence ({confidence:.0%})"
        else:
            conf_class = "confidence-low"
            conf_text = f"Low Confidence ({confidence:.0%})"

        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>AI Assistant:</strong><br>
            {answer}

            <div class="confidence-badge {conf_class}">{conf_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Citations
        if citations:
            citation_text = "<br>".join([
                f"• {c.get('doc', 'Unknown')} (Page {c.get('page', '?')}, Section: {c.get('section', 'N/A')})"
                for c in citations[:3]
            ])
            st.markdown(f"""
            <div class="citation">
                <strong>📎 Sources:</strong><br>
                {citation_text}
            </div>
            """, unsafe_allow_html=True)

        # Follow-ups (subtle)
        if followups:
            with st.expander("💡 Related Questions", expanded=False):
                for followup in followups:
                    st.markdown(f"• {followup}")

# Chat input
if prompt := st.chat_input("Ask about dimensions, materials, specs, calculations..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    if not st.session_state.qa_engine:
        response_text = "⚠️ Please configure API key and upload documents first."
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        with st.spinner("🤔 Analyzing documents..."):
            try:
                # Query with PM mode
                response = st.session_state.qa_engine.pm_query(
                    question=prompt,
                    project_id=st.session_state.project_id,
                    discipline="fenestration"
                )

                # Store structured response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "data": response.to_dict()
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                })

    st.rerun()

# ====================
# FOOTER - MINIMAL
# ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem;">
    Fenestration Pro AI • Intelligent shop drawing analysis with citations •
    <a href="https://github.com/Administrator2023/fenestration-pro-ai" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)

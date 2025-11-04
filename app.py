"""
Fenestration Pro AI - Minimal Clean Interface
Two modes: Admin (upload) | Client (ask)
"""

import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path

# Intelligent engines
try:
    from domain_qa_engine import create_qa_engine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

st.set_page_config(
    page_title="Fenestration Pro AI",
    page_icon="🏗️",
    layout="centered"
)

# Minimal CSS
st.markdown("""
<style>
    .main { max-width: 900px; }

    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }

    .header h1 { margin: 0; font-size: 2.5rem; }
    .header p { margin: 0.5rem 0 0 0; opacity: 0.95; }

    .mode-toggle {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
    }

    .mode-btn {
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        background: white;
        cursor: pointer;
        font-size: 1.1rem;
        transition: all 0.3s;
    }

    .mode-btn.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
    }

    .chat-msg {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
    }

    .user-msg {
        background: #f1f5f9;
        border-left: 4px solid #667eea;
    }

    .ai-msg {
        background: white;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .citation {
        background: #f0f9ff;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.9rem;
        border-left: 3px solid #3b82f6;
    }

    .conf-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .conf-high { background: #d1fae5; color: #065f46; }
    .conf-med { background: #fef3c7; color: #92400e; }
    .conf-low { background: #fee2e2; color: #991b1b; }

    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session state
if 'mode' not in st.session_state:
    st.session_state.mode = "client"
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = 0

# Header
st.markdown("""
<div class="header">
    <h1>🏗️ Fenestration Pro AI</h1>
    <p>Intelligent shop drawing assistant</p>
</div>
""", unsafe_allow_html=True)

# Mode toggle
col1, col2 = st.columns(2)
with col1:
    if st.button("👤 Client Mode", use_container_width=True, type="primary" if st.session_state.mode == "client" else "secondary"):
        st.session_state.mode = "client"
        st.rerun()
with col2:
    if st.button("👨‍💼 Admin Mode", use_container_width=True, type="primary" if st.session_state.mode == "admin" else "secondary"):
        st.session_state.mode = "admin"
        st.rerun()

st.markdown("---")

# Get credentials from secrets
openai_key = st.secrets.get("OPENAI_API_KEY", "")
docai_project = st.secrets.get("DOCAI_PROJECT_ID", "")
docai_location = st.secrets.get("DOCAI_LOCATION", "us")
docai_processor = st.secrets.get("DOCAI_PROCESSOR_ID", "")
pinecone_key = st.secrets.get("PINECONE_API_KEY", "")
pinecone_env = st.secrets.get("PINECONE_ENVIRONMENT", "")

# Initialize engine if not already
if not st.session_state.engine and openai_key and ENGINE_AVAILABLE:
    try:
        with st.spinner("Initializing AI..."):
            # Build credentials
            docai_creds = None
            if docai_project and docai_processor:
                docai_creds = {
                    'project_id': docai_project,
                    'location': docai_location,
                    'processor_id': docai_processor
                }

            pinecone_creds = None
            if pinecone_key and pinecone_env:
                pinecone_creds = {
                    'api_key': pinecone_key,
                    'environment': pinecone_env
                }

            st.session_state.engine = create_qa_engine(
                openai_api_key=openai_key,
                tenant_id="default",
                docai_credentials=docai_creds,
                pinecone_credentials=pinecone_creds
            )
    except Exception as e:
        st.error(f"Engine init failed: {e}")

# ===================
# ADMIN MODE
# ===================
if st.session_state.mode == "admin":
    st.markdown("### 📤 Upload Documents")
    st.caption("Upload shop drawings, specs, calculations (PDF or images)")

    files = st.file_uploader(
        "Drop files here",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if files and st.button("🚀 Process & Learn", type="primary", use_container_width=True):
        if not st.session_state.engine:
            st.error("Add OpenAI API key to .streamlit/secrets.toml")
        else:
            with st.spinner("Processing with Document AI + Pinecone..."):
                try:
                    # Save files
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []
                    for f in files:
                        path = os.path.join(temp_dir, f.name)
                        with open(path, "wb") as fp:
                            fp.write(f.getbuffer())
                        file_paths.append(path)

                    # Process with admin_ingest
                    report = st.session_state.engine.admin_ingest(
                        file_paths=file_paths,
                        project_id="default",
                        discipline="fenestration"
                    )

                    st.session_state.files_uploaded += len(files)

                    # Show clean results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Files", report['files_processed'])
                    col2.metric("Pages", report['total_pages'])
                    col3.metric("Entities", len(report['extracted_entities'].get('dimensions', [])) + len(report['extracted_entities'].get('materials', [])))

                    st.success("✅ AI learned from documents!")

                    shutil.rmtree(temp_dir)

                except Exception as e:
                    st.error(f"Error: {e}")

    # Show knowledge base count
    if st.session_state.files_uploaded > 0:
        st.info(f"📚 Knowledge base: {st.session_state.files_uploaded} documents")

# ===================
# CLIENT MODE
# ===================
else:
    st.markdown("### 💬 Ask Questions")
    st.caption("Ask about dimensions, materials, specifications, calculations")

    # Show chat history
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="chat-msg user-msg">
                <strong>You:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            data = msg.get('data', {})
            answer = data.get('answer', msg['content'])
            citations = data.get('citations', [])
            confidence = data.get('confidence', 0.0)

            # Confidence badge
            if confidence >= 0.8:
                badge = '<span class="conf-badge conf-high">High Confidence</span>'
            elif confidence >= 0.6:
                badge = '<span class="conf-badge conf-med">Medium Confidence</span>'
            else:
                badge = '<span class="conf-badge conf-low">Low Confidence</span>'

            st.markdown(f"""
            <div class="chat-msg ai-msg">
                <strong>AI:</strong> {answer}
                <br>{badge}
            </div>
            """, unsafe_allow_html=True)

            # Citations
            if citations:
                cites = "<br>".join([f"• {c.get('doc', '?')} (p{c.get('page', '?')})" for c in citations[:3]])
                st.markdown(f'<div class="citation"><strong>Sources:</strong><br>{cites}</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not st.session_state.engine:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Please add API key to .streamlit/secrets.toml"
            })
        else:
            try:
                response = st.session_state.engine.pm_query(
                    question=prompt,
                    project_id="default",
                    discipline="fenestration"
                )

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

# Get the first part of the file (before BQE tab)
import streamlit as st
import os
import tempfile
import shutil
import json
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import logging
import requests

# Import intelligent document parser
try:
    from intelligent_document_parser import (
        IntelligentDocumentParser,
        create_parser,
        TechnicalEntityExtractor,
        DOCUMENTAI_AVAILABLE,
        PINECONE_AVAILABLE
    )
    INTELLIGENT_PARSER_AVAILABLE = True
except ImportError as e:
    INTELLIGENT_PARSER_AVAILABLE = False
    logging.warning(f"Intelligent document parser not available: {e}")

# Reduce noisy Chroma PostHog errors and disable telemetry via env
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
os.environ.setdefault("CHROMADB_TELEMETRY_IMPL", "none")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# RAG imports - optional and version-compatible
RAG_AVAILABLE = False
VECTOR_STORE_AVAILABLE = False

# Resolve LangChain imports across versions
PyPDFLoader = None
RecursiveCharacterTextSplitter = None
OpenAIEmbeddings = None
ChatOpenAI = None
Chroma = None
FAISS = None
ConversationalRetrievalChain = None
ConversationBufferMemory = None
Document = None
CHROMA_CLIENT_SETTINGS = None

try:
    from langchain_community.document_loaders import PyPDFLoader as _PyPDFLoader
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter as _RecursiveCharacterTextSplitter,
    )
    from langchain_openai import (
        OpenAIEmbeddings as _OpenAIEmbeddings,
        ChatOpenAI as _ChatOpenAI,
    )
    from langchain_community.vectorstores import Chroma as _Chroma, FAISS as _FAISS
    from langchain.chains import (
        ConversationalRetrievalChain as _ConversationalRetrievalChain,
    )
    from langchain.memory import ConversationBufferMemory as _ConversationBufferMemory
    try:
        from langchain_core.documents import Document as _Document
    except ImportError:
        from langchain.docstore.document import Document as _Document

    PyPDFLoader = _PyPDFLoader
    RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter
    OpenAIEmbeddings = _OpenAIEmbeddings
    ChatOpenAI = _ChatOpenAI
    Chroma = _Chroma
    FAISS = _FAISS
    ConversationalRetrievalChain = _ConversationalRetrievalChain
    ConversationBufferMemory = _ConversationBufferMemory
    Document = _Document
    RAG_AVAILABLE = True
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    try:
        from langchain.document_loaders import PyPDFLoader as _PyPDFLoader
        from langchain.text_splitter import (
            RecursiveCharacterTextSplitter as _RecursiveCharacterTextSplitter,
        )
        from langchain.embeddings import OpenAIEmbeddings as _OpenAIEmbeddings
        from langchain.chat_models import ChatOpenAI as _ChatOpenAI
        from langchain.vectorstores import Chroma as _Chroma, FAISS as _FAISS
        from langchain.chains import (
            ConversationalRetrievalChain as _ConversationalRetrievalChain,
        )
        from langchain.memory import (
            ConversationBufferMemory as _ConversationBufferMemory,
        )
        try:
            from langchain.docstore.document import Document as _Document
        except ImportError:
            _Document = None

        PyPDFLoader = _PyPDFLoader
        RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter
        OpenAIEmbeddings = _OpenAIEmbeddings
        ChatOpenAI = _ChatOpenAI
        Chroma = _Chroma
        FAISS = _FAISS
        ConversationalRetrievalChain = _ConversationalRetrievalChain
        ConversationBufferMemory = _ConversationBufferMemory
        Document = _Document
        RAG_AVAILABLE = True
        VECTOR_STORE_AVAILABLE = True
    except ImportError:
        # RAG not available - we'll use basic mode
        pass

# Configure Chroma telemetry off if available
try:
    from chromadb.config import Settings as _ChromaSettings
    CHROMA_CLIENT_SETTINGS = _ChromaSettings(anonymized_telemetry=False)
except Exception:
    CHROMA_CLIENT_SETTINGS = None

# Project-scoped storage helpers
DATA_ROOT = "./data/projects"

def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def get_selected_project() -> str:
    return st.session_state.get("selected_project", "Default")

def get_project_dir(project: str) -> str:
    return os.path.join(DATA_ROOT, project)

def get_kb_dirs_for_current_project() -> tuple[str, str]:
    project = get_selected_project()
    base = get_project_dir(project)
    persist_dir = os.path.join(base, "chroma_db")
    faiss_dir = os.path.join(base, "faiss_index")
    ensure_directory(base)
    ensure_directory(persist_dir)
    ensure_directory(faiss_dir)
    return persist_dir, faiss_dir

def load_json_collection(project: str, name: str) -> list:
    project_dir = get_project_dir(project)
    ensure_directory(project_dir)
    fp = os.path.join(project_dir, f"{name}.json")
    if not os.path.exists(fp):
        return []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_json_collection(project: str, name: str, items: list) -> None:
    project_dir = get_project_dir(project)
    ensure_directory(project_dir)
    fp = os.path.join(project_dir, f"{name}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, default=str)

def save_attachments(files, category: str) -> list[dict]:
    """Save uploaded files under project folder and return metadata list."""
    project = get_selected_project()
    saved = []
    base = os.path.join(get_project_dir(project), "uploads", category)
    ensure_directory(base)
    for file in files or []:
        safe_name = Path(file.name).name
        dest = os.path.join(base, f"{uuid.uuid4().hex}_{safe_name}")
        with open(dest, "wb") as out:
            out.write(file.getbuffer())
        saved.append({"name": safe_name, "path": dest})
    return saved

# Try to import PyPDF2 as fallback
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Optional utilities for analytics/export
try:
    from utils import AnalyticsManager, create_download_link
except Exception:
    AnalyticsManager, create_download_link = None, None

st.set_page_config(
    page_title="Fenestration Pro AI",
    page_icon="🏗️",
    layout="wide"
)

# Initialize session state variables early to prevent AttributeError
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🚀 Welcome to Fenestration Pro AI - State-of-the-Art Edition! I'm here to help answer your questions about windows, doors, and building envelope systems. Upload a PDF, click 'Process & Learn from PDFs', and I'll give you specific answers from YOUR documents!"
    })

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

# Default parameters for an engineering office knowledge base
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 8
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "use_chroma" not in st.session_state:
    # Default to FAISS to avoid telemetry issues in some hosts
    st.session_state.use_chroma = False

# Custom CSS
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏗️ Fenestration Pro AI <span class="sota-badge">SOTA</span></h1>
    <p>Advanced AI Document Intelligence System</p>
</div>
""", unsafe_allow_html=True)

# Define functions before UI so they're available when buttons trigger
def get_embeddings(api_key: str):
    """Return OpenAIEmbeddings with compatibility across versions."""
    try:
        return OpenAIEmbeddings(api_key=api_key)
    except TypeError:
        return OpenAIEmbeddings(openai_api_key=api_key)

def create_llm(api_key: str, model_name: str):
    """Return ChatOpenAI with compatibility across versions."""
    try:
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=st.session_state.get("temperature", 0.3))
    except TypeError:
        return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=st.session_state.get("temperature", 0.3))

def create_conversation_chain(vectorstore, api_key: str, model_name: str):
    llm = create_llm(api_key, model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Prefer explicit output_key for memory-compatible chaining; fallback for older versions
    try:
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": st.session_state.get("retrieval_k", 8)}),
            memory=memory,
            return_source_documents=True,
            output_key="answer",
        )
    except TypeError:
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": st.session_state.get("retrieval_k", 8)}),
            memory=memory,
            return_source_documents=True,
        )

def load_persistent_vectorstore(api_key: str):
    """Try to load a persisted vector store from disk."""
    embeddings = get_embeddings(api_key)
    # Prefer Chroma if present
    persist_dir, faiss_dir = get_kb_dirs_for_current_project()
    try:
        if os.path.isdir(persist_dir) and os.listdir(persist_dir):
            # Try with embedding_function + client_settings
            try:
                return Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings,
                    client_settings=CHROMA_CLIENT_SETTINGS,
                )
            except TypeError:
                # Try with embedding + client_settings
                try:
                    return Chroma(
                        persist_directory=persist_dir,
                        embedding=embeddings,
                        client_settings=CHROMA_CLIENT_SETTINGS,
                    )
                except TypeError:
                    # Fallback without client_settings
                    try:
                        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
                    except TypeError:
                        return Chroma(persist_directory=persist_dir, embedding=embeddings)
    except Exception:
        pass
    # Fallback to FAISS
    try:
        if os.path.isdir(faiss_dir):
            return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        pass
    return None


def process_pdfs_intelligent(uploaded_files, api_key):
    """Process PDF files with intelligent document parser (Document AI + Pinecone)"""
    with st.spinner("🧠 Processing PDFs with advanced intelligence..."):
        try:
            # Get credentials
            docai_creds = st.session_state.get("docai_credentials", {})
            pinecone_creds = st.session_state.get("pinecone_credentials", {})
            use_docai = st.session_state.get("use_document_ai", False) and DOCUMENTAI_AVAILABLE
            use_pinecone = st.session_state.get("use_pinecone", False) and PINECONE_AVAILABLE

            # Create intelligent parser
            parser = create_parser(
                openai_api_key=api_key,
                docai_credentials=docai_creds if use_docai else None,
                pinecone_credentials=pinecone_creds if use_pinecone else None
            )

            # Process each file
            temp_dir = tempfile.mkdtemp()
            project = get_selected_project()
            processed_docs = []
            all_metadata = []

            for uploaded_file in uploaded_files:
                # Save temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Parse with intelligence
                with st.spinner(f"📄 Analyzing {uploaded_file.name}..."):
                    result = parser.index_document(temp_path, project)
                    metadata = result['metadata']
                    tables = result['tables']
                    chunks_indexed = result['chunks_indexed']

                    processed_docs.append({
                        'name': uploaded_file.name,
                        'chunks': chunks_indexed,
                        'metadata': metadata,
                        'tables': tables
                    })
                    all_metadata.append(metadata)

                    # Display extracted intelligence
                    st.success(f"✅ Processed {uploaded_file.name}")
                    with st.expander(f"📊 Extracted Data from {uploaded_file.name}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Type", metadata.get('document_type', 'Unknown'))
                            st.metric("Pages", metadata.get('page_count', 0))
                            st.metric("Confidence", f"{metadata.get('confidence_score', 0):.1%}")
                        with col2:
                            st.metric("Tables", len(tables))
                            st.metric("Dimensions", len(metadata.get('extracted_dimensions', [])))
                            st.metric("Materials", len(metadata.get('extracted_materials', [])))
                        with col3:
                            if metadata.get('drawing_number'):
                                st.metric("Drawing #", metadata['drawing_number'])
                            if metadata.get('revision'):
                                st.metric("Revision", metadata['revision'])

                        # Show dimensions
                        if metadata.get('extracted_dimensions'):
                            st.markdown("**📐 Dimensions Found:**")
                            st.write(", ".join(metadata['extracted_dimensions'][:10]))

                        # Show materials
                        if metadata.get('extracted_materials'):
                            st.markdown("**🔧 Materials Found:**")
                            st.write(", ".join(metadata['extracted_materials']))

                        # Show specs
                        if metadata.get('extracted_specs'):
                            st.markdown("**📋 Specifications:**")
                            st.json(metadata['extracted_specs'])

                        # Show tables
                        if tables:
                            st.markdown(f"**📊 Tables ({len(tables)}):**")
                            for table in tables[:3]:  # Show first 3 tables
                                st.markdown(f"*Table {table['table_index']} - {table['table_type']}*")
                                try:
                                    df = pd.DataFrame(table['rows'], columns=table['headers'])
                                    st.dataframe(df, use_container_width=True)
                                except:
                                    st.write(f"Headers: {table['headers']}")

            # Store in session
            st.session_state.processed_docs = processed_docs
            st.session_state.intelligent_metadata = all_metadata

            # Log ingested documents
            docs_log = load_json_collection(project, "ingested_docs")
            for doc in processed_docs:
                docs_log.append({
                    "name": doc['name'],
                    "chunks": doc['chunks'],
                    "metadata": doc['metadata'],
                    "ingested_at": datetime.now().isoformat(),
                })
            save_json_collection(project, "ingested_docs", docs_log)

            # Cleanup
            shutil.rmtree(temp_dir)

            st.success(f"""
            ✅ **Successfully processed {len(uploaded_files)} document(s) with advanced intelligence!**

            The system has extracted dimensions, materials, calculations, and tables.
            {'Documents are indexed in Pinecone for semantic search.' if use_pinecone else ''}
            """)

            st.info("🎯 Now ask questions about your shop drawings, specs, and calculations!")

        except Exception as e:
            st.error(f"Error in intelligent processing: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.warning("Falling back to standard PDF processing...")
            return process_pdfs(uploaded_files, api_key)


def process_pdfs(uploaded_files, api_key):
    """Process PDF files and extract content (Standard fallback method)"""
    with st.spinner("🧠 Processing PDFs and extracting content..."):
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            all_text = ""
            processed_docs = []
            all_documents = []
            
            # Process each PDF
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                doc_text = ""
                chunks_count = 0
                
                # Try different PDF processing methods
                if RAG_AVAILABLE and PyPDFLoader is not None:
                    try:
                        # Use PyPDFLoader if available
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()
                        doc_text = "\n".join([doc.page_content for doc in docs])
                        chunks_count = len(docs)
                        all_documents.extend(docs)
                    except Exception as e:
                        st.warning(f"PyPDFLoader failed, trying fallback: {str(e)}")
                
                # Fallback to PyPDF2 if needed
                if not doc_text and PYPDF2_AVAILABLE:
                    try:
                        with open(temp_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            page_count = len(pdf_reader.pages)
                            for page_num in range(page_count):
                                page = pdf_reader.pages[page_num]
                                page_text = page.extract_text() or ""
                                doc_text += page_text + "\n"
                                if Document is not None:
                                    all_documents.append(
                                        Document(
                                            page_content=page_text,
                                            metadata={"source": uploaded_file.name, "page": page_num + 1},
                                        )
                                    )
                            chunks_count = page_count
                    except Exception as e:
                        st.warning(f"PyPDF2 failed: {str(e)}")
                
                # If still no text, show error
                if not doc_text:
                    st.error(f"Could not extract text from {uploaded_file.name}")
                    continue
                
                all_text += f"\n\n--- FROM {uploaded_file.name} ---\n{doc_text}"
                
                # Track processed document
                processed_docs.append({
                    'name': uploaded_file.name,
                    'chunks': chunks_count,
                    'content': doc_text
                })
            
            # Store processed content in session state
            st.session_state.processed_docs = processed_docs
            st.session_state.document_content = all_text
            
            # If vector stores are available, try to create embeddings
            if VECTOR_STORE_AVAILABLE:
                try:
                    # Split accumulated documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=st.session_state.get("chunk_size", 1000),
                        chunk_overlap=st.session_state.get("chunk_overlap", 200),
                        length_function=len,
                    )
                    if not all_documents:
                        st.error("No document content extracted. Nothing to index.")
                        shutil.rmtree(temp_dir)
                        return
                    chunks = text_splitter.split_documents(all_documents)

                    # Create embeddings and vector store
                    embeddings = get_embeddings(api_key)

                    # Prefer FAISS by default to avoid Chroma telemetry issues
                    if not st.session_state.get("use_chroma", False):
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                        try:
                            vectorstore.save_local(get_kb_dirs_for_current_project()[1])
                        except Exception:
                            pass
                    else:
                        # Try Chroma with persistence
                        try:
                            vectorstore = None
                            # 1) embedding + client_settings
                            try:
                                vectorstore = Chroma.from_documents(
                                    documents=chunks,
                                    embedding=embeddings,
                                    persist_directory=get_kb_dirs_for_current_project()[0],
                                    client_settings=CHROMA_CLIENT_SETTINGS,
                                )
                            except TypeError:
                                # 2) embedding_function + client_settings
                                try:
                                    vectorstore = Chroma.from_documents(
                                        documents=chunks,
                                        embedding_function=embeddings,
                                        persist_directory=get_kb_dirs_for_current_project()[0],
                                        client_settings=CHROMA_CLIENT_SETTINGS,
                                    )
                                except TypeError:
                                    # 3) embedding without client_settings
                                    try:
                                        vectorstore = Chroma.from_documents(
                                            documents=chunks,
                                            embedding=embeddings,
                                            persist_directory=get_kb_dirs_for_current_project()[0],
                                        )
                                    except TypeError:
                                        # 4) embedding_function without client_settings
                                        vectorstore = Chroma.from_documents(
                                            documents=chunks,
                                            embedding_function=embeddings,
                                            persist_directory=get_kb_dirs_for_current_project()[0],
                                        )
                            # Ensure persisted
                            try:
                                vectorstore.persist()
                            except Exception:
                                pass
                        except Exception:
                            # Fallback to FAISS saved locally
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                            try:
                                vectorstore.save_local(get_kb_dirs_for_current_project()[1])
                            except Exception:
                                pass

                    # Create conversation chain with selected model
                    conversation_chain = create_conversation_chain(
                        vectorstore, api_key, st.session_state.get("selected_model", "gpt-4o-mini")
                    )

                    # Store in session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation_chain = conversation_chain

                    st.success(
                        f"✅ Successfully processed {len(uploaded_files)} PDFs and updated the persistent knowledge base!"
                    )

                    # Log ingested documents in project records
                    project = get_selected_project()
                    docs_log = load_json_collection(project, "ingested_docs")
                    for doc in processed_docs:
                        docs_log.append({
                            "name": doc['name'],
                            "chunks": doc['chunks'],
                            "ingested_at": datetime.now().isoformat(),
                        })
                    save_json_collection(project, "ingested_docs", docs_log)

                except Exception as e:
                    st.warning(f"Advanced RAG failed ({str(e)}), using basic text processing")
                    st.session_state.conversation_chain = "basic"
                    st.success(
                        f"✅ Successfully processed {len(uploaded_files)} PDFs with basic text extraction!"
                    )
            else:
                # Basic mode without vector stores
                st.session_state.conversation_chain = "basic"
                st.success(f"✅ Successfully processed {len(uploaded_files)} PDFs with basic text extraction!")
                st.info("📚 I can now answer questions based on your uploaded documents!")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            st.info("🎯 Now I can answer questions based on your specific documents!")
            
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            st.info("Make sure you have a valid OpenAI API key and try again.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API key from secrets or user input
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    
    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your key from platform.openai.com"
        )
    else:
        st.success("✅ API Key loaded from secrets")

    st.markdown("---")

    # Advanced Document Intelligence Configuration
    with st.expander("🚀 Advanced Document Intelligence", expanded=False):
        st.markdown("**Enterprise-grade parsing for shop drawings**")

        # Document AI Configuration
        st.markdown("##### Google Document AI")
        use_document_ai = st.checkbox(
            "Enable Document AI",
            value=st.session_state.get("use_document_ai", False),
            help="Advanced OCR and layout understanding for scanned drawings"
        )
        st.session_state.use_document_ai = use_document_ai

        if use_document_ai:
            docai_project_id = st.text_input(
                "GCP Project ID",
                value=st.secrets.get("DOCAI_PROJECT_ID", ""),
                help="Your Google Cloud Project ID"
            )
            docai_location = st.text_input(
                "Processor Location",
                value=st.secrets.get("DOCAI_LOCATION", "us"),
                help="e.g., 'us' or 'eu'"
            )
            docai_processor_id = st.text_input(
                "Processor ID",
                value=st.secrets.get("DOCAI_PROCESSOR_ID", ""),
                help="Document AI Processor ID"
            )
            st.session_state.docai_credentials = {
                'project_id': docai_project_id,
                'location': docai_location,
                'processor_id': docai_processor_id
            }
            if DOCUMENTAI_AVAILABLE and docai_project_id and docai_processor_id:
                st.success("✅ Document AI configured")
            elif use_document_ai and not DOCUMENTAI_AVAILABLE:
                st.warning("⚠️ Install: pip install google-cloud-documentai")

        # Pinecone Configuration
        st.markdown("##### Pinecone Vector Database")
        use_pinecone = st.checkbox(
            "Enable Pinecone",
            value=st.session_state.get("use_pinecone", False),
            help="Production-scale semantic search with metadata filtering"
        )
        st.session_state.use_pinecone = use_pinecone

        if use_pinecone:
            pinecone_api_key = st.text_input(
                "Pinecone API Key",
                type="password",
                value=st.secrets.get("PINECONE_API_KEY", ""),
                help="Get from app.pinecone.io"
            )
            pinecone_environment = st.text_input(
                "Pinecone Environment",
                value=st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1-aws"),
                help="e.g., 'us-east-1-aws'"
            )
            pinecone_index = st.text_input(
                "Index Name",
                value=st.session_state.get("pinecone_index", "fenestration-docs"),
                help="Index name for this project"
            )
            st.session_state.pinecone_credentials = {
                'api_key': pinecone_api_key,
                'environment': pinecone_environment,
                'index_name': pinecone_index
            }
            if PINECONE_AVAILABLE and pinecone_api_key:
                st.success("✅ Pinecone configured")
            elif use_pinecone and not PINECONE_AVAILABLE:
                st.warning("⚠️ Install: pip install pinecone-client")

        # Intelligence features info
        if use_document_ai or use_pinecone:
            st.info("""
            **Enhanced Capabilities:**
            - 📐 Dimension extraction from drawings
            - 🔢 Calculation understanding
            - 📊 Advanced table parsing
            - 🔍 OCR for scanned documents
            - 🎯 Entity recognition (materials, specs)
            """)

    st.markdown("---")
    # Admin authentication
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    is_admin_checked = st.checkbox("I am Admin", value=st.session_state.is_admin)
    admin_pw_ok = False
    if is_admin_checked:
        admin_secret = st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", ""))
        entered_pw = st.text_input("Admin password", type="password")
        admin_pw_ok = bool(admin_secret) and entered_pw == admin_secret
        st.session_state.is_admin = admin_pw_ok
        if admin_pw_ok:
            st.success("Admin authenticated")
        else:
            st.info("Enter admin password to manage knowledge base")
    
    # Model selection
    model_options = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-3.5-turbo",
    ]
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_options[1]
    selected_model = st.selectbox("Model", model_options, index=model_options.index(st.session_state.selected_model))
    st.session_state.selected_model = selected_model

    # Project selector
    st.markdown("---")
    existing_projects = sorted([p.name for p in Path(DATA_ROOT).glob("*") if p.is_dir()])
    col_p1, col_p2 = st.columns([3,1])
    with col_p1:
        selected_project = st.selectbox("Project", options=["Default"] + existing_projects, index=0)
    with col_p2:
        if st.button("➕ New"):
            new_name = st.text_input("New project name", key="new_project_name")
            if new_name:
                st.session_state.selected_project = new_name
                ensure_directory(get_project_dir(new_name))
        st.session_state.selected_project = selected_project

    # Engineering QA parameters
    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.session_state.temperature = st.slider("Answer temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    with col_t2:
        st.session_state.retrieval_k = st.slider("Top-K passages", 1, 20, st.session_state.retrieval_k, 1)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.session_state.chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=st.session_state.chunk_size, step=50)
    with col_c2:
        st.session_state.chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=st.session_state.chunk_overlap, step=10)

    # Knowledge base controls
    st.markdown("---")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if st.button("🔁 Reload Knowledge", help="Load persisted knowledge base from disk"):
            if RAG_AVAILABLE and api_key:
                try:
                    vs = load_persistent_vectorstore(api_key)
                    if vs is not None:
                        st.session_state.vectorstore = vs
                        st.session_state.conversation_chain = create_conversation_chain(
                            vs, api_key, st.session_state.get("selected_model", "gpt-4o-mini")
                        )
                        st.success("Reloaded knowledge base from disk")
                    else:
                        st.warning("No persisted knowledge found to load")
                except Exception as e:
                    st.error(f"Failed to reload knowledge: {str(e)}")
            # Avoid explicit rerun to prevent AttributeError on some environments
    with col_k2:
        if st.button("🧹 Clear Knowledge", help="Delete persisted vector stores from disk"):
            try:
                persist_dir, faiss_dir = get_kb_dirs_for_current_project()
                if os.path.isdir(persist_dir):
                    shutil.rmtree(persist_dir)
                if os.path.isdir(faiss_dir):
                    shutil.rmtree(faiss_dir)
                st.session_state.vectorstore = None
                st.session_state.conversation_chain = None
                st.success("Cleared knowledge base from disk")
            except Exception as e:
                st.error(f"Failed to clear knowledge base: {str(e)}")

    st.markdown("---")

    # Admin-only knowledge ingestion
    uploaded_files = None
    if st.session_state.is_admin:
        uploaded_files = st.file_uploader(
            "📎 Upload PDF Documents (Admin)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to analyze and learn from (persisted to knowledge base)"
        )
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) uploaded!")

            # Smart button text based on enabled features
            use_intelligent = (st.session_state.get("use_document_ai", False) or
                             st.session_state.get("use_pinecone", False)) and INTELLIGENT_PARSER_AVAILABLE
            button_text = "🚀 Process with Advanced Intelligence" if use_intelligent else "🧠 Process & Learn from PDFs"

            if st.button(button_text, type="primary"):
                if not api_key:
                    st.error("Please add your OpenAI API key first!")
                elif use_intelligent:
                    # Use intelligent parser for enhanced understanding
                    process_pdfs_intelligent(uploaded_files, api_key)
                elif RAG_AVAILABLE:
                    # Use standard RAG processing
                    process_pdfs(uploaded_files, api_key)
                else:
                    st.error("RAG capabilities not available. Please check dependencies.")
    
    # Show processed documents
    if st.session_state.processed_docs:
        st.markdown("### 📚 Knowledge Base")
        for doc in st.session_state.processed_docs:
            st.write(f"✅ {doc['name']} ({doc['chunks']} chunks)")
    
    st.markdown("---")
    st.markdown("### 🧠 AI Status")
    if st.session_state.vectorstore is not None:
        st.success("✅ Knowledge Base Active (persisted)")
    else:
        st.warning("⚠️ Knowledge Base inactive - load or ingest documents")
    
    st.markdown("### 📊 Stats")
    if AnalyticsManager:
        stats = AnalyticsManager.calculate_session_stats()
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Total Messages", stats.get("total_messages", 0))
        col_s2.metric("User Queries", stats.get("user_queries", 0))
        col_s3.metric("Avg Response (s)", f"{stats.get('avg_response_time', 0):.2f}")
        with st.expander("Export analytics"):
            fmt = st.selectbox("Format", ["json", "csv"], index=0)
            if st.button("Export"):
                data = AnalyticsManager.export_analytics(fmt)
                if create_download_link:
                    st.markdown(create_download_link(data, f"analytics.{fmt}"), unsafe_allow_html=True)
    else:
        st.metric("Total Queries", len(st.session_state.get('messages', [])) // 2)
        st.metric("Knowledge Base", "Active" if st.session_state.conversation_chain else "Inactive")

# Session state already initialized at the beginning of the app

def get_embeddings(api_key: str):
    """Return OpenAIEmbeddings with compatibility across versions."""
    try:
        return OpenAIEmbeddings(api_key=api_key)
    except TypeError:
        return OpenAIEmbeddings(openai_api_key=api_key)

def create_llm(api_key: str, model_name: str):
    """Return ChatOpenAI with compatibility across versions."""
    try:
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.7)
    except TypeError:
        return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.7)

def create_conversation_chain(vectorstore, api_key: str, model_name: str):
    llm = create_llm(api_key, model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        memory=memory,
        return_source_documents=True,
    )

def load_persistent_vectorstore(api_key: str):
    """Try to load a persisted vector store from disk."""
    embeddings = get_embeddings(api_key)
    persist_dir, faiss_dir = get_kb_dirs_for_current_project()
    # Prefer Chroma if present
    try:
        if os.path.isdir(persist_dir) and os.listdir(persist_dir):
            try:
                return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            except TypeError:
                return Chroma(persist_directory=persist_dir, embedding=embeddings)
    except Exception:
        pass
    # Fallback to FAISS
    try:
        if os.path.isdir(faiss_dir):
            return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        pass
    return None

# Attempt to auto-load a persisted knowledge base on startup
if RAG_AVAILABLE and (st.session_state.get("vectorstore") is None):
    api_key_autoload = st.secrets.get("OPENAI_API_KEY", "")
    if api_key_autoload:
        try:
            vs = load_persistent_vectorstore(api_key_autoload)
            if vs is not None:
                st.session_state.vectorstore = vs
                st.session_state.conversation_chain = create_conversation_chain(
                    vs, api_key_autoload, st.session_state.get("selected_model", "gpt-4o-mini")
                )
        except Exception:
            pass

def process_pdfs(uploaded_files, api_key):
    """Process PDF files and extract content"""
    with st.spinner("🧠 Processing PDFs and extracting content..."):
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            all_text = ""
            processed_docs = []
            all_documents = []
            
            # Process each PDF
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                doc_text = ""
                chunks_count = 0
                
                # Try different PDF processing methods
                if RAG_AVAILABLE and PyPDFLoader is not None:
                    try:
                        # Use PyPDFLoader if available
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()
                        doc_text = "\n".join([doc.page_content for doc in docs])
                        chunks_count = len(docs)
                        all_documents.extend(docs)
                    except Exception as e:
                        st.warning(f"PyPDFLoader failed, trying fallback: {str(e)}")
                
                # Fallback to PyPDF2 if needed
                if not doc_text and PYPDF2_AVAILABLE:
                    try:
                        with open(temp_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            page_count = len(pdf_reader.pages)
                            for page_num in range(page_count):
                                page = pdf_reader.pages[page_num]
                                page_text = page.extract_text() or ""
                                doc_text += page_text + "\n"
                                if Document is not None:
                                    all_documents.append(
                                        Document(
                                            page_content=page_text,
                                            metadata={"source": uploaded_file.name, "page": page_num + 1},
                                        )
                                    )
                            chunks_count = page_count
                    except Exception as e:
                        st.warning(f"PyPDF2 failed: {str(e)}")
                
                # If still no text, show error
                if not doc_text:
                    st.error(f"Could not extract text from {uploaded_file.name}")
                    continue
                
                all_text += f"\n\n--- FROM {uploaded_file.name} ---\n{doc_text}"
                
                # Track processed document
                processed_docs.append({
                    'name': uploaded_file.name,
                    'chunks': chunks_count,
                    'content': doc_text
                })
            
            # Store processed content in session state
            st.session_state.processed_docs = processed_docs
            st.session_state.document_content = all_text
            
            # If vector stores are available, try to create embeddings
            if VECTOR_STORE_AVAILABLE:
                try:
                    # Split accumulated documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                    )
                    if not all_documents:
                        st.error("No document content extracted. Nothing to index.")
                        shutil.rmtree(temp_dir)
                        return
                    chunks = text_splitter.split_documents(all_documents)

                    # Create embeddings and vector store
                    embeddings = get_embeddings(api_key)

                    # Try Chroma with persistence first
                    try:
                        vectorstore = None
                        try:
                            vectorstore = Chroma.from_documents(
                                documents=chunks,
                                embedding=embeddings,
                                persist_directory=persist_dir,
                            )
                        except TypeError:
                            vectorstore = Chroma.from_documents(
                                documents=chunks,
                                embedding_function=embeddings,
                                persist_directory=persist_dir,
                            )
                        # Ensure persisted
                        try:
                            vectorstore.persist()
                        except Exception:
                            pass
                    except Exception:
                        # Fallback to FAISS saved locally
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                        try:
                            vectorstore.save_local(faiss_dir)
                        except Exception:
                            pass

                    # Create conversation chain with selected model
                    conversation_chain = create_conversation_chain(
                        vectorstore, api_key, st.session_state.get("selected_model", "gpt-4o-mini")
                    )

                    # Store in session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation_chain = conversation_chain

                    st.success(
                        f"✅ Successfully processed {len(uploaded_files)} PDFs and updated the persistent knowledge base!"
                    )

                except Exception as e:
                    st.warning(f"Advanced RAG failed ({str(e)}), using basic text processing")
                    st.session_state.conversation_chain = "basic"
                    st.success(
                        f"✅ Successfully processed {len(uploaded_files)} PDFs with basic text extraction!"
                    )
            else:
                # Basic mode without vector stores
                st.session_state.conversation_chain = "basic"
                st.success(f"✅ Successfully processed {len(uploaded_files)} PDFs with basic text extraction!")
                st.info("📚 I can now answer questions based on your uploaded documents!")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            st.info("🎯 Now I can answer questions based on your specific documents!")
            
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            st.info("Make sure you have a valid OpenAI API key and try again.")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>👤 You</strong><br>{message["content"]}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Fenestration Pro AI</strong><br>{message["content"]}</div>', 
                   unsafe_allow_html=True)

# Client-side assistant area with attachments and task logging
st.markdown("---")
st.subheader("📨 Client Assistant")
col_e1, col_e2 = st.columns(2)
with col_e1:
    client_email = st.text_input("Client Email (optional)", help="Attach an email to this conversation")
with col_e2:
    ticket_id = st.text_input("Ticket/Project ID (optional)")

attachments = st.file_uploader("Attach files (PDF/images)", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True)

# Lightweight task capture
st.markdown("### ✅ Quick Task Capture")
col_ta1, col_ta2, col_ta3 = st.columns([3,1,1])
with col_ta1:
    new_task = st.text_input("Task description", key="new_task_desc")
with col_ta2:
    due_in_days = st.number_input("Due (days)", min_value=0, max_value=365, value=7, step=1)
with col_ta3:
    add_task_clicked = st.button("Add Task")

if add_task_clicked and new_task:
    project = get_selected_project()
    tasks = load_json_collection(project, "tasks")
    due_date = (date.today() + timedelta(days=int(due_in_days))).isoformat()
    tasks.append({
        "id": uuid.uuid4().hex,
        "desc": new_task,
        "due": due_date,
        "created": datetime.now().isoformat(),
        "status": "open"
    })
    save_json_collection(project, "tasks", tasks)
    st.success("Task added")

# Show current tasks
with st.expander("Project Tasks"):
    project = get_selected_project()
    tasks = load_json_collection(project, "tasks")
    if tasks:
        df = pd.DataFrame(tasks)
        st.dataframe(df[["id","desc","due","status"]], use_container_width=True)
    else:
        st.info("No tasks yet.")

# Chat input
if prompt := st.chat_input("Ask a question or describe an action..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("🧠 Processing with advanced AI..."):
        if api_key:
            try:
                # Use RAG if available and documents are processed
                if st.session_state.conversation_chain and st.session_state.conversation_chain != "basic" and RAG_AVAILABLE:
                    # Use advanced RAG system with document knowledge
                    result = st.session_state.conversation_chain({
                        "question": prompt,
                    })
                    
                    assistant_response = result["answer"]
                    
                    # Add source information
                    source_docs = result.get("source_documents", [])
                    if source_docs:
                        assistant_response += "\n\n📚 **Sources from your documents:**\n"
                        for i, doc in enumerate(source_docs[:3]):  # Show top 3 sources
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "N/A")
                            assistant_response += f"- {Path(source).name}, Page {page}\n"
                        
                        assistant_response += "\n*This answer is based on your uploaded documents.*"
                
                elif st.session_state.conversation_chain == "basic" and hasattr(st.session_state, 'document_content'):
                    # Use basic text processing with document content
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    
                    # Include document content in the prompt
                    document_context = st.session_state.document_content[:8000]  # Limit context size
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert in fenestration, windows, doors, glazing systems, and building envelope. Answer questions based on the provided document content. Be specific and cite information from the documents."},
                            {"role": "user", "content": f"Based on this document content:\n\n{document_context}\n\nQuestion: {prompt}"}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    
                    assistant_response = response.choices[0].message.content
                    assistant_response += "\n\n📄 *This answer is based on your uploaded documents.*"
                
                else:
                    # Fallback to regular OpenAI API
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        
                        # Check if we have document content even without conversation chain
                        if hasattr(st.session_state, 'document_content') and st.session_state.document_content:
                            # Use document content if available
                            document_context = st.session_state.document_content[:8000]
                            
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are an expert in fenestration, windows, doors, glazing systems, and building envelope. Answer questions based on the provided document content. Be specific and cite information from the documents."},
                                    {"role": "user", "content": f"Based on this document content:\n\n{document_context}\n\nQuestion: {prompt}"}
                                ],
                                temperature=0.7,
                                max_tokens=800
                            )
                            
                            assistant_response = response.choices[0].message.content
                            assistant_response += "\n\n📄 *This answer is based on your uploaded documents.*"
                        else:
                            # No documents processed yet
                            context = ""
                            if st.session_state.processed_docs:
                                doc_names = [doc['name'] for doc in st.session_state.processed_docs]
                                context = f"\n\nNote: The user has uploaded these documents: {', '.join(doc_names)}. However, I cannot access their content directly. Please process the documents first using the 'Process & Learn from PDFs' button."
                            
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are an expert in fenestration, windows, doors, glazing systems, and building envelope. Provide detailed, technical answers."},
                                    {"role": "user", "content": prompt + context}
                                ],
                                temperature=0.7,
                                max_tokens=800
                            )
                            
                            assistant_response = response.choices[0].message.content
                    except Exception as e:
                        # If new API fails, provide basic response
                        assistant_response = f"I'm having trouble connecting to the AI service. Error: {str(e)}\n\nPlease make sure your OpenAI API key is valid."
                    
                    # Add helpful message if documents are uploaded but not processed
                    if st.session_state.processed_docs and not st.session_state.conversation_chain:
                        assistant_response += "\n\n💡 **Tip**: Click 'Process & Learn from PDFs' in the sidebar to enable me to answer questions based on your specific documents!"
                
            except Exception as e:
                assistant_response = f"Error: {str(e)}. Please check your API key or try again."
        else:
            assistant_response = "Please add your OpenAI API key in the sidebar to enable AI responses."
    
    # Save attachments
    saved_meta = save_attachments(attachments, category="client") if attachments else []

    # Add assistant response and optionally append metadata for email/ticket
    meta_prefix = ""
    if client_email or ticket_id:
        meta = []
        if client_email:
            meta.append(f"Email: {client_email}")
        if ticket_id:
            meta.append(f"Ticket: {ticket_id}")
        if saved_meta:
            meta.append(f"Attachments: {len(saved_meta)}")
        meta_prefix = "[" + ", ".join(meta) + "]\n\n"
    st.session_state.messages.append({"role": "assistant", "content": meta_prefix + assistant_response})
    # Avoid rerun calls to prevent AttributeError in certain hosting environments

# PM Tabs: RFIs, Submittals, Schedule, Documents, Contacts, Reports
st.markdown("---")
st.subheader("🗂️ Project Management")
tab_rfis, tab_submittals, tab_schedule, tab_documents, tab_contacts, tab_reports, tab_bqe = st.tabs([
    "RFIs", "Submittals", "Schedule", "Documents", "Contacts", "Reports", "BQE"
])

with tab_rfis:
    st.markdown("#### RFIs")
    with st.form("rfi_form"):
        rfi_subject = st.text_input("Subject")
        rfi_question = st.text_area("Question / Clarification Needed", height=120)
        rfi_due = st.date_input("Due date", value=date.today() + timedelta(days=7))
        rfi_files = st.file_uploader("Attach files (optional)", accept_multiple_files=True)
        submitted = st.form_submit_button("Create RFI")
    if submitted and rfi_subject and rfi_question:
        project = get_selected_project()
        rfis = load_json_collection(project, "rfis")
        attachments_meta = save_attachments(rfi_files, category="rfis") if rfi_files else []
        rfis.append({
            "id": uuid.uuid4().hex,
            "subject": rfi_subject,
            "question": rfi_question,
            "due": rfi_due.isoformat(),
            "status": "open",
            "created": datetime.now().isoformat(),
            "attachments": attachments_meta,
        })
        save_json_collection(project, "rfis", rfis)
        st.success("RFI created")

    # Show RFIs
    project = get_selected_project()
    rfis = load_json_collection(project, "rfis")
    if rfis:
        df = pd.DataFrame(rfis)
        st.dataframe(df[["id", "subject", "due", "status"]], use_container_width=True)
        with st.expander("Update RFI status"):
            sel = st.selectbox("RFI", options=[f"{r['id']} — {r['subject'][:40]}" for r in rfis], index=0)
            new_status = st.selectbox("Status", options=["open", "answered", "closed"], index=0)
            if st.button("Update RFI"):
                sel_id = sel.split(" — ")[0]
                for r in rfis:
                    if r["id"] == sel_id:
                        r["status"] = new_status
                        r["updated"] = datetime.now().isoformat()
                        break
                save_json_collection(project, "rfis", rfis)
                st.success("RFI updated")
    else:
        st.info("No RFIs yet.")

with tab_submittals:
    st.markdown("#### Submittals")
    with st.form("submittal_form"):
        sub_title = st.text_input("Title")
        sub_spec = st.text_input("Spec Section")
        sub_status = st.selectbox("Status", ["draft", "submitted", "approved", "rejected"], index=0)
        sub_files = st.file_uploader("Attach files (optional)", accept_multiple_files=True)
        submitted = st.form_submit_button("Add Submittal")
    if submitted and sub_title:
        project = get_selected_project()
        subs = load_json_collection(project, "submittals")
        attachments_meta = save_attachments(sub_files, category="submittals") if sub_files else []
        subs.append({
            "id": uuid.uuid4().hex,
            "title": sub_title,
            "spec": sub_spec,
            "status": sub_status,
            "created": datetime.now().isoformat(),
            "attachments": attachments_meta,
        })
        save_json_collection(project, "submittals", subs)
        st.success("Submittal added")

    # Show Submittals
    project = get_selected_project()
    subs = load_json_collection(project, "submittals")
    if subs:
        df = pd.DataFrame(subs)
        st.dataframe(df[["id", "title", "spec", "status"]], use_container_width=True)
        with st.expander("Update Submittal status"):
            sel = st.selectbox("Submittal", options=[f"{s['id']} — {s['title'][:40]}" for s in subs], index=0)
            new_status = st.selectbox("Status", ["draft", "submitted", "approved", "rejected"], index=0)
            if st.button("Update Submittal"):
                sel_id = sel.split(" — ")[0]
                for s in subs:
                    if s["id"] == sel_id:
                        s["status"] = new_status
                        s["updated"] = datetime.now().isoformat()
                        break
                save_json_collection(project, "submittals", subs)
                st.success("Submittal updated")
    else:
        st.info("No submittals yet.")

with tab_schedule:
    st.markdown("#### Schedule / Milestones")
    with st.form("milestone_form"):
        ms_name = st.text_input("Milestone name")
        ms_date = st.date_input("Date", value=date.today() + timedelta(days=14))
        ms_submit = st.form_submit_button("Add Milestone")
    if ms_submit and ms_name:
        project = get_selected_project()
        mss = load_json_collection(project, "milestones")
        mss.append({
            "id": uuid.uuid4().hex,
            "name": ms_name,
            "date": ms_date.isoformat(),
            "created": datetime.now().isoformat(),
        })
        save_json_collection(project, "milestones", mss)
        st.success("Milestone added")
    # Show milestones
    project = get_selected_project()
    mss = load_json_collection(project, "milestones")
    if mss:
        df = pd.DataFrame(mss)
        st.dataframe(df[["id", "name", "date"]], use_container_width=True)
    else:
        st.info("No milestones yet.")

with tab_documents:
    st.markdown("#### Documents")
    project = get_selected_project()
    # Ingested docs log
    docs_log = load_json_collection(project, "ingested_docs")
    if docs_log:
        st.write("Ingested Knowledge Base Documents:")
        df = pd.DataFrame(docs_log)
        st.dataframe(df[["name", "chunks", "ingested_at"]], use_container_width=True)
    else:
        st.info("No ingested documents logged yet.")
    # Uploaded files overview
    uploads_dir = os.path.join(get_project_dir(project), "uploads")
    if os.path.isdir(uploads_dir):
        files = []
        for root, _dirs, filenames in os.walk(uploads_dir):
            for fn in filenames:
                files.append({"file": fn, "path": os.path.join(root, fn)})
        if files:
            st.write("Uploaded Files:")
            st.dataframe(pd.DataFrame(files), use_container_width=True)
    
with tab_contacts:
    st.markdown("#### Contacts / Stakeholders")
    with st.form("contact_form"):
        c_name = st.text_input("Name")
        c_role = st.text_input("Role")
        c_email = st.text_input("Email")
        c_org = st.text_input("Organization")
        c_submit = st.form_submit_button("Add Contact")
    if c_submit and c_name:
        project = get_selected_project()
        contacts = load_json_collection(project, "contacts")
        contacts.append({
            "id": uuid.uuid4().hex,
            "name": c_name,
            "role": c_role,
            "email": c_email,
            "org": c_org,
            "created": datetime.now().isoformat(),
        })
        save_json_collection(project, "contacts", contacts)
        st.success("Contact added")
    # Show contacts
    project = get_selected_project()
    contacts = load_json_collection(project, "contacts")
    if contacts:
        st.dataframe(pd.DataFrame(contacts)[["id", "name", "role", "email", "org"]], use_container_width=True)
    else:
        st.info("No contacts yet.")

with tab_reports:
    st.markdown("#### Reports & Export")
    project = get_selected_project()
    snapshot = {
        "project": project,
        "generated_at": datetime.now().isoformat(),
        "tasks": load_json_collection(project, "tasks"),
        "rfis": load_json_collection(project, "rfis"),
        "submittals": load_json_collection(project, "submittals"),
        "milestones": load_json_collection(project, "milestones"),
        "contacts": load_json_collection(project, "contacts"),
        "ingested_docs": load_json_collection(project, "ingested_docs"),
    }
    json_str = json.dumps(snapshot, indent=2)
    if create_download_link:
        st.markdown(create_download_link(json_str, f"{project}_snapshot.json", "application/json"), unsafe_allow_html=True)
    st.download_button("Download JSON", data=json_str, file_name=f"{project}_snapshot.json", mime="application/json")

# BQE OAuth Helper Functions
def get_oauth_url():
    """Generate BQE OAuth authorization URL"""
    import secrets
    from urllib.parse import urlencode
    
    state = secrets.token_urlsafe(32)
    st.session_state.oauth_state = state
    
    # According to BQE Core docs, the correct OAuth flow parameters
    params = {
        "response_type": "code",
        "client_id": st.session_state.bqe_client_id,
        "redirect_uri": "https://fenestrationpro.streamlit.app/",
        "scope": "openid offline_access api",  # Updated scopes per BQE docs
        "state": state
    }
    
    # BQE OAuth authorization endpoint
    # Use configured URL or default
    if hasattr(st.session_state, 'bqe_auth_url') and st.session_state.bqe_auth_url:
        auth_endpoint = st.session_state.bqe_auth_url
    else:
        # Default to core.bqe.com
        auth_endpoint = "https://core.bqe.com/identity/connect/authorize"
    return f"{auth_endpoint}?{urlencode(params)}"

def exchange_code_for_token(code):
    """Exchange authorization code for access token"""
    # BQE token endpoint
    if hasattr(st.session_state, 'bqe_token_url') and st.session_state.bqe_token_url:
        token_endpoint = st.session_state.bqe_token_url
    else:
        token_endpoint = "https://core.bqe.com/identity/connect/token"
    
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": "https://fenestrationpro.streamlit.app/",
        "client_id": st.session_state.bqe_client_id,
        "client_secret": st.session_state.bqe_client_secret
    }
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    response = requests.post(token_endpoint, data=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Token exchange failed: {response.status_code} - {response.text}")
        return None

def refresh_access_token():
    """Refresh the access token using refresh token"""
    if not st.session_state.get("bqe_refresh_token"):
        return None
    
    # BQE token endpoint
    if hasattr(st.session_state, 'bqe_token_url') and st.session_state.bqe_token_url:
        token_endpoint = st.session_state.bqe_token_url
    else:
        token_endpoint = "https://core.bqe.com/identity/connect/token"
    
    data = {
        "grant_type": "refresh_token",
        "refresh_token": st.session_state.bqe_refresh_token,
        "client_id": st.session_state.bqe_client_id,
        "client_secret": st.session_state.bqe_client_secret
    }
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    response = requests.post(token_endpoint, data=data, headers=headers)
    
    if response.status_code == 200:
        token_data = response.json()
        st.session_state.bqe_token = token_data.get("access_token", "")
        if "refresh_token" in token_data:
            st.session_state.bqe_refresh_token = token_data["refresh_token"]
        return True
    else:
        return False

# Simple BQE tab with direct integration
with tab_bqe:
    st.markdown("#### BQE Core Integration")
    
    # Initialize session state for BQE with hardcoded token
    if "bqe_base_url" not in st.session_state:
        st.session_state.bqe_base_url = "https://api.bqecore.com/api"
    if "bqe_token" not in st.session_state:
        # Use the provided token directly
        st.session_state.bqe_token = "qiXSQ2uKoeF9b5M7bOKtRYNpBxBaVw1c955M0fFU_ldZ2cjovtMSlkbT28aJaBPl"
    
    st.success("✅ BQE Core integration is configured")
    
    # Show current configuration
    with st.expander("⚙️ Configuration", expanded=False):
        st.markdown(f"**API Base URL**: `{st.session_state.bqe_base_url}`")
        st.markdown(f"**API Token**: `{st.session_state.bqe_token[:20]}...`")
        
        # Allow manual override
        new_url = st.text_input("API Base URL", value=st.session_state.bqe_base_url)
        if new_url != st.session_state.bqe_base_url:
            st.session_state.bqe_base_url = new_url
            st.rerun()
        
        new_token = st.text_input("API Token", value=st.session_state.bqe_token, type="password")
        if new_token != st.session_state.bqe_token:
            st.session_state.bqe_token = new_token
            st.rerun()
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔌 Test Connection", type="primary"):
            with st.spinner("Testing BQE connection..."):
                try:
                    # Try different authentication methods
                    # Method 1: Bearer token
                    headers = {
                        "Authorization": f"Bearer {st.session_state.bqe_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                    
                    # Test with employee endpoint
                    test_url = f"{st.session_state.bqe_base_url}/employee"
                    response = requests.get(test_url, headers=headers, timeout=10)
                    
                    if response.status_code == 401:
                        # Try Method 2: API Key header
                        headers = {
                            "X-API-Key": st.session_state.bqe_token,
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                        response = requests.get(test_url, headers=headers, timeout=10)
                    
                    if response.status_code == 401:
                        # Try Method 3: Query parameter
                        response = requests.get(f"{test_url}?apikey={st.session_state.bqe_token}", 
                                              headers={"Content-Type": "application/json"}, timeout=10)
                    
                    if response.status_code == 200:
                        st.success("✅ Successfully connected to BQE Core!")
                        try:
                            data = response.json()
                            with st.expander("Connection Details"):
                                st.json(data if isinstance(data, dict) else data[0] if data else {})
                        except:
                            st.info("Connected successfully")
                    elif response.status_code == 401:
                        st.error("❌ Authentication failed. The token may be invalid.")
                        st.info("The provided string might be a Client Secret, not an API token. You may need to obtain an access token through OAuth.")
                    else:
                        st.error(f"❌ Connection failed: {response.status_code}")
                        if response.text:
                            with st.expander("Error details"):
                                st.code(response.text[:500])
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: {str(e)}")
    
    with col2:
        if st.button("📥 Import Projects & Contacts"):
            with st.spinner("Importing data from BQE..."):
                try:
                    # Use the same auth method that works for test connection
                    headers = {
                        "Authorization": f"Bearer {st.session_state.bqe_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                    
                    imported_data = {
                        "projects": 0,
                        "contacts": 0
                    }
                    
                    # Import Projects
                    try:
                        projects_url = f"{st.session_state.bqe_base_url}/project"
                        response = requests.get(projects_url, headers=headers, timeout=10)
                        if response.status_code == 200:
                            bqe_projects = response.json()
                            if isinstance(bqe_projects, list):
                                projects_list = bqe_projects
                            elif isinstance(bqe_projects, dict) and 'data' in bqe_projects:
                                projects_list = bqe_projects['data']
                            else:
                                projects_list = []
                            
                            # Convert BQE Core projects to our format
                            for bqe_project in projects_list:
                                # BQE Core uses different field names
                                project_name = bqe_project.get('projectName', bqe_project.get('name', f"BQE Project {bqe_project.get('projectID', bqe_project.get('id', 'Unknown'))}"))
                                project_dir = get_project_dir(project_name)
                                ensure_directory(project_dir)
                                
                                # Store project metadata with BQE Core fields
                                project_meta = {
                                    "bqe_id": bqe_project.get('projectID', bqe_project.get('id')),
                                    "name": project_name,
                                    "description": bqe_project.get('description', ''),
                                    "client": bqe_project.get('clientName', ''),
                                    "status": bqe_project.get('projectStatus', bqe_project.get('status', 'active')),
                                    "imported_from_bqe": True,
                                    "imported_at": datetime.now().isoformat()
                                }
                                
                                # Save to project metadata
                                meta_file = os.path.join(project_dir, "project_meta.json")
                                with open(meta_file, 'w') as f:
                                    json.dump(project_meta, f, indent=2)
                                
                                imported_data["projects"] += 1
                    except Exception as e:
                        st.warning(f"Failed to import projects: {str(e)}")
                    
                    # Import Contacts
                    try:
                        # BQE Core uses 'contact' endpoint
                        contacts_url = f"{st.session_state.bqe_base_url}/contact"
                        response = requests.get(contacts_url, headers=headers, timeout=10)
                        if response.status_code == 200:
                            bqe_contacts = response.json()
                            if isinstance(bqe_contacts, list):
                                contacts_list = bqe_contacts
                            elif isinstance(bqe_contacts, dict) and 'data' in bqe_contacts:
                                contacts_list = bqe_contacts['data']
                            else:
                                contacts_list = []
                            
                            # Add BQE contacts to current project
                            project = get_selected_project()
                            existing_contacts = load_json_collection(project, "contacts")
                            
                            for bqe_contact in contacts_list:
                                # BQE Core uses different field names
                                full_name = f"{bqe_contact.get('firstName', '')} {bqe_contact.get('lastName', '')}".strip()
                                if not full_name:
                                    full_name = bqe_contact.get('fullName', bqe_contact.get('name', 'Unknown'))
                                
                                contact = {
                                    "id": uuid.uuid4().hex,
                                    "bqe_id": bqe_contact.get('contactID', bqe_contact.get('id')),
                                    "name": full_name,
                                    "role": bqe_contact.get('title', ''),
                                    "email": bqe_contact.get('email1', bqe_contact.get('email', '')),
                                    "org": bqe_contact.get('companyName', bqe_contact.get('company', '')),
                                    "phone": bqe_contact.get('mobilePhone', bqe_contact.get('phone1', bqe_contact.get('phone', ''))),
                                    "imported_from_bqe": True,
                                    "created": datetime.now().isoformat()
                                }
                                existing_contacts.append(contact)
                                imported_data["contacts"] += 1
                            
                            save_json_collection(project, "contacts", existing_contacts)
                    except Exception as e:
                        st.warning(f"Failed to import contacts: {str(e)}")
                    
                    # Show results
                    if imported_data["projects"] > 0 or imported_data["contacts"] > 0:
                        st.success(f"✅ Imported {imported_data['projects']} projects and {imported_data['contacts']} contacts from BQE")
                    else:
                        st.warning("No data was imported. Please check your BQE API configuration.")
                        
                except Exception as e:
                    st.error(f"❌ Import failed: {str(e)}")
    
    # Show BQE sync status
    st.markdown("---")
    st.markdown("##### BQE Sync Status")
    project = get_selected_project()
    project_dir = get_project_dir(project)
    meta_file = os.path.join(project_dir, "project_meta.json")
    
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            project_meta = json.load(f)
            if project_meta.get('imported_from_bqe'):
                st.info(f"This project was imported from BQE on {project_meta.get('imported_at', 'Unknown date')}")
                if project_meta.get('bqe_id'):
                    st.text(f"BQE Project ID: {project_meta['bqe_id']}")
    
    # Show imported contacts with BQE flag
    contacts = load_json_collection(project, "contacts")
    bqe_contacts = [c for c in contacts if c.get('imported_from_bqe')]
    if bqe_contacts:
        st.markdown(f"##### BQE Contacts ({len(bqe_contacts)})")
        df = pd.DataFrame(bqe_contacts)
        display_cols = ["name", "role", "email", "org"]
        if all(col in df.columns for col in display_cols):
            st.dataframe(df[display_cols], use_container_width=True)
    
    # Help and Status
    with st.expander("⚠️ IMPORTANT: BQE Authentication Issue", expanded=True):
        st.error("""
        **The BQE integration is NOT working because:**
        
        1. **Wrong Credential Type**: The string you provided (`qiXSQ2uKoeF9b5M7bOKtRYNpBxBaVw1c955M0fFU_ldZ2cjovtMSlkbT28aJaBPl`) 
           is an OAuth Client Secret, NOT an API access token.
        
        2. **OAuth Domain Issue**: Your Client ID uses `apps.bqe.com` which doesn't exist (DNS error).
        
        3. **What You Need**: Either:
           - A valid API access token (not a client secret)
           - OR the correct OAuth URLs for your BQE instance
        """)
        
        st.markdown("""
        ### 🔧 How to Fix This:
        
        **Option 1: Get an API Token**
        1. Log into your BQE Core account
        2. Go to Settings → API → Access Tokens
        3. Generate a new API token
        4. Update it in the Configuration section above
        
        **Option 2: Fix OAuth**
        1. Find the correct OAuth URL from your BQE admin
        2. It might be something like:
           - `https://yourcompany.bqecore.com`
           - `https://login.bqecore.com`
           - `https://auth.bqe.com`
        3. Contact BQE support if you can't find it
        
        **Your OAuth App Info:**
        - Client ID: `U2pwazJCTFbCq7Re6VkR31YQc48pcL_O.apps.bqe.com`
        - Client Secret: The 64-character string you provided
        - Redirect URI: `https://fenestrationpro.streamlit.app/`
        """)
        
        new_url = st.text_input(
            "Override BQE Base URL",
            value="",
            help="Leave blank to use default URL"
        )
        if new_url:
            st.session_state.bqe_base_url = new_url
            st.success("Using custom base URL")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>🚀 Fenestration Pro AI - State-of-the-Art Edition</h4>
    <p>Powered by Advanced AI • OpenAI GPT • Intelligent Document Processing</p>
    <p>Features: Smart Chat • Document Analysis • Expert Knowledge • Modern UI</p>
    <a href='https://github.com/administrator2023/fenestration-pro-ai' style='color: #667eea;'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)
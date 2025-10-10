import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path

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

# Persistent storage locations
PERSIST_DIR = "./chroma_db"
FAISS_DIR = "./faiss_index"

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
    try:
        if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            # Try with embedding_function + client_settings
            try:
                return Chroma(
                    persist_directory=PERSIST_DIR,
                    embedding_function=embeddings,
                    client_settings=CHROMA_CLIENT_SETTINGS,
                )
            except TypeError:
                # Try with embedding + client_settings
                try:
                    return Chroma(
                        persist_directory=PERSIST_DIR,
                        embedding=embeddings,
                        client_settings=CHROMA_CLIENT_SETTINGS,
                    )
                except TypeError:
                    # Fallback without client_settings
                    try:
                        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
                    except TypeError:
                        return Chroma(persist_directory=PERSIST_DIR, embedding=embeddings)
    except Exception:
        pass
    # Fallback to FAISS
    try:
        if os.path.isdir(FAISS_DIR):
            return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        pass
    return None

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

                    # Try Chroma with persistence first
                    try:
                        vectorstore = None
                        # 1) embedding + client_settings
                        try:
                            vectorstore = Chroma.from_documents(
                                documents=chunks,
                                embedding=embeddings,
                                persist_directory=PERSIST_DIR,
                                client_settings=CHROMA_CLIENT_SETTINGS,
                            )
                        except TypeError:
                            # 2) embedding_function + client_settings
                            try:
                                vectorstore = Chroma.from_documents(
                                    documents=chunks,
                                    embedding_function=embeddings,
                                    persist_directory=PERSIST_DIR,
                                    client_settings=CHROMA_CLIENT_SETTINGS,
                                )
                            except TypeError:
                                # 3) embedding without client_settings
                                try:
                                    vectorstore = Chroma.from_documents(
                                        documents=chunks,
                                        embedding=embeddings,
                                        persist_directory=PERSIST_DIR,
                                    )
                                except TypeError:
                                    # 4) embedding_function without client_settings
                                    vectorstore = Chroma.from_documents(
                                        documents=chunks,
                                        embedding_function=embeddings,
                                        persist_directory=PERSIST_DIR,
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
                            vectorstore.save_local(FAISS_DIR)
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
                if os.path.isdir(PERSIST_DIR):
                    shutil.rmtree(PERSIST_DIR)
                if os.path.isdir(FAISS_DIR):
                    shutil.rmtree(FAISS_DIR)
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
            if st.button("🧠 Process & Learn from PDFs", type="primary"):
                if RAG_AVAILABLE and api_key:
                    process_pdfs(uploaded_files, api_key)
                elif not api_key:
                    st.error("Please add your OpenAI API key first!")
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
    # Prefer Chroma if present
    try:
        if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            try:
                return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            except TypeError:
                return Chroma(persist_directory=PERSIST_DIR, embedding=embeddings)
    except Exception:
        pass
    # Fallback to FAISS
    try:
        if os.path.isdir(FAISS_DIR):
            return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
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
                                persist_directory=PERSIST_DIR,
                            )
                        except TypeError:
                            vectorstore = Chroma.from_documents(
                                documents=chunks,
                                embedding_function=embeddings,
                                persist_directory=PERSIST_DIR,
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
                            vectorstore.save_local(FAISS_DIR)
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

# Client-side assistant area with email attachment metadata
st.markdown("---")
st.subheader("📨 Client Assistant")
col_e1, col_e2 = st.columns(2)
with col_e1:
    client_email = st.text_input("Client Email (optional)", help="Attach an email to this conversation")
with col_e2:
    ticket_id = st.text_input("Ticket/Project ID (optional)")

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
    
    # Add assistant response and optionally append metadata for email/ticket
    meta_prefix = ""
    if client_email or ticket_id:
        meta = []
        if client_email:
            meta.append(f"Email: {client_email}")
        if ticket_id:
            meta.append(f"Ticket: {ticket_id}")
        meta_prefix = "[" + ", ".join(meta) + "]\n\n"
    st.session_state.messages.append({"role": "assistant", "content": meta_prefix + assistant_response})
    # Avoid rerun calls to prevent AttributeError in certain hosting environments

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
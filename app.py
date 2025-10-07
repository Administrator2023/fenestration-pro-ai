import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path

# RAG imports
try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.chat_models import ChatOpenAI
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

st.set_page_config(
    page_title="Fenestration Pro AI",
    page_icon="🏗️",
    layout="wide"
)

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
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📎 Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents to analyze and learn from"
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
    if st.session_state.conversation_chain:
        st.success("✅ Knowledge Base Active")
        st.info("I can now answer questions from your documents!")
    else:
        st.warning("⚠️ Basic Mode")
        st.info("Upload & process PDFs for document-specific answers")
    
    st.markdown("### 📊 Stats")
    st.metric("Total Queries", len(st.session_state.get('messages', [])) // 2)
    st.metric("Knowledge Base", "Active" if st.session_state.conversation_chain else "Inactive")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🚀 Welcome to Fenestration Pro AI - State-of-the-Art Edition! I'm here to help answer your questions about windows, doors, and building envelope systems. Upload a PDF and I'll learn from it to give you specific answers!"
    })

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

def process_pdfs(uploaded_files, api_key):
    """Process PDF files and create RAG knowledge base"""
    with st.spinner("🧠 Processing PDFs and building knowledge base..."):
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            all_documents = []
            
            # Process each PDF
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load PDF
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                all_documents.extend(documents)
                
                # Track processed document
                st.session_state.processed_docs.append({
                    'name': uploaded_file.name,
                    'chunks': len(documents)
                })
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(all_documents)
            
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
                model_name="gpt-3.5-turbo",
                temperature=0.7
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
            
            # Store in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation_chain = conversation_chain
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            st.success(f"✅ Successfully processed {len(uploaded_files)} PDFs into {len(chunks)} knowledge chunks!")
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

# Chat input
if prompt := st.chat_input("Ask about fenestration, windows, doors, or upload a PDF..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("🧠 Processing with advanced AI..."):
        if api_key:
            try:
                # Use RAG if available and documents are processed
                if st.session_state.conversation_chain and RAG_AVAILABLE:
                    # Use RAG system with document knowledge
                    result = st.session_state.conversation_chain({
                        "question": prompt
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
                
                else:
                    # Fallback to regular OpenAI API
                    import openai
                    openai.api_key = api_key
                    
                    # Include context about uploaded files
                    context = ""
                    if st.session_state.processed_docs:
                        doc_names = [doc['name'] for doc in st.session_state.processed_docs]
                        context = f"\n\nNote: The user has uploaded these documents: {', '.join(doc_names)}. However, I cannot access their content directly. Please process the documents first using the 'Process & Learn from PDFs' button."
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert in fenestration, windows, doors, glazing systems, and building envelope. Provide detailed, technical answers."},
                            {"role": "user", "content": prompt + context}
                        ],
                        temperature=0.7,
                        max_tokens=800
                    )
                    
                    assistant_response = response.choices[0].message['content']
                    
                    # Add helpful message if documents are uploaded but not processed
                    if st.session_state.processed_docs and not st.session_state.conversation_chain:
                        assistant_response += "\n\n💡 **Tip**: Click 'Process & Learn from PDFs' in the sidebar to enable me to answer questions based on your specific documents!"
                
            except Exception as e:
                assistant_response = f"Error: {str(e)}. Please check your API key or try again."
        else:
            assistant_response = "Please add your OpenAI API key in the sidebar to enable AI responses."
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.rerun()

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
"""
Advanced RAG (Retrieval Augmented Generation) Pipeline
State-of-the-art document processing and retrieval system
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil
import hashlib
from datetime import datetime

# Core imports
import streamlit as st
import pandas as pd
import numpy as np

# Document processing
from langchain.document_loaders import (
    PyPDFLoader, UnstructuredPDFLoader, PDFMinerLoader,
    Docx2txtLoader, UnstructuredWordDocumentLoader,
    TextLoader, CSVLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter
)

# Embeddings and Vector Stores
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings
)
from langchain.vectorstores import (
    Chroma,
    FAISS,
    Pinecone
)

# LLM and Chains
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import (
    ConversationalRetrievalChain,
    RetrievalQA,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# Prompts and Tools
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Advanced features
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from keybert import KeyBERT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Enhanced document metadata"""
    filename: str
    file_type: str
    file_size: int
    upload_time: datetime
    processing_time: float
    chunk_count: int
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    keywords: Optional[List[str]] = None
    summary: Optional[str] = None
    hash: Optional[str] = None

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with metadata"""
    content: str
    source: str
    page: Optional[int]
    score: float
    chunk_id: str
    metadata: Dict[str, Any]

class AdvancedDocumentProcessor:
    """Advanced document processing with multiple strategies"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': [PyPDFLoader, UnstructuredPDFLoader, PDFMinerLoader],
            '.docx': [Docx2txtLoader, UnstructuredWordDocumentLoader],
            '.txt': [TextLoader],
            '.csv': [CSVLoader]
        }
        
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.kw_model = KeyBERT()
    
    def process_document(self, file_path: str, loader_type: str = "auto") -> Tuple[List[Document], DocumentMetadata]:
        """Process document with enhanced metadata extraction"""
        start_time = datetime.now()
        
        # Determine file type and loader
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Select best loader
        loaders = self.supported_formats[file_ext]
        loader_class = loaders[0]  # Default to first loader
        
        if loader_type == "unstructured" and len(loaders) > 1:
            loader_class = loaders[1]
        elif loader_type == "advanced" and len(loaders) > 2:
            loader_class = loaders[2]
        
        # Load document
        loader = loader_class(file_path)
        documents = loader.load()
        
        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Extract metadata
        file_stats = Path(file_path).stat()
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract text for analysis
        full_text = " ".join([doc.page_content for doc in documents])
        
        # Language detection and keyword extraction
        language = self._detect_language(full_text)
        keywords = self._extract_keywords(full_text)
        summary = self._generate_summary(full_text)
        
        metadata = DocumentMetadata(
            filename=Path(file_path).name,
            file_type=file_ext,
            file_size=file_stats.st_size,
            upload_time=datetime.now(),
            processing_time=processing_time,
            chunk_count=len(documents),
            page_count=len(documents) if file_ext == '.pdf' else None,
            word_count=len(full_text.split()),
            language=language,
            keywords=keywords,
            summary=summary,
            hash=file_hash
        )
        
        return documents, metadata
    
    def _detect_language(self, text: str) -> str:
        """Detect document language"""
        if self.nlp:
            doc = self.nlp(text[:1000])  # Sample first 1000 chars
            return doc.lang_
        return "en"  # Default to English
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text"""
        try:
            keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), 
                                                    stop_words='english', top_k=top_k)
            return [kw[0] for kw in keywords]
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate document summary"""
        # Simple extractive summary - first few sentences
        sentences = text.split('. ')
        summary = '. '.join(sentences[:3])
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary

class AdvancedTextSplitter:
    """Advanced text splitting with multiple strategies"""
    
    def __init__(self):
        self.splitters = {
            'recursive': RecursiveCharacterTextSplitter,
            'token': TokenTextSplitter,
            'spacy': SpacyTextSplitter,
            'nltk': NLTKTextSplitter
        }
    
    def split_documents(self, documents: List[Document], 
                       strategy: str = 'recursive',
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200,
                       **kwargs) -> List[Document]:
        """Split documents using specified strategy"""
        
        if strategy not in self.splitters:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
        
        splitter_class = self.splitters[strategy]
        
        if strategy == 'recursive':
            splitter = splitter_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif strategy == 'token':
            splitter = splitter_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif strategy == 'spacy':
            splitter = splitter_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif strategy == 'nltk':
            splitter = splitter_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        chunks = splitter.split_documents(documents)
        
        # Add chunk IDs and enhanced metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': f"chunk_{i}",
                'chunk_index': i,
                'total_chunks': len(chunks),
                'splitting_strategy': strategy
            })
        
        return chunks

class MultiVectorRetriever:
    """Advanced retrieval with multiple vector stores and strategies"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.vector_stores = {}
        self.embeddings_models = {
            'openai': OpenAIEmbeddings(openai_api_key=api_key),
            'sentence_transformers': SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            ),
            'huggingface': HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        }
    
    def create_vector_store(self, documents: List[Document], 
                          store_type: str = 'chroma',
                          embedding_model: str = 'openai',
                          store_name: str = 'default') -> Any:
        """Create vector store with specified configuration"""
        
        embeddings = self.embeddings_models[embedding_model]
        
        if store_type == 'chroma':
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=f"./vector_stores/{store_name}_chroma"
            )
        elif store_type == 'faiss':
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            # Save FAISS index
            vector_store.save_local(f"./vector_stores/{store_name}_faiss")
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        self.vector_stores[store_name] = vector_store
        return vector_store
    
    def hybrid_search(self, query: str, 
                     store_names: List[str] = None,
                     top_k: int = 5,
                     score_threshold: float = 0.7) -> List[RetrievalResult]:
        """Perform hybrid search across multiple vector stores"""
        
        if store_names is None:
            store_names = list(self.vector_stores.keys())
        
        all_results = []
        
        for store_name in store_names:
            if store_name not in self.vector_stores:
                continue
            
            vector_store = self.vector_stores[store_name]
            
            # Similarity search with scores
            results = vector_store.similarity_search_with_score(query, k=top_k)
            
            for doc, score in results:
                if score >= score_threshold:
                    result = RetrievalResult(
                        content=doc.page_content,
                        source=doc.metadata.get('source', 'unknown'),
                        page=doc.metadata.get('page'),
                        score=score,
                        chunk_id=doc.metadata.get('chunk_id', ''),
                        metadata=doc.metadata
                    )
                    all_results.append(result)
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

class AdvancedConversationChain:
    """Advanced conversation chain with multiple LLM strategies"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Custom prompts for fenestration
        self.fenestration_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in fenestration, windows, doors, glazing systems, and building envelope technology. 
            You have deep knowledge of:
            - Window and door manufacturing processes and materials
            - Glass types, coatings, and performance characteristics
            - Energy efficiency standards (ENERGY STAR, NFRC ratings, SHGC, U-values)
            - Building codes and regulations (IBC, IRC, AAMA, NFRC)
            - Installation best practices and weatherization
            - Thermal performance and condensation control
            - Air leakage and water penetration testing
            - Structural glazing and curtain wall systems
            - Hardware, operators, and security features
            - Sustainable and green building practices
            
            Use the provided context to give detailed, technical answers. Always cite sources when available.
            """),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
    
    def create_chain(self, retriever: Any, chain_type: str = "stuff") -> Any:
        """Create conversation chain with retriever"""
        
        if chain_type == "stuff":
            # Stuff chain - puts all retrieved docs into prompt
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
        elif chain_type == "map_reduce":
            # Map-reduce chain for large documents
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                chain_type="map_reduce"
            )
        elif chain_type == "refine":
            # Refine chain for iterative improvement
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                chain_type="refine"
            )
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")
        
        return chain
    
    async def aquery(self, chain: Any, question: str) -> Dict[str, Any]:
        """Async query for better performance"""
        try:
            result = await chain.arun({"question": question})
            return result
        except Exception as e:
            logger.error(f"Async query failed: {e}")
            # Fallback to sync
            return chain({"question": question})

class RAGPipeline:
    """Complete RAG pipeline orchestrator"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.document_processor = AdvancedDocumentProcessor()
        self.text_splitter = AdvancedTextSplitter()
        self.retriever = MultiVectorRetriever(api_key)
        self.conversation = AdvancedConversationChain(api_key)
        
        self.processed_documents = {}
        self.active_chains = {}
    
    async def process_documents_async(self, file_paths: List[str],
                                    processing_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async document processing pipeline"""
        
        if processing_config is None:
            processing_config = {
                'loader_type': 'auto',
                'splitting_strategy': 'recursive',
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'vector_store_type': 'chroma',
                'embedding_model': 'openai'
            }
        
        results = {
            'processed_files': [],
            'total_chunks': 0,
            'processing_time': 0,
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            all_documents = []
            all_metadata = []
            
            # Process each file
            for file_path in file_paths:
                try:
                    documents, metadata = self.document_processor.process_document(
                        file_path, processing_config['loader_type']
                    )
                    
                    # Split documents
                    chunks = self.text_splitter.split_documents(
                        documents,
                        strategy=processing_config['splitting_strategy'],
                        chunk_size=processing_config['chunk_size'],
                        chunk_overlap=processing_config['chunk_overlap']
                    )
                    
                    all_documents.extend(chunks)
                    all_metadata.append(metadata)
                    results['processed_files'].append(metadata.filename)
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # Create vector store
            if all_documents:
                vector_store = self.retriever.create_vector_store(
                    all_documents,
                    store_type=processing_config['vector_store_type'],
                    embedding_model=processing_config['embedding_model'],
                    store_name='main'
                )
                
                # Create conversation chain
                chain = self.conversation.create_chain(
                    vector_store.as_retriever(search_kwargs={"k": 5})
                )
                
                self.active_chains['main'] = chain
                results['total_chunks'] = len(all_documents)
            
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    async def query_async(self, question: str, chain_name: str = 'main') -> Dict[str, Any]:
        """Async query processing"""
        
        if chain_name not in self.active_chains:
            return {
                'answer': "No active conversation chain. Please process documents first.",
                'source_documents': [],
                'error': 'No active chain'
            }
        
        chain = self.active_chains[chain_name]
        
        try:
            start_time = datetime.now()
            result = await self.conversation.aquery(chain, question)
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Enhanced result formatting
            formatted_result = {
                'answer': result.get('answer', ''),
                'source_documents': result.get('source_documents', []),
                'response_time': response_time,
                'model_used': self.conversation.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            return formatted_result
            
        except Exception as e:
            error_msg = f"Query error: {str(e)}"
            logger.error(error_msg)
            return {
                'answer': f"Error processing query: {error_msg}",
                'source_documents': [],
                'error': error_msg
            }

# Streamlit integration functions
@st.cache_resource
def get_rag_pipeline(api_key: str) -> RAGPipeline:
    """Cached RAG pipeline instance"""
    return RAGPipeline(api_key)

@st.cache_data(ttl=3600)
def cache_document_processing(file_hashes: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Cache document processing results"""
    # This would integrate with the actual processing
    return {"cached": True}

async def streamlit_process_documents(uploaded_files, api_key: str, config: Dict[str, Any]):
    """Streamlit-compatible async document processing"""
    
    pipeline = get_rag_pipeline(api_key)
    
    # Save uploaded files temporarily
    temp_files = []
    try:
        for uploaded_file in uploaded_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
            temp_file.write(uploaded_file.getbuffer())
            temp_file.close()
            temp_files.append(temp_file.name)
        
        # Process documents
        results = await pipeline.process_documents_async(temp_files, config)
        
        return results, pipeline
        
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_file}: {e}")

# Export main classes and functions
__all__ = [
    'RAGPipeline',
    'AdvancedDocumentProcessor', 
    'AdvancedTextSplitter',
    'MultiVectorRetriever',
    'AdvancedConversationChain',
    'DocumentMetadata',
    'RetrievalResult',
    'get_rag_pipeline',
    'streamlit_process_documents'
]
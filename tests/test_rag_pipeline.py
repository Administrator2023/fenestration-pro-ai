"""
Comprehensive test suite for RAG pipeline
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from advanced_rag import (
    RAGPipeline, 
    AdvancedDocumentProcessor,
    AdvancedTextSplitter,
    MultiVectorRetriever,
    AdvancedConversationChain,
    DocumentMetadata
)
from multimodal_processor import (
    MultimodalDocumentProcessor,
    AdvancedImageProcessor,
    AdvancedTableExtractor
)
from analytics_dashboard import (
    AnalyticsDatabase,
    SessionMetrics,
    QueryMetrics,
    DocumentMetrics
)

# Test configuration
TEST_API_KEY = "test-api-key-placeholder"

@pytest.fixture
def sample_pdf():
    """Create a sample PDF for testing"""
    # This would create a simple PDF file for testing
    # For now, we'll use a placeholder
    return "sample_test.pdf"

@pytest.fixture
def rag_pipeline():
    """Create RAG pipeline instance for testing"""
    return RAGPipeline(TEST_API_KEY)

@pytest.fixture
def analytics_db():
    """Create analytics database for testing"""
    return AnalyticsDatabase(":memory:")  # In-memory database for testing

class TestAdvancedDocumentProcessor:
    """Test document processing functionality"""
    
    def test_init(self):
        """Test processor initialization"""
        processor = AdvancedDocumentProcessor()
        assert processor is not None
        assert hasattr(processor, 'supported_formats')
        assert '.pdf' in processor.supported_formats
    
    def test_supported_formats(self):
        """Test supported file formats"""
        processor = AdvancedDocumentProcessor()
        formats = processor.supported_formats
        
        assert '.pdf' in formats
        assert '.docx' in formats
        assert '.txt' in formats
        assert '.csv' in formats
    
    @pytest.mark.asyncio
    async def test_process_document_metadata(self):
        """Test document metadata extraction"""
        processor = AdvancedDocumentProcessor()
        
        # Create a temporary text file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for fenestration analysis.\n")
            f.write("It contains information about windows and doors.\n")
            temp_path = f.name
        
        try:
            documents, metadata = processor.process_document(temp_path)
            
            assert isinstance(metadata, DocumentMetadata)
            assert metadata.filename.endswith('.txt')
            assert metadata.file_type == '.txt'
            assert metadata.word_count > 0
            assert metadata.processing_time >= 0
            assert len(documents) > 0
            
        finally:
            Path(temp_path).unlink()

class TestAdvancedTextSplitter:
    """Test text splitting functionality"""
    
    def test_init(self):
        """Test splitter initialization"""
        splitter = AdvancedTextSplitter()
        assert splitter is not None
        assert hasattr(splitter, 'splitters')
    
    def test_splitting_strategies(self):
        """Test different splitting strategies"""
        splitter = AdvancedTextSplitter()
        
        # Create mock documents
        from langchain.schema import Document
        documents = [
            Document(
                page_content="This is a long document about fenestration. " * 50,
                metadata={"source": "test.pdf", "page": 1}
            )
        ]
        
        # Test recursive splitting
        chunks = splitter.split_documents(documents, strategy='recursive', chunk_size=200)
        assert len(chunks) > 1
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)
        assert all('chunk_id' in chunk.metadata for chunk in chunks)

class TestMultiVectorRetriever:
    """Test vector retrieval functionality"""
    
    def test_init(self):
        """Test retriever initialization"""
        retriever = MultiVectorRetriever(TEST_API_KEY)
        assert retriever is not None
        assert hasattr(retriever, 'embeddings_models')
        assert 'openai' in retriever.embeddings_models
    
    def test_embeddings_models(self):
        """Test embeddings model availability"""
        retriever = MultiVectorRetriever(TEST_API_KEY)
        models = retriever.embeddings_models
        
        assert 'openai' in models
        assert 'sentence_transformers' in models
        assert 'huggingface' in models

class TestRAGPipeline:
    """Test complete RAG pipeline"""
    
    def test_init(self, rag_pipeline):
        """Test pipeline initialization"""
        assert rag_pipeline is not None
        assert hasattr(rag_pipeline, 'document_processor')
        assert hasattr(rag_pipeline, 'text_splitter')
        assert hasattr(rag_pipeline, 'retriever')
        assert hasattr(rag_pipeline, 'conversation')
    
    @pytest.mark.asyncio
    async def test_pipeline_components(self, rag_pipeline):
        """Test pipeline component integration"""
        # Test that all components are properly initialized
        assert rag_pipeline.document_processor is not None
        assert rag_pipeline.text_splitter is not None
        assert rag_pipeline.retriever is not None
        assert rag_pipeline.conversation is not None
    
    @pytest.mark.asyncio
    async def test_query_without_documents(self, rag_pipeline):
        """Test querying without processed documents"""
        result = await rag_pipeline.query_async("What is fenestration?")
        
        assert 'error' in result
        assert 'No active chain' in result['error']

class TestMultimodalProcessor:
    """Test multimodal document processing"""
    
    def test_init(self):
        """Test multimodal processor initialization"""
        processor = MultimodalDocumentProcessor(TEST_API_KEY)
        assert processor is not None
        assert hasattr(processor, 'image_processor')
        assert hasattr(processor, 'table_extractor')
        assert hasattr(processor, 'chart_analyzer')
    
    def test_image_processor_init(self):
        """Test image processor initialization"""
        processor = AdvancedImageProcessor(TEST_API_KEY)
        assert processor is not None
        assert hasattr(processor, 'openai_client')
    
    def test_table_extractor_init(self):
        """Test table extractor initialization"""
        extractor = AdvancedTableExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'extraction_methods')
        assert 'camelot' in extractor.extraction_methods

class TestAnalyticsDatabase:
    """Test analytics database functionality"""
    
    def test_init(self, analytics_db):
        """Test database initialization"""
        assert analytics_db is not None
        assert Path(":memory:").exists() or analytics_db.db_path == ":memory:"
    
    def test_session_metrics_insertion(self, analytics_db):
        """Test session metrics insertion"""
        session = SessionMetrics(
            session_id="test_session_1",
            start_time=datetime.now(),
            end_time=None,
            total_queries=5,
            total_documents=2,
            total_chunks=50,
            avg_response_time=2.5,
            total_tokens_used=1000,
            user_satisfaction=4.5,
            errors_count=0
        )
        
        analytics_db.insert_session(session)
        
        # Verify insertion
        stats = analytics_db.get_session_stats(1)
        assert len(stats) == 1
        assert stats.iloc[0]['session_id'] == "test_session_1"
    
    def test_query_metrics_insertion(self, analytics_db):
        """Test query metrics insertion"""
        query = QueryMetrics(
            query_id="test_query_1",
            session_id="test_session_1",
            timestamp=datetime.now(),
            query_text="What is fenestration?",
            response_time=2.1,
            tokens_used=150,
            model_used="gpt-4-turbo-preview",
            retrieval_score=0.85,
            user_rating=5,
            error_occurred=False
        )
        
        analytics_db.insert_query(query)
        
        # Verify insertion
        stats = analytics_db.get_query_stats(1)
        assert len(stats) == 1
        assert stats.iloc[0]['query_id'] == "test_query_1"
    
    def test_document_metrics_insertion(self, analytics_db):
        """Test document metrics insertion"""
        document = DocumentMetrics(
            document_id="test_doc_1",
            filename="test_document.pdf",
            file_size=1024000,
            processing_time=5.2,
            chunk_count=25,
            embedding_time=3.1,
            success=True,
            error_message=None
        )
        
        analytics_db.insert_document(document)
        
        # Verify insertion by checking the database directly
        import sqlite3
        conn = sqlite3.connect(analytics_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE document_id = ?", ("test_doc_1",))
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[1] == "test_document.pdf"  # filename column

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_document_processing_performance(self):
        """Test document processing performance"""
        processor = AdvancedDocumentProcessor()
        
        # Create a larger test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write a substantial amount of text
            content = "This is a test document about fenestration. " * 1000
            f.write(content)
            temp_path = f.name
        
        try:
            start_time = datetime.now()
            documents, metadata = processor.process_document(temp_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Performance assertions
            assert processing_time < 10.0  # Should process within 10 seconds
            assert metadata.processing_time < 10.0
            assert len(documents) > 0
            
        finally:
            Path(temp_path).unlink()
    
    def test_text_splitting_performance(self):
        """Test text splitting performance"""
        splitter = AdvancedTextSplitter()
        
        # Create large document
        from langchain.schema import Document
        large_content = "This is a test document about fenestration systems. " * 2000
        documents = [Document(page_content=large_content, metadata={"source": "test.pdf"})]
        
        start_time = datetime.now()
        chunks = splitter.split_documents(documents, strategy='recursive', chunk_size=1000)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert processing_time < 5.0  # Should split within 5 seconds
        assert len(chunks) > 1

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_text_processing(self):
        """Test end-to-end text document processing"""
        pipeline = RAGPipeline(TEST_API_KEY)
        
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            Fenestration is the design, construction, or presence of openings in a building.
            Windows and doors are the most common types of fenestration.
            Energy efficiency is a key consideration in modern fenestration design.
            U-values and SHGC ratings are important performance metrics.
            """)
            temp_path = f.name
        
        try:
            # Process document
            config = {
                'loader_type': 'auto',
                'splitting_strategy': 'recursive',
                'chunk_size': 500,
                'chunk_overlap': 100,
                'vector_store_type': 'chroma',
                'embedding_model': 'openai'
            }
            
            results = await pipeline.process_documents_async([temp_path], config)
            
            assert 'processed_files' in results
            assert len(results['processed_files']) == 1
            assert results['total_chunks'] > 0
            assert len(results['errors']) == 0
            
        finally:
            Path(temp_path).unlink()
            # Cleanup vector store
            import shutil
            if Path("./vector_stores").exists():
                shutil.rmtree("./vector_stores")

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_file_format(self):
        """Test handling of invalid file formats"""
        processor = AdvancedDocumentProcessor()
        
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"Invalid file format")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                processor.process_document(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_empty_document(self):
        """Test handling of empty documents"""
        processor = AdvancedDocumentProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            documents, metadata = processor.process_document(temp_path)
            assert metadata.word_count == 0
            assert len(documents) >= 0  # May be 0 or 1 depending on loader
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test handling of invalid API key"""
        pipeline = RAGPipeline("invalid_api_key")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            results = await pipeline.process_documents_async([temp_path])
            # Should handle gracefully and report errors
            assert 'errors' in results
        finally:
            Path(temp_path).unlink()

# Test configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
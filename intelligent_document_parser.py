"""
Intelligent Document Parser for Fenestration Shop Drawings and Calculations
Uses Google Document AI for advanced OCR and Pinecone for semantic search

This module provides enterprise-grade document intelligence specifically designed
for fenestration shop drawings, technical specifications, and calculation sheets.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import tempfile

# Core libraries
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# Document AI - Enterprise OCR and Layout Understanding
try:
    from google.cloud import documentai_v1 as documentai
    from google.cloud import storage
    DOCUMENTAI_AVAILABLE = True
except ImportError:
    DOCUMENTAI_AVAILABLE = False
    logging.warning("Google Document AI not available - install google-cloud-documentai")

# Pinecone - Production Vector Database
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not available - install pinecone-client")

# Fallback PDF parsing
try:
    import pdfplumber
    import camelot
    ADVANCED_PDF_AVAILABLE = True
except ImportError:
    ADVANCED_PDF_AVAILABLE = False
    logging.warning("Advanced PDF parsing not available")

# OpenAI for embeddings
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Structured metadata extracted from technical documents"""
    filename: str
    document_type: str  # "shop_drawing", "calculation_sheet", "specification", "submittal"
    project_name: Optional[str] = None
    drawing_number: Optional[str] = None
    revision: Optional[str] = None
    date: Optional[str] = None
    page_count: int = 0
    contains_tables: bool = False
    contains_calculations: bool = False
    contains_drawings: bool = False
    extracted_dimensions: List[str] = None
    extracted_materials: List[str] = None
    extracted_specs: Dict[str, Any] = None
    confidence_score: float = 0.0

    def __post_init__(self):
        if self.extracted_dimensions is None:
            self.extracted_dimensions = []
        if self.extracted_materials is None:
            self.extracted_materials = []
        if self.extracted_specs is None:
            self.extracted_specs = {}


@dataclass
class ExtractedTable:
    """Represents an extracted table with structure"""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    confidence: float
    table_type: str  # "calculation", "specification", "schedule", "bill_of_materials"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert table to pandas DataFrame"""
        return pd.DataFrame(self.rows, columns=self.headers)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ExtractedCalculation:
    """Represents a calculation found in the document"""
    page_number: int
    formula: str
    variables: Dict[str, Any]
    result: Optional[float]
    context: str
    confidence: float


class TechnicalEntityExtractor:
    """Extract technical entities from fenestration documents"""

    # Fenestration-specific patterns
    DIMENSION_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(?:\'|ft|feet)',  # Feet
        r'(\d+(?:\.\d+)?)\s*(?:\"|in|inch)',  # Inches
        r'(\d+(?:\.\d+)?)\s*(?:mm|millimeter)',  # Millimeters
        r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)',  # Width x Height
        r'(\d+(?:\.\d+)?)\s*[wW]\s*x\s*(\d+(?:\.\d+)?)\s*[hH]',  # W x H
    ]

    MATERIAL_PATTERNS = [
        r'\b(aluminum|aluminium)\b',
        r'\b(vinyl|PVC)\b',
        r'\b(wood|timber)\b',
        r'\b(fiberglass)\b',
        r'\b(steel)\b',
        r'\b(glass|glazing)\b',
        r'\b(Low-E|low emissivity)\b',
        r'\b(tempered|laminated|insulated)\b',
        r'\b(argon|krypton)\s+(?:filled|gas)',
    ]

    WINDOW_TYPES = [
        r'\b(casement|double hung|single hung|sliding|awning|hopper|fixed|picture)\b',
        r'\b(bay|bow)\s+window\b',
        r'\b(curtain wall|storefront)\b',
    ]

    SPEC_PATTERNS = [
        r'U-(?:factor|value)[:=\s]+(\d+(?:\.\d+)?)',
        r'SHGC[:=\s]+(\d+(?:\.\d+)?)',
        r'VT[:=\s]+(\d+(?:\.\d+)?)',  # Visible Transmittance
        r'STC[:=\s]+(\d+)',  # Sound Transmission Class
        r'DP[:=\s]+(\d+)',  # Design Pressure
        r'(\d+)\s*psf',  # Pounds per square foot
    ]

    CALCULATION_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',  # Simple multiplication
        r'(?:area|Area)\s*=\s*(\d+(?:\.\d+)?)',
        r'(?:load|Load)\s*=\s*(\d+(?:\.\d+)?)',
        r'(?:weight|Weight)\s*=\s*(\d+(?:\.\d+)?)',
    ]

    @classmethod
    def extract_dimensions(cls, text: str) -> List[str]:
        """Extract dimensions from text"""
        dimensions = []
        for pattern in cls.DIMENSION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dimensions.append(match.group(0))
        return list(set(dimensions))  # Remove duplicates

    @classmethod
    def extract_materials(cls, text: str) -> List[str]:
        """Extract materials from text"""
        materials = []
        for pattern in cls.MATERIAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                materials.append(match.group(0).lower())
        return list(set(materials))

    @classmethod
    def extract_window_types(cls, text: str) -> List[str]:
        """Extract window types from text"""
        types = []
        for pattern in cls.WINDOW_TYPES:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                types.append(match.group(0))
        return list(set(types))

    @classmethod
    def extract_specifications(cls, text: str) -> Dict[str, Any]:
        """Extract performance specifications"""
        specs = {}
        for pattern in cls.SPEC_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                key = match.group(0).split(':')[0].split('=')[0].strip()
                try:
                    value = float(match.group(1))
                    specs[key] = value
                except (IndexError, ValueError):
                    pass
        return specs

    @classmethod
    def extract_calculations(cls, text: str, page_number: int) -> List[ExtractedCalculation]:
        """Extract calculations from text"""
        calculations = []
        for pattern in cls.CALCULATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                calc = ExtractedCalculation(
                    page_number=page_number,
                    formula=match.group(0),
                    variables={},
                    result=None,
                    context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                    confidence=0.8
                )
                try:
                    if len(match.groups()) >= 3:
                        calc.result = float(match.group(3))
                except (IndexError, ValueError):
                    pass
                calculations.append(calc)
        return calculations


class DocumentAIParser:
    """Google Document AI integration for advanced document understanding"""

    def __init__(self, project_id: str, location: str, processor_id: str):
        """
        Initialize Document AI parser

        Args:
            project_id: Google Cloud project ID
            location: Processor location (e.g., 'us' or 'eu')
            processor_id: Document AI processor ID
        """
        if not DOCUMENTAI_AVAILABLE:
            raise ImportError("Google Document AI not available. Install: pip install google-cloud-documentai")

        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.client = documentai.DocumentProcessorServiceClient()

    def process_document(self, file_path: str) -> documentai.Document:
        """
        Process a document using Document AI

        Args:
            file_path: Path to PDF or image file

        Returns:
            Processed document with OCR and layout information
        """
        # Construct processor name
        name = self.client.processor_path(self.project_id, self.location, self.processor_id)

        # Read file
        with open(file_path, "rb") as file:
            file_content = file.read()

        # Determine MIME type
        mime_type = "application/pdf" if file_path.endswith('.pdf') else "image/jpeg"

        # Create request
        request = documentai.ProcessRequest(
            name=name,
            raw_document=documentai.RawDocument(content=file_content, mime_type=mime_type)
        )

        # Process document
        result = self.client.process_document(request=request)
        return result.document

    def extract_text_with_layout(self, document: documentai.Document) -> List[Dict[str, Any]]:
        """
        Extract text with layout information

        Returns:
            List of text blocks with position and confidence
        """
        text_blocks = []
        for page in document.pages:
            for block in page.blocks:
                text = self._get_text_from_layout(document.text, block.layout)
                text_blocks.append({
                    'text': text,
                    'confidence': block.layout.confidence,
                    'page': page.page_number,
                    'type': 'block'
                })
        return text_blocks

    def extract_tables(self, document: documentai.Document) -> List[ExtractedTable]:
        """
        Extract tables from document with structure

        Returns:
            List of ExtractedTable objects
        """
        tables = []
        for page_idx, page in enumerate(document.pages):
            for table_idx, table in enumerate(page.tables):
                headers = []
                rows = []

                # Extract headers
                if table.header_rows:
                    for header_row in table.header_rows:
                        header_cells = []
                        for cell in header_row.cells:
                            text = self._get_text_from_layout(document.text, cell.layout)
                            header_cells.append(text.strip())
                        headers = header_cells

                # Extract rows
                for body_row in table.body_rows:
                    row_cells = []
                    for cell in body_row.cells:
                        text = self._get_text_from_layout(document.text, cell.layout)
                        row_cells.append(text.strip())
                    rows.append(row_cells)

                # Determine table type
                table_type = self._classify_table(headers, rows)

                tables.append(ExtractedTable(
                    page_number=page_idx + 1,
                    table_index=table_idx,
                    headers=headers,
                    rows=rows,
                    confidence=0.9,  # Document AI typically has high confidence
                    table_type=table_type
                ))

        return tables

    def extract_form_fields(self, document: documentai.Document) -> Dict[str, str]:
        """
        Extract form fields (key-value pairs)

        Returns:
            Dictionary of field names and values
        """
        form_fields = {}
        for page in document.pages:
            for field in page.form_fields:
                field_name = self._get_text_from_layout(document.text, field.field_name.layout)
                field_value = self._get_text_from_layout(document.text, field.field_value.layout)
                form_fields[field_name.strip()] = field_value.strip()
        return form_fields

    def _get_text_from_layout(self, document_text: str, layout: documentai.Document.Page.Layout) -> str:
        """Extract text from layout segments"""
        text = ""
        for segment in layout.text_anchor.text_segments:
            start = int(segment.start_index) if segment.start_index else 0
            end = int(segment.end_index) if segment.end_index else 0
            text += document_text[start:end]
        return text

    def _classify_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Classify table type based on headers"""
        header_text = " ".join(headers).lower()

        if any(word in header_text for word in ['qty', 'quantity', 'item', 'description', 'material']):
            return "bill_of_materials"
        elif any(word in header_text for word in ['calc', 'formula', 'result', 'load', 'stress']):
            return "calculation"
        elif any(word in header_text for word in ['spec', 'requirement', 'standard', 'value']):
            return "specification"
        elif any(word in header_text for word in ['date', 'milestone', 'activity', 'duration']):
            return "schedule"
        else:
            return "general"


class PineconeVectorStore:
    """Pinecone vector database integration for semantic search"""

    def __init__(self, api_key: str, environment: str, index_name: str = "fenestration-docs"):
        """
        Initialize Pinecone vector store

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-west1-gcp')
            index_name: Name of the index to use
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install: pip install pinecone-client")

        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name

        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)

        # Create or connect to index
        self._ensure_index_exists()
        self.index = self.pc.Index(index_name)

    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            logger.info(f"Created Pinecone index: {self.index_name}")

    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Upsert documents with embeddings to Pinecone

        Args:
            documents: List of document dictionaries with metadata
            embeddings: List of embedding vectors
        """
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_id = f"{doc.get('source', 'unknown')}_{doc.get('page', 0)}_{i}"
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'text': doc.get('text', ''),
                    'source': doc.get('source', ''),
                    'page': doc.get('page', 0),
                    'document_type': doc.get('document_type', ''),
                    'project': doc.get('project', ''),
                    **doc.get('metadata', {})
                }
            })

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)

        logger.info(f"Upserted {len(vectors)} vectors to Pinecone")

    def query(self, query_embedding: List[float], top_k: int = 10,
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query similar documents

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Metadata filters (e.g., {'project': 'ProjectX'})

        Returns:
            List of matching documents with scores
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_metadata
        )

        return [
            {
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'source': match['metadata'].get('source', ''),
                'page': match['metadata'].get('page', 0),
                'metadata': match['metadata']
            }
            for match in results['matches']
        ]

    def hybrid_search(self, query_text: str, query_embedding: List[float],
                     top_k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search

        Args:
            query_text: Query text for keyword search
            query_embedding: Query embedding for semantic search
            top_k: Number of results
            alpha: Weight for semantic vs keyword (0=keyword only, 1=semantic only)

        Returns:
            Ranked list of results
        """
        # For now, just do semantic search
        # Future: Implement BM25 + semantic fusion
        return self.query(query_embedding, top_k)


class IntelligentDocumentParser:
    """
    Main parser class that orchestrates Document AI and Pinecone
    for intelligent understanding of fenestration shop drawings
    """

    def __init__(self,
                 openai_api_key: str,
                 docai_project_id: Optional[str] = None,
                 docai_location: Optional[str] = None,
                 docai_processor_id: Optional[str] = None,
                 pinecone_api_key: Optional[str] = None,
                 pinecone_environment: Optional[str] = None):
        """
        Initialize the intelligent parser

        Args:
            openai_api_key: OpenAI API key for embeddings
            docai_project_id: Google Cloud project ID
            docai_location: Document AI processor location
            docai_processor_id: Document AI processor ID
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment
        """
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Initialize Document AI if credentials provided
        self.docai_parser = None
        if docai_project_id and docai_location and docai_processor_id:
            try:
                self.docai_parser = DocumentAIParser(
                    docai_project_id, docai_location, docai_processor_id
                )
                logger.info("Document AI initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Document AI: {e}")

        # Initialize Pinecone if credentials provided
        self.pinecone_store = None
        if pinecone_api_key and pinecone_environment:
            try:
                self.pinecone_store = PineconeVectorStore(
                    pinecone_api_key, pinecone_environment
                )
                logger.info("Pinecone initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Pinecone: {e}")

    def parse_document(self, file_path: str, project_name: str = "default") -> Tuple[DocumentMetadata, List[ExtractedTable], str]:
        """
        Parse a technical document with full intelligence

        Args:
            file_path: Path to PDF or image file
            project_name: Project name for organization

        Returns:
            Tuple of (metadata, tables, full_text)
        """
        logger.info(f"Parsing document: {file_path}")

        # Use Document AI if available
        if self.docai_parser:
            doc = self.docai_parser.process_document(file_path)
            full_text = doc.text
            tables = self.docai_parser.extract_tables(doc)
            form_fields = self.docai_parser.extract_form_fields(doc)
            confidence = 0.95
        else:
            # Fallback to pdfplumber
            full_text, tables = self._fallback_parse(file_path)
            form_fields = {}
            confidence = 0.75

        # Extract technical entities
        dimensions = TechnicalEntityExtractor.extract_dimensions(full_text)
        materials = TechnicalEntityExtractor.extract_materials(full_text)
        window_types = TechnicalEntityExtractor.extract_window_types(full_text)
        specs = TechnicalEntityExtractor.extract_specifications(full_text)

        # Classify document type
        doc_type = self._classify_document(full_text, tables, form_fields)

        # Extract metadata
        metadata = DocumentMetadata(
            filename=Path(file_path).name,
            document_type=doc_type,
            project_name=project_name,
            drawing_number=self._extract_drawing_number(full_text, form_fields),
            revision=self._extract_revision(full_text, form_fields),
            date=self._extract_date(full_text, form_fields),
            page_count=full_text.count('\f') + 1,
            contains_tables=len(tables) > 0,
            contains_calculations=any(calc in full_text.lower() for calc in ['calculation', 'load', 'stress', '=']),
            contains_drawings=doc_type == "shop_drawing",
            extracted_dimensions=dimensions,
            extracted_materials=materials,
            extracted_specs=specs,
            confidence_score=confidence
        )

        logger.info(f"Extracted {len(tables)} tables, {len(dimensions)} dimensions, {len(materials)} materials")

        return metadata, tables, full_text

    def index_document(self, file_path: str, project_name: str = "default") -> Dict[str, Any]:
        """
        Parse and index document in Pinecone for semantic search

        Args:
            file_path: Path to document
            project_name: Project name

        Returns:
            Dictionary with parsing results and indexing status
        """
        # Parse document
        metadata, tables, full_text = self.parse_document(file_path, project_name)

        # Split text into chunks
        chunks = self._chunk_text(full_text, chunk_size=500, overlap=100)

        # Create embeddings
        embeddings = self._create_embeddings(chunks)

        # Prepare documents for indexing
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            documents.append({
                'text': chunk,
                'source': metadata.filename,
                'page': i // 4,  # Approximate page
                'document_type': metadata.document_type,
                'project': project_name,
                'metadata': asdict(metadata)
            })

        # Index in Pinecone if available
        if self.pinecone_store:
            self.pinecone_store.upsert_documents(documents, embeddings)
            logger.info(f"Indexed {len(documents)} chunks in Pinecone")

        return {
            'metadata': asdict(metadata),
            'tables': [t.to_dict() for t in tables],
            'chunks_indexed': len(documents),
            'full_text': full_text[:1000] + "..." if len(full_text) > 1000 else full_text
        }

    def query_documents(self, query: str, project_name: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query indexed documents with semantic search

        Args:
            query: Natural language query
            project_name: Filter by project
            top_k: Number of results

        Returns:
            List of relevant document chunks
        """
        if not self.pinecone_store:
            raise ValueError("Pinecone not initialized. Provide API credentials.")

        # Create query embedding
        query_embedding = self._create_embeddings([query])[0]

        # Build filter
        filter_metadata = {'project': project_name} if project_name else None

        # Query Pinecone
        results = self.pinecone_store.query(query_embedding, top_k, filter_metadata)

        return results

    def _fallback_parse(self, file_path: str) -> Tuple[str, List[ExtractedTable]]:
        """Fallback parsing using pdfplumber"""
        if not ADVANCED_PDF_AVAILABLE:
            raise ImportError("Advanced PDF parsing not available. Install pdfplumber and camelot-py")

        full_text = ""
        tables = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                full_text += page.extract_text() or ""
                full_text += "\f"  # Page break

                # Extract tables
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        headers = table[0]
                        rows = table[1:]
                        tables.append(ExtractedTable(
                            page_number=page_num + 1,
                            table_index=table_idx,
                            headers=headers,
                            rows=rows,
                            confidence=0.7,
                            table_type="general"
                        ))

        return full_text, tables

    def _classify_document(self, text: str, tables: List[ExtractedTable], form_fields: Dict[str, str]) -> str:
        """Classify document type"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['shop drawing', 'detail drawing', 'fabrication']):
            return "shop_drawing"
        elif any(word in text_lower for word in ['calculation', 'structural calc', 'load calc']):
            return "calculation_sheet"
        elif any(word in text_lower for word in ['specification', 'spec section', 'requirements']):
            return "specification"
        elif any(word in text_lower for word in ['submittal', 'product data', 'approval']):
            return "submittal"
        else:
            return "technical_document"

    def _extract_drawing_number(self, text: str, form_fields: Dict[str, str]) -> Optional[str]:
        """Extract drawing number"""
        # Check form fields first
        for key in form_fields:
            if 'drawing' in key.lower() or 'number' in key.lower() or 'dwg' in key.lower():
                return form_fields[key]

        # Pattern matching
        patterns = [
            r'(?:drawing|dwg|detail)\s*(?:no|number|#)?\s*[:.]?\s*([A-Z0-9\-\.]+)',
            r'([A-Z]{1,3}\d{3,6})',  # Common drawing number format
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_revision(self, text: str, form_fields: Dict[str, str]) -> Optional[str]:
        """Extract revision number"""
        for key in form_fields:
            if 'rev' in key.lower():
                return form_fields[key]

        patterns = [
            r'(?:revision|rev)[:.]?\s*([A-Z0-9]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_date(self, text: str, form_fields: Dict[str, str]) -> Optional[str]:
        """Extract document date"""
        for key in form_fields:
            if 'date' in key.lower():
                return form_fields[key]

        patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        return None

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.7:  # At least 70% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI"""
        embeddings = []

        # Process in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.openai_client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            embeddings.extend([item.embedding for item in response.data])

        return embeddings


# Convenience functions
def create_parser(openai_api_key: str,
                 docai_credentials: Optional[Dict[str, str]] = None,
                 pinecone_credentials: Optional[Dict[str, str]] = None) -> IntelligentDocumentParser:
    """
    Create an intelligent document parser with optional Document AI and Pinecone

    Args:
        openai_api_key: OpenAI API key
        docai_credentials: Dict with 'project_id', 'location', 'processor_id'
        pinecone_credentials: Dict with 'api_key', 'environment'

    Returns:
        IntelligentDocumentParser instance
    """
    docai_args = docai_credentials or {}
    pinecone_args = pinecone_credentials or {}

    return IntelligentDocumentParser(
        openai_api_key=openai_api_key,
        docai_project_id=docai_args.get('project_id'),
        docai_location=docai_args.get('location'),
        docai_processor_id=docai_args.get('processor_id'),
        pinecone_api_key=pinecone_args.get('api_key'),
        pinecone_environment=pinecone_args.get('environment')
    )

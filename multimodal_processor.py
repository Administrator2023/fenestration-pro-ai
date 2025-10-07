"""
Multimodal Document Processing
Support for images, tables, charts, and complex document structures
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import tempfile
import base64
from io import BytesIO

# Core imports
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import pytesseract

# Document processing
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from unstructured.staging.base import dict_to_elements

# Table extraction
import camelot
import tabula
from pdfplumber import PDF

# OpenAI Vision
import openai
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageContent:
    """Image content with metadata"""
    image_data: bytes
    image_type: str
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    ocr_text: str
    description: str
    confidence: float

@dataclass
class TableContent:
    """Table content with metadata"""
    dataframe: pd.DataFrame
    page_number: int
    bbox: Tuple[float, float, float, float]
    table_type: str  # 'structured', 'semi-structured', 'unstructured'
    confidence: float
    headers: List[str]
    caption: Optional[str] = None

@dataclass
class ChartContent:
    """Chart/diagram content with metadata"""
    image_data: bytes
    chart_type: str  # 'bar', 'line', 'pie', 'technical_drawing', etc.
    page_number: int
    bbox: Tuple[float, float, float, float]
    description: str
    data_extracted: Optional[Dict[str, Any]] = None

class AdvancedImageProcessor:
    """Advanced image processing with OCR and AI vision"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Configure Tesseract (adjust path as needed)
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    
    def extract_images_from_pdf(self, pdf_path: str, dpi: int = 300) -> List[ImageContent]:
        """Extract all images from PDF with metadata"""
        images = []
        
        try:
            # Method 1: Using PyMuPDF for embedded images
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            # Get image bbox
                            img_rect = page.get_image_rects(img)[0] if page.get_image_rects(img) else fitz.Rect(0, 0, 100, 100)
                            bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
                            
                            # Perform OCR
                            ocr_text = self._perform_ocr(img_data)
                            
                            # Get AI description
                            description = await self._get_ai_description(img_data)
                            
                            image_content = ImageContent(
                                image_data=img_data,
                                image_type="png",
                                page_number=page_num + 1,
                                bbox=bbox,
                                ocr_text=ocr_text,
                                description=description,
                                confidence=0.9
                            )
                            
                            images.append(image_content)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error extracting image {img_index} from page {page_num}: {e}")
            
            pdf_document.close()
            
            # Method 2: Convert pages to images for full page analysis
            if not images:  # Fallback if no embedded images found
                page_images = convert_from_path(pdf_path, dpi=dpi)
                
                for page_num, page_image in enumerate(page_images):
                    # Convert PIL image to bytes
                    img_buffer = BytesIO()
                    page_image.save(img_buffer, format='PNG')
                    img_data = img_buffer.getvalue()
                    
                    # Perform OCR on full page
                    ocr_text = self._perform_ocr(img_data)
                    
                    # Get AI description
                    description = await self._get_ai_description(img_data)
                    
                    image_content = ImageContent(
                        image_data=img_data,
                        image_type="png",
                        page_number=page_num + 1,
                        bbox=(0, 0, page_image.width, page_image.height),
                        ocr_text=ocr_text,
                        description=description,
                        confidence=0.8
                    )
                    
                    images.append(image_content)
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
        
        return images
    
    def _perform_ocr(self, image_data: bytes) -> str:
        """Perform OCR on image data"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data))
            
            # Enhance image for better OCR
            image = image.convert('RGB')
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(image, config='--psm 6')
            
            return ocr_text.strip()
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""
    
    async def _get_ai_description(self, image_data: bytes) -> str:
        """Get AI-powered image description using OpenAI Vision"""
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image from a fenestration/construction document. 
                                Describe what you see, focusing on:
                                - Technical drawings, diagrams, or schematics
                                - Window/door components and specifications
                                - Charts, graphs, or data visualizations
                                - Installation details or cross-sections
                                - Performance data or test results
                                Provide a detailed technical description."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI image description failed: {e}")
            return "Image description unavailable"

class AdvancedTableExtractor:
    """Advanced table extraction from PDFs"""
    
    def __init__(self):
        self.extraction_methods = ['camelot', 'tabula', 'pdfplumber', 'unstructured']
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[TableContent]:
        """Extract tables using multiple methods"""
        all_tables = []
        
        # Method 1: Camelot (best for well-structured tables)
        try:
            camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(camelot_tables):
                if table.accuracy > 50:  # Only high-confidence tables
                    table_content = TableContent(
                        dataframe=table.df,
                        page_number=table.page,
                        bbox=(0, 0, 0, 0),  # Camelot doesn't provide bbox easily
                        table_type='structured',
                        confidence=table.accuracy / 100.0,
                        headers=table.df.iloc[0].tolist() if not table.df.empty else []
                    )
                    all_tables.append(table_content)
                    
        except Exception as e:
            logger.error(f"Camelot table extraction failed: {e}")
        
        # Method 2: Tabula (good for stream tables)
        try:
            tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            for i, df in enumerate(tabula_tables):
                if not df.empty and len(df.columns) > 1:
                    table_content = TableContent(
                        dataframe=df,
                        page_number=1,  # Tabula doesn't provide page info easily
                        bbox=(0, 0, 0, 0),
                        table_type='semi-structured',
                        confidence=0.7,
                        headers=df.columns.tolist()
                    )
                    all_tables.append(table_content)
                    
        except Exception as e:
            logger.error(f"Tabula table extraction failed: {e}")
        
        # Method 3: PDFPlumber (good for complex layouts)
        try:
            with PDF.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_data in tables:
                        if table_data and len(table_data) > 1:
                            # Convert to DataFrame
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            
                            table_content = TableContent(
                                dataframe=df,
                                page_number=page_num + 1,
                                bbox=(0, 0, page.width, page.height),
                                table_type='unstructured',
                                confidence=0.6,
                                headers=table_data[0] if table_data else []
                            )
                            all_tables.append(table_content)
                            
        except Exception as e:
            logger.error(f"PDFPlumber table extraction failed: {e}")
        
        return all_tables
    
    def clean_and_validate_table(self, table_content: TableContent) -> TableContent:
        """Clean and validate extracted table data"""
        df = table_content.dataframe.copy()
        
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Detect and convert numeric columns
        for col in df.columns:
            try:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        # Update table content
        table_content.dataframe = df
        table_content.headers = df.columns.tolist()
        
        return table_content

class ChartAnalyzer:
    """Analyze charts and technical drawings"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
    
    def analyze_charts_in_images(self, images: List[ImageContent]) -> List[ChartContent]:
        """Analyze images to identify and extract chart data"""
        charts = []
        
        for image in images:
            try:
                # Use AI to determine if image contains a chart
                chart_analysis = await self._analyze_chart_with_ai(image.image_data)
                
                if chart_analysis['is_chart']:
                    chart_content = ChartContent(
                        image_data=image.image_data,
                        chart_type=chart_analysis['chart_type'],
                        page_number=image.page_number,
                        bbox=image.bbox,
                        description=chart_analysis['description'],
                        data_extracted=chart_analysis.get('data')
                    )
                    charts.append(chart_content)
                    
            except Exception as e:
                logger.error(f"Chart analysis failed for image on page {image.page_number}: {e}")
        
        return charts
    
    async def _analyze_chart_with_ai(self, image_data: bytes) -> Dict[str, Any]:
        """Use AI to analyze chart content"""
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image to determine if it contains charts, graphs, or technical drawings.
                                
                                If it's a chart/graph, identify:
                                1. Chart type (bar, line, pie, scatter, technical drawing, etc.)
                                2. What data it represents
                                3. Key values or trends
                                4. Axis labels and units
                                
                                If it's a technical drawing, identify:
                                1. Type of drawing (cross-section, elevation, detail, etc.)
                                2. Components shown
                                3. Dimensions or specifications
                                4. Technical details
                                
                                Respond in JSON format:
                                {
                                    "is_chart": true/false,
                                    "chart_type": "type",
                                    "description": "detailed description",
                                    "data": {"key": "value", ...}
                                }"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"AI chart analysis failed: {e}")
            return {
                "is_chart": False,
                "chart_type": "unknown",
                "description": "Analysis failed",
                "data": {}
            }

class MultimodalDocumentProcessor:
    """Complete multimodal document processing pipeline"""
    
    def __init__(self, openai_api_key: str):
        self.image_processor = AdvancedImageProcessor(openai_api_key)
        self.table_extractor = AdvancedTableExtractor()
        self.chart_analyzer = ChartAnalyzer(openai_api_key)
    
    async def process_document_multimodal(self, pdf_path: str) -> Dict[str, Any]:
        """Process document with full multimodal extraction"""
        
        results = {
            'text_content': [],
            'images': [],
            'tables': [],
            'charts': [],
            'processing_time': 0,
            'errors': []
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract text using unstructured
            elements = partition_pdf(pdf_path, strategy="hi_res", infer_table_structure=True)
            
            for element in elements:
                if hasattr(element, 'text') and element.text.strip():
                    results['text_content'].append({
                        'content': element.text,
                        'type': str(type(element).__name__),
                        'metadata': element.metadata.to_dict() if hasattr(element, 'metadata') else {}
                    })
            
            # Extract images
            images = self.image_processor.extract_images_from_pdf(pdf_path)
            results['images'] = images
            
            # Extract tables
            tables = self.table_extractor.extract_tables_from_pdf(pdf_path)
            # Clean and validate tables
            cleaned_tables = []
            for table in tables:
                cleaned_table = self.table_extractor.clean_and_validate_table(table)
                cleaned_tables.append(cleaned_table)
            results['tables'] = cleaned_tables
            
            # Analyze charts
            charts = await self.chart_analyzer.analyze_charts_in_images(images)
            results['charts'] = charts
            
        except Exception as e:
            error_msg = f"Multimodal processing error: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        results['processing_time'] = asyncio.get_event_loop().time() - start_time
        
        return results
    
    def create_enhanced_documents(self, multimodal_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create enhanced document objects with multimodal content"""
        
        enhanced_docs = []
        
        # Text content
        for text_item in multimodal_results['text_content']:
            doc = {
                'content': text_item['content'],
                'type': 'text',
                'metadata': text_item['metadata']
            }
            enhanced_docs.append(doc)
        
        # Image content
        for image in multimodal_results['images']:
            doc = {
                'content': f"[IMAGE] {image.description}\nOCR Text: {image.ocr_text}",
                'type': 'image',
                'metadata': {
                    'page_number': image.page_number,
                    'bbox': image.bbox,
                    'confidence': image.confidence,
                    'image_data': base64.b64encode(image.image_data).decode('utf-8')
                }
            }
            enhanced_docs.append(doc)
        
        # Table content
        for table in multimodal_results['tables']:
            # Convert table to text representation
            table_text = f"[TABLE] Page {table.page_number}\n"
            table_text += f"Headers: {', '.join(table.headers)}\n"
            table_text += table.dataframe.to_string(index=False)
            
            doc = {
                'content': table_text,
                'type': 'table',
                'metadata': {
                    'page_number': table.page_number,
                    'table_type': table.table_type,
                    'confidence': table.confidence,
                    'headers': table.headers,
                    'data': table.dataframe.to_dict('records')
                }
            }
            enhanced_docs.append(doc)
        
        # Chart content
        for chart in multimodal_results['charts']:
            doc = {
                'content': f"[CHART] {chart.chart_type.upper()}\n{chart.description}",
                'type': 'chart',
                'metadata': {
                    'page_number': chart.page_number,
                    'chart_type': chart.chart_type,
                    'bbox': chart.bbox,
                    'data_extracted': chart.data_extracted,
                    'image_data': base64.b64encode(chart.image_data).decode('utf-8')
                }
            }
            enhanced_docs.append(doc)
        
        return enhanced_docs

# Streamlit integration
@st.cache_resource
def get_multimodal_processor(openai_api_key: str) -> MultimodalDocumentProcessor:
    """Cached multimodal processor instance"""
    return MultimodalDocumentProcessor(openai_api_key)

async def process_document_with_multimodal(pdf_path: str, openai_api_key: str) -> Dict[str, Any]:
    """Process document with full multimodal capabilities"""
    processor = get_multimodal_processor(openai_api_key)
    results = await processor.process_document_multimodal(pdf_path)
    enhanced_docs = processor.create_enhanced_documents(results)
    
    return {
        'multimodal_results': results,
        'enhanced_documents': enhanced_docs
    }

# Export main classes
__all__ = [
    'MultimodalDocumentProcessor',
    'AdvancedImageProcessor',
    'AdvancedTableExtractor', 
    'ChartAnalyzer',
    'ImageContent',
    'TableContent',
    'ChartContent',
    'get_multimodal_processor',
    'process_document_with_multimodal'
]
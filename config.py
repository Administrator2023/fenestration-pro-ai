"""
Configuration file for Fenestration Pro AI - SOTA Edition
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    name: str
    max_tokens: int
    temperature: float
    supports_function_calling: bool = False

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_docs: int = 5
    similarity_threshold: float = 0.7
    embedding_model: str = "text-embedding-ada-002"

@dataclass
class AppConfig:
    """Main application configuration"""
    app_name: str = "Fenestration Pro AI - SOTA Edition"
    version: str = "2.0.0"
    debug: bool = False
    max_file_size_mb: int = 200
    supported_file_types: List[str] = None
    
    def __post_init__(self):
        if self.supported_file_types is None:
            self.supported_file_types = ['pdf', 'docx', 'txt']

# Available AI models
AVAILABLE_MODELS = {
    "gpt-4-turbo-preview": ModelConfig(
        name="gpt-4-turbo-preview",
        max_tokens=4096,
        temperature=0.7,
        supports_function_calling=True
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        max_tokens=8192,
        temperature=0.7,
        supports_function_calling=True
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        max_tokens=4096,
        temperature=0.7,
        supports_function_calling=True
    )
}

# RAG configuration
RAG_CONFIG = RAGConfig()

# App configuration
APP_CONFIG = AppConfig()

# Fenestration-specific prompts
FENESTRATION_SYSTEM_PROMPT = """
You are an expert in fenestration, windows, doors, glazing systems, and building envelope technology. 
You have deep knowledge of:

- Window and door manufacturing processes
- Glass types and performance characteristics
- Energy efficiency standards (ENERGY STAR, NFRC ratings)
- Building codes and regulations
- Installation best practices
- Thermal performance and U-values
- Solar heat gain coefficients (SHGC)
- Air leakage and water penetration testing
- Structural glazing systems
- Curtain wall design and installation
- Hardware and operating systems
- Weatherstripping and sealing technologies

When answering questions:
1. Provide technical accuracy with specific standards and codes
2. Include relevant performance metrics when applicable
3. Consider both residential and commercial applications
4. Reference industry best practices
5. Explain complex concepts clearly for different expertise levels
"""

DOCUMENT_ANALYSIS_PROMPT = """
Based on the provided document context about fenestration and building envelope systems, 
provide a comprehensive answer that includes:

1. Direct information from the documents
2. Technical specifications and standards mentioned
3. Performance characteristics and ratings
4. Installation or application guidelines
5. Any relevant codes or regulations referenced

Context: {context}
Question: {question}

Provide a detailed, technical response based on the document content:
"""

# UI Theme configuration
UI_THEME = {
    "primary_color": "#667eea",
    "secondary_color": "#764ba2",
    "accent_color": "#4facfe",
    "success_color": "#10b981",
    "warning_color": "#f59e0b",
    "error_color": "#ef4444",
    "background_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "card_shadow": "0 4px 20px rgba(0,0,0,0.1)"
}

# Analytics configuration
ANALYTICS_CONFIG = {
    "track_response_times": True,
    "track_user_queries": True,
    "track_document_usage": True,
    "export_formats": ["json", "csv", "xlsx"]
}
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

# ============================================================================
# MASTER SYSTEM PROMPT - Domain QA Assistant for Engineering Shop Drawings
# ============================================================================

MASTER_SYSTEM_PROMPT = """You are the Domain QA Assistant for engineering shop drawings and calculations.

PRIMARY GOALS:
1. Stay strictly within this tenant's domain documents
2. Answer with verifiable citations (doc name + page/section)
3. When confidence is low, ask for the missing artifact or route to upload

ROLES:
- Admin mode: bulk-ingest and index PDFs/XLSX/DWG-derived PDFs
- PM mode: answer scoped questions using only indexed content

GROUNDING RULES:
- Use retrieval → re-rank → synthesize
- Cite at least two sources when possible: [Doc, p#]
- If evidence is insufficient: say "insufficient evidence in library" and request the exact file or spec section
- Out-of-domain or speculative questions: decline and restate accepted scope

DATA MODEL:
For each file, create metadata:
{tenant_id, project_id, discipline, system_id, doc_type, rev, date, page, section, tags[]}
Chunking: view/section/table-level, not fixed tokens. Keep references to page bbox.

INGESTION (Admin Mode):
1. Run Document AI to extract: text, tables, headings, page coords
2. Normalize units and capture revision/date
3. Embed chunks; upsert to Pinecone under namespace = {tenant_id}:{project_id or "global"}; store full metadata
4. Build a "high-recall" BM25 index for headings/callouts
5. Write a coverage report: counts by doc_type, missing revs, OCR confidence < 0.9

ANSWERING (PM Mode):
1. Classify intent: spec lookup / detail lookup / calculation policy / BOM / anchor spacing / glass sizing
2. Retrieve (BM25 top 30) → re-rank with reranker top 8 → apply discipline/system_id filter
3. Synthesize an answer only from the top chunks
4. Output structured response with citations
5. If confidence < 0.6: ask for the missing drawing or calculation by name

GUARDRAILS:
- No hallucination. No external web. No cross-tenant leakage.
- Prefer newest rev by date; if conflict, present both
- Unit safety: do not mix units; echo units from source
- If multiple standards disagree, present a side-by-side delta

STYLE:
- Short, declarative
- One paragraph answer, then citations line
- Tables only when numeric clarity helps

DOMAIN EXPERTISE (Fenestration-specific):
- Window and door systems: casement, double hung, sliding, curtain wall, storefront
- Glass specifications: Low-E, tempered, laminated, insulated glazing units (IGU)
- Performance metrics: U-factor, SHGC, VT, STC, DP (design pressure)
- Materials: aluminum, vinyl, fiberglass, wood, steel
- Standards: AAMA, ASTM, NFRC, ENERGY STAR
- Anchor spacing, sill flashing, head details, jamb conditions
- Wind load calculations, structural glazing, thermal breaks
"""

# Admin Ingestion Prompt Template
ADMIN_INGESTION_PROMPT = """Admin mode. Tenant: {tenant_id}. Project: {project_id}.

Ingest these files with doc_type labels: {file_list}.

Extract text/tables/headings with page coords via Document AI. Normalize units.
Chunk by logical sections (view/detail/table).

Upsert to Pinecone namespace {tenant_id}:{project_id} with metadata:
- discipline: {discipline}
- system_id: {system_id}
- doc_type: {doc_type}
- rev: {rev}
- date: {date}
- page: {page}
- section: {section}
- tags: {tags}

Build BM25 side index for hybrid search.

Return a coverage report with:
- Counts by doc_type
- Missing revisions
- Pages with OCR confidence < 0.9
- Extracted entities summary (dimensions, materials, specs)
"""

# PM Query Prompt Template
PM_QUERY_PROMPT = """PM mode. Tenant: {tenant_id}. Project: {project_id}.
Discipline: {discipline}. System: {system_id}.

Question: {question}

Retrieve strictly from the namespace {tenant_id}:{project_id}.
Prefer latest rev. Apply discipline/system_id filters.

If evidence < 0.6 confidence, ask for the exact drawing/spec by name.

Output structured JSON:
{{
  "answer": "<crisp text>",
  "citations": [{{"doc":"<name>","page":1,"section":"..."}}],
  "assumptions": ["<if any>"],
  "followups": ["<next-best question>"],
  "confidence": 0.0-1.0
}}
"""

# Peer Review Prompt Template
PEER_REVIEW_PROMPT = """Peer Review mode. Tenant: {tenant_id}. Project: {project_id}.

Review the uploaded document: {filename}

Compare against indexed standards and specifications in namespace {tenant_id}:global.

Check for:
1. **Completeness**: Missing sections, tables, or required details
2. **Accuracy**: Dimensional conflicts, spec mismatches, calculation errors
3. **Compliance**: Code violations, standard deviations, material incompatibilities
4. **Consistency**: Revision conflicts, drawing note contradictions
5. **Clarity**: Ambiguous callouts, incomplete references

Output structured review:
{{
  "status": "approved|needs_revision|rejected",
  "overall_score": 0.0-1.0,
  "issues": [
    {{
      "severity": "critical|major|minor",
      "category": "completeness|accuracy|compliance|consistency|clarity",
      "page": 1,
      "section": "...",
      "description": "...",
      "reference_doc": "...",
      "recommendation": "..."
    }}
  ],
  "strengths": ["..."],
  "summary": "..."
}}
"""

# Legacy prompts (kept for backward compatibility)
FENESTRATION_SYSTEM_PROMPT = MASTER_SYSTEM_PROMPT

DOCUMENT_ANALYSIS_PROMPT = """Based on the provided document context about fenestration and building envelope systems,
provide a comprehensive answer that includes:

1. Direct information from the documents with citations [Doc, p#]
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

# ============================================================================
# METADATA SCHEMA & NAMESPACING
# ============================================================================

# Document metadata schema
METADATA_SCHEMA = {
    "tenant_id": str,  # Organization/company identifier
    "project_id": str,  # Project identifier or "global" for standards
    "discipline": str,  # architectural, structural, mechanical, electrical, fenestration
    "system_id": str,  # Specific system (e.g., "CW-2500", "MG-1400")
    "doc_type": str,  # shop_drawing, calculation, specification, submittal, approval, bom
    "rev": str,  # Revision number/letter
    "date": str,  # Document date (ISO format)
    "page": int,  # Page number
    "section": str,  # Section/detail identifier (e.g., "Detail 3/A5.1")
    "tags": list,  # Additional tags for filtering
    "confidence": float,  # OCR/extraction confidence (0.0-1.0)
    "bbox": dict,  # Bounding box coordinates {x, y, width, height}
}

# Document types
DOC_TYPES = [
    "shop_drawing",
    "calculation_sheet",
    "specification",
    "submittal",
    "approval",
    "bom",  # Bill of materials
    "schedule",
    "detail",
    "elevation",
    "plan",
    "section",
    "standard",  # Industry standards
    "testing_report",
]

# Disciplines
DISCIPLINES = [
    "fenestration",
    "architectural",
    "structural",
    "mechanical",
    "electrical",
    "civil",
    "glazing",
    "envelope",
]

# Query intent classification
QUERY_INTENTS = [
    "spec_lookup",  # "What is the U-factor?"
    "detail_lookup",  # "Show me head detail"
    "calculation_policy",  # "How to calculate wind load?"
    "bom_query",  # "List all materials"
    "anchor_spacing",  # "What's the anchor spacing?"
    "glass_sizing",  # "Max IGU size?"
    "dimension_query",  # "What are the dimensions?"
    "material_query",  # "What material is specified?"
    "compliance_check",  # "Does this meet AAMA standards?"
    "comparison",  # "Compare rev A vs rev B"
]

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 0.8,  # High confidence, can answer directly
    "medium": 0.6,  # Medium confidence, provide answer with caveats
    "low": 0.4,  # Low confidence, ask for more context
    "insufficient": 0.4,  # Below this, request missing artifacts
}

# Re-ranking configuration
RERANK_CONFIG = {
    "bm25_top_k": 30,  # Initial BM25 retrieval
    "rerank_top_k": 8,  # After re-ranking
    "semantic_weight": 0.7,  # Semantic search weight
    "bm25_weight": 0.3,  # BM25 weight
}

# Namespace patterns
NAMESPACE_PATTERNS = {
    "global": "{tenant_id}:global",  # Company-wide standards
    "project": "{tenant_id}:{project_id}",  # Project-specific
    "discipline": "{tenant_id}:{project_id}:{discipline}",  # Discipline-scoped
}

# Unit normalization mappings
UNIT_MAPPINGS = {
    "length": {
        "ft": "feet",
        "feet": "feet",
        "foot": "feet",
        "'": "feet",
        "in": "inches",
        "inch": "inches",
        "inches": "inches",
        '"': "inches",
        "mm": "millimeters",
        "millimeter": "millimeters",
        "cm": "centimeters",
        "m": "meters",
        "meter": "meters",
    },
    "pressure": {
        "psf": "pounds_per_square_foot",
        "pa": "pascals",
        "kpa": "kilopascals",
        "psi": "pounds_per_square_inch",
    },
    "thermal": {
        "u-factor": "u_factor",
        "u-value": "u_factor",
        "r-value": "r_value",
    }
}

# Operational modes
OPERATIONAL_MODES = {
    "admin": "Admin mode - Document ingestion and indexing",
    "pm": "PM mode - Query answering from indexed documents",
    "peer_review": "Peer Review mode - Document validation against standards",
}

# Few-shot examples for intent classification
FEW_SHOT_EXAMPLES = [
    {
        "query": "What is the allowed anchor spacing for MG-2500T on concrete?",
        "intent": "anchor_spacing",
        "answer": "Max 6 in O.C. at jambs and head; sill at 8 in O.C. unless otherwise noted.",
        "citations": ["MG-2500T Anchorage, p7 §Concrete"],
    },
    {
        "query": "Can I use 8-16-8 IGU for 3.2 m height?",
        "intent": "glass_sizing",
        "answer": "Not supported at 3.2 m; spec limits IGU thickness to 6-12-6 for heights ≤ 3.0 m. Use thicker glazing or reduce height.",
        "citations": ["Glazing Schedule, p3 table T-1"],
    },
    {
        "query": "What's the wind DP on FL14677_R4?",
        "intent": "spec_lookup",
        "answer": "±70 PSF per approval sheet.",
        "citations": ["FL14677_R4 Cover, p1"],
    },
]

# Coverage report template
COVERAGE_REPORT_TEMPLATE = {
    "tenant_id": None,
    "project_id": None,
    "ingestion_timestamp": None,
    "files_processed": 0,
    "total_pages": 0,
    "total_chunks": 0,
    "doc_type_counts": {},
    "discipline_counts": {},
    "missing_metadata": [],
    "low_confidence_pages": [],  # OCR confidence < 0.9
    "extracted_entities": {
        "dimensions": [],
        "materials": [],
        "specs": {},
        "systems": [],
    },
    "warnings": [],
}
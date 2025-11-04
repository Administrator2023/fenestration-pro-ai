"""
Domain QA Engine - Master System Prompt Implementation
Implements Admin, PM, and Peer Review modes with full grounding and guardrails
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import re

# Import configurations
from config import (
    MASTER_SYSTEM_PROMPT,
    ADMIN_INGESTION_PROMPT,
    PM_QUERY_PROMPT,
    PEER_REVIEW_PROMPT,
    CONFIDENCE_THRESHOLDS,
    RERANK_CONFIG,
    METADATA_SCHEMA,
    DOC_TYPES,
    DISCIPLINES,
    QUERY_INTENTS,
    FEW_SHOT_EXAMPLES,
    COVERAGE_REPORT_TEMPLATE,
    UNIT_MAPPINGS,
)

# Import the intelligent document parser
from intelligent_document_parser import (
    IntelligentDocumentParser,
    TechnicalEntityExtractor,
    DocumentMetadata,
    ExtractedTable,
    PINECONE_AVAILABLE,
    DOCUMENTAI_AVAILABLE,
)

# Import continuous learning engine
from continuous_learning_engine import (
    ContinuousLearningEngine,
    LearningStats,
)

# OpenAI for LLM
from openai import OpenAI

# BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank_bm25 not available - install: pip install rank-bm25")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedMetadata:
    """Enhanced metadata following the master system prompt schema"""
    tenant_id: str
    project_id: str
    discipline: str
    system_id: str
    doc_type: str
    rev: str
    date: str
    page: int
    section: str
    tags: List[str]
    confidence: float
    bbox: Optional[Dict[str, float]] = None

    # Additional fields
    filename: str = ""
    extracted_dimensions: List[str] = None
    extracted_materials: List[str] = None
    extracted_specs: Dict[str, Any] = None

    def __post_init__(self):
        if self.extracted_dimensions is None:
            self.extracted_dimensions = []
        if self.extracted_materials is None:
            self.extracted_materials = []
        if self.extracted_specs is None:
            self.extracted_specs = {}

    def to_pinecone_metadata(self) -> Dict[str, Any]:
        """Convert to Pinecone-compatible metadata (flat structure)"""
        return {
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "discipline": self.discipline,
            "system_id": self.system_id,
            "doc_type": self.doc_type,
            "rev": self.rev,
            "date": self.date,
            "page": self.page,
            "section": self.section,
            "tags": json.dumps(self.tags),  # Serialize list
            "confidence": self.confidence,
            "filename": self.filename,
        }


@dataclass
class PMResponse:
    """Structured response for PM queries"""
    answer: str
    citations: List[Dict[str, Any]]
    assumptions: List[str]
    followups: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PeerReviewIssue:
    """Issue found during peer review"""
    severity: str  # critical, major, minor
    category: str  # completeness, accuracy, compliance, consistency, clarity
    page: int
    section: str
    description: str
    reference_doc: str
    recommendation: str


@dataclass
class PeerReviewResult:
    """Result of peer review"""
    status: str  # approved, needs_revision, rejected
    overall_score: float
    issues: List[PeerReviewIssue]
    strengths: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "overall_score": self.overall_score,
            "issues": [asdict(issue) for issue in self.issues],
            "strengths": self.strengths,
            "summary": self.summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class IntentClassifier:
    """Classify user query intent"""

    INTENT_PATTERNS = {
        "spec_lookup": [
            r"\b(u-factor|shgc|vt|stc|dp|rating|performance)\b",
            r"\bwhat (is|are) the (spec|specification)\b",
        ],
        "detail_lookup": [
            r"\b(detail|section|view|elevation)\b",
            r"\bshow me\b",
        ],
        "calculation_policy": [
            r"\b(calculate|calculation|formula|compute)\b",
            r"\bhow to\b",
        ],
        "bom_query": [
            r"\b(bill of materials|bom|parts list|materials list)\b",
            r"\blist all\b",
        ],
        "anchor_spacing": [
            r"\b(anchor|fastener|spacing|o\.?c\.?)\b",
        ],
        "glass_sizing": [
            r"\b(glass|glazing|igu|lite) (size|dimension|thickness)\b",
            r"\bmax(imum)? (size|dimension)\b",
        ],
        "dimension_query": [
            r"\b(dimension|size|width|height|length)\b",
            r"\bhow (big|large|wide|tall)\b",
        ],
        "material_query": [
            r"\b(material|aluminum|vinyl|steel|glass)\b",
            r"\bwhat material\b",
        ],
        "compliance_check": [
            r"\b(comply|compliance|meet|standard|code|aama|astm|nfrc)\b",
            r"\bdoes .* meet\b",
        ],
        "comparison": [
            r"\b(compare|difference|vs|versus|rev)\b",
        ],
    }

    @classmethod
    def classify(cls, query: str) -> str:
        """Classify query intent"""
        query_lower = query.lower()

        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        return "general"  # Default


class BM25Index:
    """BM25 index for keyword-based retrieval"""

    def __init__(self):
        if not BM25_AVAILABLE:
            raise ImportError("BM25 not available. Install: pip install rank-bm25")

        self.bm25 = None
        self.documents = []
        self.metadata = []

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 retrieval"""
        self.documents = documents
        self.metadata = [doc.get('metadata', {}) for doc in documents]

        # Tokenize documents
        tokenized_docs = [
            doc.get('text', '').lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"Indexed {len(documents)} documents in BM25")

    def search(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        """Search documents using BM25"""
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k results
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx].get('text', ''),
                'metadata': self.metadata[idx],
                'score': float(scores[idx]),
                'source': 'bm25'
            })

        return results


class DomainQAEngine:
    """
    Master Domain QA Engine implementing the full system prompt workflow
    Modes: Admin, PM, Peer Review
    """

    def __init__(self,
                 openai_api_key: str,
                 tenant_id: str,
                 docai_credentials: Optional[Dict[str, str]] = None,
                 pinecone_credentials: Optional[Dict[str, str]] = None):
        """
        Initialize the Domain QA Engine

        Args:
            openai_api_key: OpenAI API key
            tenant_id: Tenant identifier for namespace isolation
            docai_credentials: Document AI credentials
            pinecone_credentials: Pinecone credentials
        """
        self.tenant_id = tenant_id
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Initialize intelligent parser
        self.parser = IntelligentDocumentParser(
            openai_api_key=openai_api_key,
            docai_project_id=docai_credentials.get('project_id') if docai_credentials else None,
            docai_location=docai_credentials.get('location') if docai_credentials else None,
            docai_processor_id=docai_credentials.get('processor_id') if docai_credentials else None,
            pinecone_api_key=pinecone_credentials.get('api_key') if pinecone_credentials else None,
            pinecone_environment=pinecone_credentials.get('environment') if pinecone_credentials else None,
        )

        # Initialize BM25 index
        self.bm25_index = BM25Index() if BM25_AVAILABLE else None

        # Initialize continuous learning engine
        learning_dir = f"./data/learning/{tenant_id}"
        self.learning_engine = ContinuousLearningEngine(storage_dir=learning_dir)

        logger.info(f"Initialized DomainQAEngine for tenant: {tenant_id}")

    def get_namespace(self, project_id: str, discipline: Optional[str] = None) -> str:
        """Get Pinecone namespace for tenant/project/discipline"""
        if discipline:
            return f"{self.tenant_id}:{project_id}:{discipline}"
        else:
            return f"{self.tenant_id}:{project_id}"

    def admin_ingest(self,
                    file_paths: List[str],
                    project_id: str,
                    discipline: str = "fenestration",
                    doc_type: str = "shop_drawing",
                    system_id: Optional[str] = None,
                    rev: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Admin Mode: Ingest and index documents with full metadata

        Returns coverage report
        """
        logger.info(f"Admin ingestion started: {len(file_paths)} files for {self.tenant_id}:{project_id}")

        # Initialize coverage report
        report = COVERAGE_REPORT_TEMPLATE.copy()
        report['tenant_id'] = self.tenant_id
        report['project_id'] = project_id
        report['ingestion_timestamp'] = datetime.now().isoformat()

        namespace = self.get_namespace(project_id, discipline)
        all_chunks = []

        for file_path in file_paths:
            try:
                # Parse document with intelligent parser
                metadata, tables, full_text = self.parser.parse_document(file_path, project_id)

                # ★ CONTINUOUS LEARNING: Learn from this document
                learning_results = self.learning_engine.learn_from_document(
                    metadata,
                    full_text,
                    user_role="admin"  # Trainer upload
                )
                logger.info(f"Learning results: {learning_results}")

                # Extract enhanced metadata
                enhanced_meta = EnhancedMetadata(
                    tenant_id=self.tenant_id,
                    project_id=project_id,
                    discipline=discipline,
                    system_id=system_id or metadata.drawing_number or "unknown",
                    doc_type=doc_type or metadata.document_type,
                    rev=rev or metadata.revision or "A",
                    date=metadata.date or datetime.now().strftime("%Y-%m-%d"),
                    page=1,  # Will be updated per chunk
                    section="",
                    tags=tags or [],
                    confidence=metadata.confidence_score,
                    filename=metadata.filename,
                    extracted_dimensions=metadata.extracted_dimensions,
                    extracted_materials=metadata.extracted_materials,
                    extracted_specs=metadata.extracted_specs,
                )

                # Chunk by logical sections (not fixed tokens)
                chunks = self._chunk_by_sections(full_text, tables, enhanced_meta)
                all_chunks.extend(chunks)

                # Update coverage report
                report['files_processed'] += 1
                report['total_pages'] += metadata.page_count
                report['total_chunks'] += len(chunks)

                doc_type_key = enhanced_meta.doc_type
                report['doc_type_counts'][doc_type_key] = report['doc_type_counts'].get(doc_type_key, 0) + 1

                # Track low confidence pages
                if metadata.confidence_score < 0.9:
                    report['low_confidence_pages'].append({
                        'file': metadata.filename,
                        'confidence': metadata.confidence_score
                    })

                # Add extracted entities
                report['extracted_entities']['dimensions'].extend(metadata.extracted_dimensions)
                report['extracted_entities']['materials'].extend(metadata.extracted_materials)
                report['extracted_entities']['specs'].update(metadata.extracted_specs)
                if enhanced_meta.system_id != "unknown":
                    report['extracted_entities']['systems'].append(enhanced_meta.system_id)

                logger.info(f"Processed {file_path}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                report['warnings'].append(f"Failed to process {file_path}: {str(e)}")

        # Index in Pinecone if available
        if self.parser.pinecone_store and all_chunks:
            # Create embeddings
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self.parser._create_embeddings(texts)

            # Prepare documents for Pinecone
            documents = []
            for chunk, embedding in zip(all_chunks, embeddings):
                documents.append({
                    'text': chunk['text'],
                    'source': chunk['metadata'].filename,
                    'page': chunk['metadata'].page,
                    'section': chunk['metadata'].section,
                    'document_type': chunk['metadata'].doc_type,
                    'project': project_id,
                    'metadata': chunk['metadata'].to_pinecone_metadata()
                })

            self.parser.pinecone_store.upsert_documents(documents, embeddings)
            logger.info(f"Indexed {len(documents)} chunks in Pinecone namespace: {namespace}")

        # Build BM25 index if available
        if self.bm25_index and all_chunks:
            self.bm25_index.index_documents(all_chunks)

        # Deduplicate extracted entities
        report['extracted_entities']['dimensions'] = list(set(report['extracted_entities']['dimensions']))
        report['extracted_entities']['materials'] = list(set(report['extracted_entities']['materials']))
        report['extracted_entities']['systems'] = list(set(report['extracted_entities']['systems']))

        logger.info(f"Admin ingestion complete: {report['total_chunks']} chunks indexed")

        return report

    def _chunk_by_sections(self,
                          full_text: str,
                          tables: List[ExtractedTable],
                          base_metadata: EnhancedMetadata) -> List[Dict[str, Any]]:
        """
        Chunk text by logical sections (views, details, tables) rather than fixed tokens
        """
        chunks = []

        # Split by page breaks
        pages = full_text.split('\f')

        for page_idx, page_text in enumerate(pages):
            if not page_text.strip():
                continue

            # Split by common section markers
            section_patterns = [
                r'(DETAIL \d+/[A-Z]\d+\.\d+)',
                r'(SECTION [A-Z]-[A-Z])',
                r'(VIEW [A-Z])',
                r'(ELEVATION [A-Z])',
                r'(NOTES?:)',
                r'(GENERAL NOTES)',
            ]

            sections = [page_text]  # Start with full page
            for pattern in section_patterns:
                new_sections = []
                for section in sections:
                    parts = re.split(pattern, section, flags=re.IGNORECASE)
                    new_sections.extend([p for p in parts if p.strip()])
                sections = new_sections

            # Create chunks from sections
            for section_idx, section_text in enumerate(sections):
                if len(section_text.strip()) < 50:  # Skip tiny fragments
                    continue

                metadata = EnhancedMetadata(
                    tenant_id=base_metadata.tenant_id,
                    project_id=base_metadata.project_id,
                    discipline=base_metadata.discipline,
                    system_id=base_metadata.system_id,
                    doc_type=base_metadata.doc_type,
                    rev=base_metadata.rev,
                    date=base_metadata.date,
                    page=page_idx + 1,
                    section=f"p{page_idx+1}_s{section_idx}",
                    tags=base_metadata.tags,
                    confidence=base_metadata.confidence,
                    filename=base_metadata.filename,
                    extracted_dimensions=base_metadata.extracted_dimensions,
                    extracted_materials=base_metadata.extracted_materials,
                    extracted_specs=base_metadata.extracted_specs,
                )

                chunks.append({
                    'text': section_text.strip(),
                    'metadata': metadata,
                })

        # Add tables as separate chunks
        for table in tables:
            table_text = f"Table {table.table_index} ({table.table_type}):\n"
            table_text += "Headers: " + " | ".join(table.headers) + "\n"
            for row in table.rows[:10]:  # First 10 rows
                table_text += " | ".join(str(cell) for cell in row) + "\n"

            metadata = EnhancedMetadata(
                tenant_id=base_metadata.tenant_id,
                project_id=base_metadata.project_id,
                discipline=base_metadata.discipline,
                system_id=base_metadata.system_id,
                doc_type=base_metadata.doc_type,
                rev=base_metadata.rev,
                date=base_metadata.date,
                page=table.page_number,
                section=f"table_{table.table_index}",
                tags=base_metadata.tags + [table.table_type],
                confidence=table.confidence,
                filename=base_metadata.filename,
            )

            chunks.append({
                'text': table_text,
                'metadata': metadata,
            })

        return chunks

    def pm_query(self,
                 question: str,
                 project_id: str,
                 discipline: Optional[str] = None,
                 system_id: Optional[str] = None) -> PMResponse:
        """
        PM Mode: Answer questions using only indexed content
        Returns structured JSON response with citations
        """
        logger.info(f"PM query: {question} for {self.tenant_id}:{project_id}")

        # Classify intent
        intent = IntentClassifier.classify(question)
        logger.info(f"Classified intent: {intent}")

        # Hybrid retrieval: BM25 + Semantic
        bm25_results = []
        if self.bm25_index:
            bm25_results = self.bm25_index.search(question, top_k=RERANK_CONFIG['bm25_top_k'])

        semantic_results = []
        if self.parser.pinecone_store:
            # Build filter
            filter_meta = {'project': project_id}
            if discipline:
                filter_meta['discipline'] = discipline
            if system_id:
                filter_meta['system_id'] = system_id

            semantic_results = self.parser.query_documents(
                question,
                project_name=project_id,
                top_k=RERANK_CONFIG['rerank_top_k']
            )

        # Combine and re-rank results
        all_results = self._rerank_results(question, bm25_results, semantic_results)

        # If insufficient results, return low confidence
        if not all_results or len(all_results) < 2:
            return PMResponse(
                answer="Insufficient evidence in library. Please upload the relevant shop drawing, specification, or calculation sheet.",
                citations=[],
                assumptions=[],
                followups=["Which specific document contains this information?"],
                confidence=0.0
            )

        # Generate answer using LLM with master system prompt
        answer, citations, confidence = self._generate_answer(question, all_results, project_id, discipline, system_id)

        # Generate follow-up questions
        followups = self._generate_followups(intent, all_results)

        # Extract assumptions
        assumptions = self._extract_assumptions(all_results)

        response = PMResponse(
            answer=answer,
            citations=citations,
            assumptions=assumptions,
            followups=followups,
            confidence=confidence
        )

        # ★ CONTINUOUS LEARNING: Learn from this query
        self.learning_engine.learn_from_query(question, intent, all_results, confidence)

        logger.info(f"PM response generated: confidence={confidence:.2f}")

        return response

    def _rerank_results(self,
                        query: str,
                        bm25_results: List[Dict],
                        semantic_results: List[Dict]) -> List[Dict]:
        """Combine BM25 and semantic results with weighted re-ranking"""
        # Normalize scores
        max_bm25 = max([r['score'] for r in bm25_results], default=1.0)
        max_semantic = max([r['score'] for r in semantic_results], default=1.0)

        for r in bm25_results:
            r['normalized_score'] = r['score'] / max_bm25

        for r in semantic_results:
            r['normalized_score'] = r['score'] / max_semantic

        # Merge results by text similarity (avoid duplicates)
        merged = {}

        for result in bm25_results:
            key = result['text'][:100]  # First 100 chars as key
            merged[key] = result
            merged[key]['final_score'] = result['normalized_score'] * RERANK_CONFIG['bm25_weight']

        for result in semantic_results:
            key = result['text'][:100]
            if key in merged:
                # Combine scores
                merged[key]['final_score'] += result['normalized_score'] * RERANK_CONFIG['semantic_weight']
            else:
                merged[key] = result
                merged[key]['final_score'] = result['normalized_score'] * RERANK_CONFIG['semantic_weight']

        # Sort by final score
        reranked = sorted(merged.values(), key=lambda x: x['final_score'], reverse=True)

        return reranked[:RERANK_CONFIG['rerank_top_k']]

    def _generate_answer(self,
                        question: str,
                        results: List[Dict],
                        project_id: str,
                        discipline: Optional[str],
                        system_id: Optional[str]) -> Tuple[str, List[Dict], float]:
        """Generate answer using LLM with strict grounding"""
        # Build context from top results
        context = "\n\n".join([
            f"[Source: {r.get('source', 'unknown')}, Page: {r.get('page', '?')}]\n{r['text']}"
            for r in results[:5]
        ])

        # Format PM query prompt
        prompt = PM_QUERY_PROMPT.format(
            tenant_id=self.tenant_id,
            project_id=project_id,
            discipline=discipline or "all",
            system_id=system_id or "all",
            question=question
        )

        # Call OpenAI with master system prompt
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
                ],
                temperature=0.0,  # Deterministic for consistency
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content.strip()

            # Try to parse as JSON
            try:
                response_json = json.loads(response_text)
                answer = response_json.get('answer', response_text)
                confidence = response_json.get('confidence', 0.7)
                citations_raw = response_json.get('citations', [])
            except json.JSONDecodeError:
                # Fallback to plain text
                answer = response_text
                confidence = 0.6
                citations_raw = []

            # Extract citations from results
            citations = []
            for result in results[:3]:
                citations.append({
                    "doc": result.get('source', 'unknown'),
                    "page": result.get('page', 0),
                    "section": result.get('section', ''),
                })

            return answer, citations, confidence

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer", [], 0.0

    def _generate_followups(self, intent: str, results: List[Dict]) -> List[str]:
        """Generate relevant follow-up questions"""
        followups = []

        if intent == "spec_lookup":
            followups.append("Are there any special installation requirements?")
            followups.append("What are the test results or approvals?")
        elif intent == "dimension_query":
            followups.append("What are the clearance requirements?")
            followups.append("Are there any tolerance specifications?")
        elif intent == "material_query":
            followups.append("What finish or coating is specified?")
            followups.append("Are there alternative materials approved?")
        else:
            followups.append("Are there related details or sections?")

        return followups[:2]

    def _extract_assumptions(self, results: List[Dict]) -> List[str]:
        """Extract assumptions from results"""
        assumptions = []

        # Check for revision consistency
        revs = set([r.get('metadata', {}).get('rev', 'unknown') for r in results])
        if len(revs) > 1 and 'unknown' in revs:
            revs.remove('unknown')
        if len(revs) > 1:
            assumptions.append(f"Multiple revisions found: {', '.join(revs)}. Using latest.")

        # Check for confidence
        avg_confidence = sum([r.get('metadata', {}).get('confidence', 1.0) for r in results]) / len(results)
        if avg_confidence < 0.9:
            assumptions.append(f"OCR confidence is {avg_confidence:.1%}. Verify critical dimensions.")

        return assumptions

    def peer_review(self,
                   file_path: str,
                   project_id: str,
                   discipline: str = "fenestration") -> PeerReviewResult:
        """
        Peer Review Mode: Validate document against standards
        """
        logger.info(f"Peer review started for {file_path}")

        # Parse document
        metadata, tables, full_text = self.parser.parse_document(file_path, project_id)

        # Query global standards
        if self.parser.pinecone_store:
            # Check against global namespace
            global_namespace = f"{self.tenant_id}:global"
            # Implementation would query standards and compare
            pass

        # Perform checks (simplified for now)
        issues = []
        strengths = []

        # Check completeness
        if not metadata.drawing_number:
            issues.append(PeerReviewIssue(
                severity="major",
                category="completeness",
                page=1,
                section="title block",
                description="Drawing number not found",
                reference_doc="Drawing Standards",
                recommendation="Add drawing number in title block"
            ))

        if not metadata.revision:
            issues.append(PeerReviewIssue(
                severity="minor",
                category="completeness",
                page=1,
                section="title block",
                description="Revision not clearly marked",
                reference_doc="Drawing Standards",
                recommendation="Add revision letter/number"
            ))

        # Check for tables
        if len(tables) > 0:
            strengths.append(f"Document includes {len(tables)} well-structured tables")

        # Check OCR confidence
        if metadata.confidence_score < 0.9:
            issues.append(PeerReviewIssue(
                severity="minor",
                category="clarity",
                page=0,
                section="overall",
                description=f"OCR confidence is {metadata.confidence_score:.1%}",
                reference_doc="Quality Standards",
                recommendation="Consider rescanning at higher resolution"
            ))
        else:
            strengths.append(f"High OCR confidence ({metadata.confidence_score:.1%})")

        # Calculate overall score
        critical_count = sum(1 for i in issues if i.severity == "critical")
        major_count = sum(1 for i in issues if i.severity == "major")
        minor_count = sum(1 for i in issues if i.severity == "minor")

        overall_score = max(0, 1.0 - (critical_count * 0.3 + major_count * 0.15 + minor_count * 0.05))

        # Determine status
        if critical_count > 0:
            status = "rejected"
        elif major_count > 2:
            status = "needs_revision"
        else:
            status = "approved"

        result = PeerReviewResult(
            status=status,
            overall_score=overall_score,
            issues=issues,
            strengths=strengths,
            summary=f"Review complete: {status}. Found {len(issues)} issues and {len(strengths)} strengths."
        )

        logger.info(f"Peer review complete: {status}, score={overall_score:.2f}")

        return result

    def get_learning_stats(self) -> LearningStats:
        """Get comprehensive learning statistics"""
        return self.learning_engine.get_learning_stats()

    def get_top_patterns(self, top_k: int = 10) -> List[Any]:
        """Get most frequently observed patterns"""
        return self.learning_engine.pattern_engine.get_top_patterns(top_k)

    def get_knowledge_insights(self) -> Dict[str, Any]:
        """Get insights from the knowledge graph"""
        return {
            "top_systems": [n.to_dict() for n in self.learning_engine.knowledge_graph.get_top_nodes("system", 5)],
            "top_materials": [n.to_dict() for n in self.learning_engine.knowledge_graph.get_top_nodes("material", 5)],
            "top_specs": [n.to_dict() for n in self.learning_engine.knowledge_graph.get_top_nodes("spec", 5)],
        }

    def export_learning_for_finetuning(self, output_dir: str):
        """Export learning data for model fine-tuning"""
        self.learning_engine.export_knowledge_base(output_dir)


# Convenience functions
def create_qa_engine(openai_api_key: str,
                     tenant_id: str,
                     docai_credentials: Optional[Dict[str, str]] = None,
                     pinecone_credentials: Optional[Dict[str, str]] = None) -> DomainQAEngine:
    """Create a Domain QA Engine instance"""
    return DomainQAEngine(
        openai_api_key=openai_api_key,
        tenant_id=tenant_id,
        docai_credentials=docai_credentials,
        pinecone_credentials=pinecone_credentials
    )

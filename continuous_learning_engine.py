"""
Continuous Learning Engine - Self-Improving AI from All Uploads
Learns from trainer (admin) and user document uploads to build ever-growing knowledge
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re

# Machine learning for pattern recognition
try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("scikit-learn not available for clustering")

# Import domain components
from config import (
    CONFIDENCE_THRESHOLDS,
    DOC_TYPES,
    DISCIPLINES,
    QUERY_INTENTS,
)

from intelligent_document_parser import (
    TechnicalEntityExtractor,
    DocumentMetadata,
    ExtractedTable,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentPattern:
    """Represents a learned pattern from documents"""
    pattern_id: str
    pattern_type: str  # dimension_pattern, material_combination, calc_template, detail_type
    pattern_data: Dict[str, Any]
    frequency: int
    confidence: float
    first_seen: str  # ISO timestamp
    last_seen: str  # ISO timestamp
    source_docs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentRelationship:
    """Represents a relationship between two documents"""
    doc1: str
    doc2: str
    relationship_type: str  # similar, referenced_by, revision_of, related_system
    similarity_score: float
    shared_entities: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    node_id: str
    node_type: str  # system, detail, spec, material, calc_type
    name: str
    properties: Dict[str, Any]
    related_docs: List[str]
    frequency: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningStats:
    """Statistics about what the AI has learned"""
    total_documents: int
    total_pages: int
    total_chunks: int
    unique_systems: int
    unique_materials: int
    unique_dimensions: int
    unique_specs: int
    patterns_discovered: int
    relationships_mapped: int
    queries_answered: int
    knowledge_nodes: int
    learning_rate: float  # How fast knowledge is growing

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PatternRecognitionEngine:
    """Recognizes and learns patterns across documents"""

    def __init__(self):
        self.patterns = {}  # pattern_id -> DocumentPattern
        self.pattern_index = defaultdict(list)  # pattern_type -> [pattern_ids]

    def extract_patterns(self, metadata: DocumentMetadata, full_text: str) -> List[DocumentPattern]:
        """Extract patterns from a document"""
        patterns = []

        # 1. Dimension patterns (e.g., "48 x 72 window")
        dimension_patterns = self._extract_dimension_patterns(metadata, full_text)
        patterns.extend(dimension_patterns)

        # 2. Material combinations (e.g., "aluminum + low-e glass + argon")
        material_patterns = self._extract_material_patterns(metadata)
        patterns.extend(material_patterns)

        # 3. Spec combinations (e.g., "U-factor 0.29 + SHGC 0.32")
        spec_patterns = self._extract_spec_patterns(metadata)
        patterns.extend(spec_patterns)

        # 4. Calculation templates (e.g., "wind load = pressure x area")
        calc_patterns = self._extract_calc_patterns(full_text)
        patterns.extend(calc_patterns)

        # 5. Detail naming patterns (e.g., "Detail 3/A5.1")
        detail_patterns = self._extract_detail_patterns(full_text)
        patterns.extend(detail_patterns)

        return patterns

    def _extract_dimension_patterns(self, metadata: DocumentMetadata, text: str) -> List[DocumentPattern]:
        """Extract dimensional patterns"""
        patterns = []

        # Common dimension patterns
        dimension_combos = []
        dims = metadata.extracted_dimensions

        if len(dims) >= 2:
            # Look for width x height patterns
            for i in range(len(dims) - 1):
                combo = f"{dims[i]} x {dims[i+1]}"
                dimension_combos.append(combo)

        for combo in dimension_combos:
            pattern_id = f"dim_{hash(combo) % 10000}"
            pattern = DocumentPattern(
                pattern_id=pattern_id,
                pattern_type="dimension_pattern",
                pattern_data={"dimensions": combo},
                frequency=1,
                confidence=0.8,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                source_docs=[metadata.filename]
            )
            patterns.append(pattern)

        return patterns

    def _extract_material_patterns(self, metadata: DocumentMetadata) -> List[DocumentPattern]:
        """Extract material combination patterns"""
        patterns = []

        materials = metadata.extracted_materials
        if len(materials) >= 2:
            # Create combination pattern
            material_combo = " + ".join(sorted(materials))
            pattern_id = f"mat_{hash(material_combo) % 10000}"

            pattern = DocumentPattern(
                pattern_id=pattern_id,
                pattern_type="material_combination",
                pattern_data={"materials": materials},
                frequency=1,
                confidence=0.9,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                source_docs=[metadata.filename]
            )
            patterns.append(pattern)

        return patterns

    def _extract_spec_patterns(self, metadata: DocumentMetadata) -> List[DocumentPattern]:
        """Extract specification combination patterns"""
        patterns = []

        specs = metadata.extracted_specs
        if specs:
            spec_combo = json.dumps(specs, sort_keys=True)
            pattern_id = f"spec_{hash(spec_combo) % 10000}"

            pattern = DocumentPattern(
                pattern_id=pattern_id,
                pattern_type="spec_combination",
                pattern_data={"specs": specs},
                frequency=1,
                confidence=0.95,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                source_docs=[metadata.filename]
            )
            patterns.append(pattern)

        return patterns

    def _extract_calc_patterns(self, text: str) -> List[DocumentPattern]:
        """Extract calculation template patterns"""
        patterns = []

        # Common calculation patterns
        calc_patterns_re = [
            (r'(\w+)\s*=\s*(\w+)\s*[x×]\s*(\w+)', 'multiplication'),
            (r'(\w+)\s*=\s*(\w+)\s*[+]\s*(\w+)', 'addition'),
            (r'(\w+)\s*=\s*(\w+)\s*/\s*(\w+)', 'division'),
            (r'load\s*=\s*([^,\n]+)', 'load_calculation'),
            (r'area\s*=\s*([^,\n]+)', 'area_calculation'),
        ]

        for pattern_re, calc_type in calc_patterns_re:
            matches = re.finditer(pattern_re, text, re.IGNORECASE)
            for match in matches:
                formula = match.group(0)
                pattern_id = f"calc_{calc_type}_{hash(formula) % 10000}"

                pattern = DocumentPattern(
                    pattern_id=pattern_id,
                    pattern_type="calc_template",
                    pattern_data={"formula": formula, "calc_type": calc_type},
                    frequency=1,
                    confidence=0.7,
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat(),
                    source_docs=[]
                )
                patterns.append(pattern)

        return patterns

    def _extract_detail_patterns(self, text: str) -> List[DocumentPattern]:
        """Extract detail naming patterns"""
        patterns = []

        # Detail patterns like "Detail 3/A5.1" or "Section A-A"
        detail_patterns_re = [
            r'Detail\s+\d+/[A-Z]\d+\.\d+',
            r'Section\s+[A-Z]-[A-Z]',
            r'View\s+[A-Z]',
            r'Elevation\s+[A-Z]',
        ]

        for pattern_re in detail_patterns_re:
            matches = re.finditer(pattern_re, text, re.IGNORECASE)
            for match in matches:
                detail_name = match.group(0)
                pattern_id = f"detail_{hash(detail_name) % 10000}"

                pattern = DocumentPattern(
                    pattern_id=pattern_id,
                    pattern_type="detail_type",
                    pattern_data={"detail_name": detail_name},
                    frequency=1,
                    confidence=0.9,
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat(),
                    source_docs=[]
                )
                patterns.append(pattern)

        return patterns

    def update_pattern(self, pattern: DocumentPattern, new_source: str):
        """Update existing pattern with new occurrence"""
        pattern.frequency += 1
        pattern.last_seen = datetime.now().isoformat()
        if new_source not in pattern.source_docs:
            pattern.source_docs.append(new_source)

        # Increase confidence as pattern is seen more
        pattern.confidence = min(0.99, pattern.confidence + 0.01)

    def learn_patterns(self, patterns: List[DocumentPattern]):
        """Learn from extracted patterns"""
        for pattern in patterns:
            if pattern.pattern_id in self.patterns:
                # Update existing pattern
                self.update_pattern(self.patterns[pattern.pattern_id], pattern.source_docs[0])
            else:
                # Add new pattern
                self.patterns[pattern.pattern_id] = pattern
                self.pattern_index[pattern.pattern_type].append(pattern.pattern_id)

        logger.info(f"Learned {len(patterns)} patterns. Total patterns: {len(self.patterns)}")

    def get_patterns_by_type(self, pattern_type: str) -> List[DocumentPattern]:
        """Get all patterns of a specific type"""
        pattern_ids = self.pattern_index.get(pattern_type, [])
        return [self.patterns[pid] for pid in pattern_ids]

    def get_top_patterns(self, top_k: int = 10) -> List[DocumentPattern]:
        """Get most frequent patterns"""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.frequency,
            reverse=True
        )
        return sorted_patterns[:top_k]


class DocumentRelationshipMapper:
    """Maps relationships between documents"""

    def __init__(self):
        self.relationships = []  # List of DocumentRelationship
        self.doc_graph = defaultdict(list)  # doc -> [related_docs]

    def build_relationships(self, docs_metadata: List[DocumentMetadata]) -> List[DocumentRelationship]:
        """Build relationships between documents"""
        relationships = []

        for i, doc1 in enumerate(docs_metadata):
            for doc2 in docs_metadata[i+1:]:
                # Check for various relationship types

                # 1. Similarity based on shared entities
                shared_dims = set(doc1.extracted_dimensions) & set(doc2.extracted_dimensions)
                shared_materials = set(doc1.extracted_materials) & set(doc2.extracted_materials)
                shared_specs = set(doc1.extracted_specs.keys()) & set(doc2.extracted_specs.keys())

                shared_entities = list(shared_dims) + list(shared_materials) + list(shared_specs)

                if len(shared_entities) >= 2:
                    # Calculate similarity score
                    total_entities = len(set(doc1.extracted_dimensions + doc2.extracted_dimensions +
                                            doc1.extracted_materials + doc2.extracted_materials))
                    similarity = len(shared_entities) / max(total_entities, 1)

                    rel = DocumentRelationship(
                        doc1=doc1.filename,
                        doc2=doc2.filename,
                        relationship_type="similar",
                        similarity_score=similarity,
                        shared_entities=shared_entities
                    )
                    relationships.append(rel)

                # 2. Revision relationship
                if doc1.drawing_number and doc2.drawing_number:
                    if doc1.drawing_number == doc2.drawing_number and doc1.revision != doc2.revision:
                        rel = DocumentRelationship(
                            doc1=doc1.filename,
                            doc2=doc2.filename,
                            relationship_type="revision_of",
                            similarity_score=1.0,
                            shared_entities=[doc1.drawing_number]
                        )
                        relationships.append(rel)

                # 3. Related system (similar drawing numbers)
                if doc1.drawing_number and doc2.drawing_number:
                    # Extract system prefix (e.g., "MG-2500" from "MG-2500-101")
                    prefix1 = '-'.join(doc1.drawing_number.split('-')[:2])
                    prefix2 = '-'.join(doc2.drawing_number.split('-')[:2])

                    if prefix1 == prefix2 and doc1.drawing_number != doc2.drawing_number:
                        rel = DocumentRelationship(
                            doc1=doc1.filename,
                            doc2=doc2.filename,
                            relationship_type="related_system",
                            similarity_score=0.8,
                            shared_entities=[prefix1]
                        )
                        relationships.append(rel)

        return relationships

    def learn_relationships(self, relationships: List[DocumentRelationship]):
        """Learn from relationships"""
        for rel in relationships:
            self.relationships.append(rel)
            self.doc_graph[rel.doc1].append(rel.doc2)
            self.doc_graph[rel.doc2].append(rel.doc1)

        logger.info(f"Learned {len(relationships)} relationships. Total: {len(self.relationships)}")

    def get_related_docs(self, doc_name: str, relationship_type: Optional[str] = None) -> List[str]:
        """Get related documents"""
        related = []
        for rel in self.relationships:
            if rel.doc1 == doc_name or rel.doc2 == doc_name:
                if relationship_type is None or rel.relationship_type == relationship_type:
                    other_doc = rel.doc2 if rel.doc1 == doc_name else rel.doc1
                    related.append(other_doc)
        return related


class KnowledgeGraphBuilder:
    """Builds a knowledge graph of entities and their relationships"""

    def __init__(self):
        self.nodes = {}  # node_id -> KnowledgeNode
        self.edges = defaultdict(list)  # node_id -> [connected_node_ids]

    def build_graph(self, docs_metadata: List[DocumentMetadata]):
        """Build knowledge graph from documents"""

        # Create nodes for systems
        for doc in docs_metadata:
            if doc.drawing_number:
                system_id = doc.drawing_number.split('-')[0] if '-' in doc.drawing_number else doc.drawing_number
                node_id = f"system_{system_id}"

                if node_id in self.nodes:
                    self.nodes[node_id].frequency += 1
                    self.nodes[node_id].related_docs.append(doc.filename)
                else:
                    node = KnowledgeNode(
                        node_id=node_id,
                        node_type="system",
                        name=system_id,
                        properties={"drawing_numbers": [doc.drawing_number]},
                        related_docs=[doc.filename],
                        frequency=1
                    )
                    self.nodes[node_id] = node

            # Create nodes for materials
            for material in doc.extracted_materials:
                node_id = f"material_{material}"

                if node_id in self.nodes:
                    self.nodes[node_id].frequency += 1
                    self.nodes[node_id].related_docs.append(doc.filename)
                else:
                    node = KnowledgeNode(
                        node_id=node_id,
                        node_type="material",
                        name=material,
                        properties={},
                        related_docs=[doc.filename],
                        frequency=1
                    )
                    self.nodes[node_id] = node

            # Create nodes for specs
            for spec_key, spec_value in doc.extracted_specs.items():
                node_id = f"spec_{spec_key}"

                if node_id in self.nodes:
                    self.nodes[node_id].frequency += 1
                    self.nodes[node_id].related_docs.append(doc.filename)
                    if 'values' not in self.nodes[node_id].properties:
                        self.nodes[node_id].properties['values'] = []
                    self.nodes[node_id].properties['values'].append(spec_value)
                else:
                    node = KnowledgeNode(
                        node_id=node_id,
                        node_type="spec",
                        name=spec_key,
                        properties={"values": [spec_value]},
                        related_docs=[doc.filename],
                        frequency=1
                    )
                    self.nodes[node_id] = node

        # Build edges between co-occurring entities
        for doc in docs_metadata:
            doc_nodes = []

            # Collect all nodes from this document
            if doc.drawing_number:
                system_id = doc.drawing_number.split('-')[0] if '-' in doc.drawing_number else doc.drawing_number
                doc_nodes.append(f"system_{system_id}")

            for material in doc.extracted_materials:
                doc_nodes.append(f"material_{material}")

            for spec_key in doc.extracted_specs.keys():
                doc_nodes.append(f"spec_{spec_key}")

            # Create edges between all nodes in this document
            for i, node1 in enumerate(doc_nodes):
                for node2 in doc_nodes[i+1:]:
                    if node2 not in self.edges[node1]:
                        self.edges[node1].append(node2)
                    if node1 not in self.edges[node2]:
                        self.edges[node2].append(node1)

        logger.info(f"Built knowledge graph: {len(self.nodes)} nodes, {sum(len(e) for e in self.edges.values())} edges")

    def get_connected_nodes(self, node_id: str) -> List[KnowledgeNode]:
        """Get nodes connected to a given node"""
        connected_ids = self.edges.get(node_id, [])
        return [self.nodes[nid] for nid in connected_ids if nid in self.nodes]

    def get_top_nodes(self, node_type: Optional[str] = None, top_k: int = 10) -> List[KnowledgeNode]:
        """Get most frequent nodes"""
        nodes = list(self.nodes.values())

        if node_type:
            nodes = [n for n in nodes if n.node_type == node_type]

        sorted_nodes = sorted(nodes, key=lambda n: n.frequency, reverse=True)
        return sorted_nodes[:top_k]


class QueryLearningEngine:
    """Learns from user queries and feedback"""

    def __init__(self):
        self.query_log = []  # List of query records
        self.query_patterns = Counter()  # Track common query patterns
        self.successful_retrievals = defaultdict(list)  # query -> [successful_docs]

    def log_query(self, query: str, intent: str, results: List[Dict], confidence: float, user_feedback: Optional[str] = None):
        """Log a query for learning"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent": intent,
            "num_results": len(results),
            "confidence": confidence,
            "top_docs": [r.get('source', 'unknown') for r in results[:3]],
            "user_feedback": user_feedback  # "helpful" / "not_helpful" / None
        }

        self.query_log.append(record)
        self.query_patterns[intent] += 1

        # If query was successful, remember which docs were useful
        if confidence >= CONFIDENCE_THRESHOLDS['medium']:
            for result in results[:3]:
                doc_name = result.get('source', 'unknown')
                self.successful_retrievals[query].append(doc_name)

        logger.info(f"Logged query: intent={intent}, confidence={confidence:.2f}")

    def get_query_stats(self) -> Dict[str, Any]:
        """Get statistics about queries"""
        return {
            "total_queries": len(self.query_log),
            "intent_distribution": dict(self.query_patterns),
            "avg_confidence": np.mean([q['confidence'] for q in self.query_log]) if self.query_log else 0.0,
            "low_confidence_queries": [q for q in self.query_log if q['confidence'] < CONFIDENCE_THRESHOLDS['low']],
        }

    def export_training_data(self, output_path: str):
        """Export query-document pairs for fine-tuning"""
        training_data = []

        for query, docs in self.successful_retrievals.items():
            for doc in docs:
                training_data.append({
                    "query": query,
                    "document": doc,
                    "label": "relevant"
                })

        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        logger.info(f"Exported {len(training_data)} training examples to {output_path}")


class ContinuousLearningEngine:
    """
    Main continuous learning engine that orchestrates all learning components
    """

    def __init__(self, storage_dir: str = "./data/learning"):
        """Initialize the continuous learning engine"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-engines
        self.pattern_engine = PatternRecognitionEngine()
        self.relationship_mapper = DocumentRelationshipMapper()
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.query_engine = QueryLearningEngine()

        # Document memory
        self.document_memory = []  # All ingested documents metadata

        # Load existing learning state
        self.load_state()

        logger.info("Continuous Learning Engine initialized")

    def learn_from_document(self, metadata: DocumentMetadata, full_text: str, user_role: str = "user"):
        """
        Learn from a newly uploaded document (admin or user upload)

        Args:
            metadata: Document metadata
            full_text: Full document text
            user_role: "admin" (trainer) or "user"
        """
        logger.info(f"Learning from document: {metadata.filename} (uploaded by {user_role})")

        # 1. Add to document memory
        self.document_memory.append(metadata)

        # 2. Extract and learn patterns
        patterns = self.pattern_engine.extract_patterns(metadata, full_text)
        self.pattern_engine.learn_patterns(patterns)

        # 3. Build relationships with existing documents
        relationships = self.relationship_mapper.build_relationships([metadata] + self.document_memory[:-1])
        self.relationship_mapper.learn_relationships(relationships)

        # 4. Update knowledge graph
        self.knowledge_graph.build_graph(self.document_memory)

        # 5. Save state
        self.save_state()

        logger.info(f"Learning complete. Total documents in memory: {len(self.document_memory)}")

        return {
            "patterns_extracted": len(patterns),
            "relationships_found": len(relationships),
            "knowledge_nodes": len(self.knowledge_graph.nodes),
        }

    def learn_from_query(self, query: str, intent: str, results: List[Dict], confidence: float):
        """Learn from a user query"""
        self.query_engine.log_query(query, intent, results, confidence)
        self.save_state()

    def get_learning_stats(self) -> LearningStats:
        """Get comprehensive learning statistics"""
        total_pages = sum(doc.page_count for doc in self.document_memory)

        # Count unique entities
        all_systems = set()
        all_materials = set()
        all_dimensions = set()
        all_specs = set()

        for doc in self.document_memory:
            if doc.drawing_number:
                all_systems.add(doc.drawing_number.split('-')[0] if '-' in doc.drawing_number else doc.drawing_number)
            all_materials.update(doc.extracted_materials)
            all_dimensions.update(doc.extracted_dimensions)
            all_specs.update(doc.extracted_specs.keys())

        # Calculate learning rate (documents per day)
        if len(self.document_memory) >= 2:
            # This is simplified - would calculate based on actual timestamps
            learning_rate = len(self.document_memory) / max(1, (datetime.now() - datetime.now()).days + 1)
        else:
            learning_rate = 0.0

        stats = LearningStats(
            total_documents=len(self.document_memory),
            total_pages=total_pages,
            total_chunks=0,  # Would track from ingestion
            unique_systems=len(all_systems),
            unique_materials=len(all_materials),
            unique_dimensions=len(all_dimensions),
            unique_specs=len(all_specs),
            patterns_discovered=len(self.pattern_engine.patterns),
            relationships_mapped=len(self.relationship_mapper.relationships),
            queries_answered=len(self.query_engine.query_log),
            knowledge_nodes=len(self.knowledge_graph.nodes),
            learning_rate=learning_rate
        )

        return stats

    def get_intelligent_suggestions(self, query: str) -> List[str]:
        """Get intelligent suggestions based on learned patterns"""
        suggestions = []

        # Analyze query for entities
        query_lower = query.lower()

        # Suggest related queries based on patterns
        if "dimension" in query_lower or "size" in query_lower:
            top_dims = self.pattern_engine.get_patterns_by_type("dimension_pattern")
            if top_dims:
                suggestions.append(f"Common dimensions found: {', '.join([p.pattern_data['dimensions'] for p in top_dims[:3]])}")

        if "material" in query_lower:
            top_materials = self.pattern_engine.get_patterns_by_type("material_combination")
            if top_materials:
                suggestions.append(f"Common material combinations: {', '.join([' + '.join(p.pattern_data['materials']) for p in top_materials[:3]])}")

        # Suggest related documents
        suggestions.append("Try asking about specific drawing numbers or systems I've seen before")

        return suggestions[:3]

    def export_knowledge_base(self, output_dir: str):
        """Export entire knowledge base for backup or transfer"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export patterns
        with open(output_path / "patterns.json", 'w') as f:
            json.dump([p.to_dict() for p in self.pattern_engine.patterns.values()], f, indent=2)

        # Export relationships
        with open(output_path / "relationships.json", 'w') as f:
            json.dump([r.to_dict() for r in self.relationship_mapper.relationships], f, indent=2)

        # Export knowledge graph
        with open(output_path / "knowledge_graph.json", 'w') as f:
            json.dump({
                "nodes": [n.to_dict() for n in self.knowledge_graph.nodes.values()],
                "edges": dict(self.knowledge_graph.edges)
            }, f, indent=2)

        # Export query log
        with open(output_path / "query_log.json", 'w') as f:
            json.dump(self.query_engine.query_log, f, indent=2)

        # Export training data for fine-tuning
        self.query_engine.export_training_data(str(output_path / "training_data.json"))

        logger.info(f"Exported knowledge base to {output_dir}")

    def save_state(self):
        """Save learning state to disk"""
        state = {
            "patterns": [p.to_dict() for p in self.pattern_engine.patterns.values()],
            "relationships": [r.to_dict() for r in self.relationship_mapper.relationships],
            "knowledge_graph_nodes": [n.to_dict() for n in self.knowledge_graph.nodes.values()],
            "knowledge_graph_edges": dict(self.knowledge_graph.edges),
            "query_log": self.query_engine.query_log,
            "document_count": len(self.document_memory),
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.storage_dir / "learning_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.debug("Saved learning state")

    def load_state(self):
        """Load learning state from disk"""
        state_file = self.storage_dir / "learning_state.json"

        if not state_file.exists():
            logger.info("No previous learning state found")
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore patterns
            for pattern_dict in state.get("patterns", []):
                pattern = DocumentPattern(**pattern_dict)
                self.pattern_engine.patterns[pattern.pattern_id] = pattern
                self.pattern_engine.pattern_index[pattern.pattern_type].append(pattern.pattern_id)

            # Restore relationships
            for rel_dict in state.get("relationships", []):
                rel = DocumentRelationship(**rel_dict)
                self.relationship_mapper.relationships.append(rel)

            # Restore knowledge graph
            for node_dict in state.get("knowledge_graph_nodes", []):
                node = KnowledgeNode(**node_dict)
                self.knowledge_graph.nodes[node.node_id] = node

            self.knowledge_graph.edges = defaultdict(list, state.get("knowledge_graph_edges", {}))

            # Restore query log
            self.query_engine.query_log = state.get("query_log", [])

            logger.info(f"Loaded learning state: {state.get('document_count', 0)} documents, {len(self.pattern_engine.patterns)} patterns")

        except Exception as e:
            logger.error(f"Error loading learning state: {e}")


# Convenience function
def create_learning_engine(storage_dir: str = "./data/learning") -> ContinuousLearningEngine:
    """Create a continuous learning engine instance"""
    return ContinuousLearningEngine(storage_dir=storage_dir)

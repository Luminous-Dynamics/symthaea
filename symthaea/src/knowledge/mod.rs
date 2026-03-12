//! Knowledge Engine — General-Purpose Reasoning Infrastructure
//!
//! Bridges Symthaea's HDC ontology, causal reasoning, and epistemic verification
//! into a unified knowledge system that the cognitive loop queries every cycle.
//!
//! # Architecture
//!
//! ```text
//! Input (text/percept)
//!   │
//!   ├─► Extraction ─► (entity, relation, event) tuples
//!   │                      │
//!   ├─► HDC Encoding ◄────┘   composite fact vectors
//!   │       │
//!   ├─► Knowledge Graph ◄──── temporal index, confidence decay, contradiction detection
//!   │       │
//!   ├─► Causal Bridge ◄────── auto-construct DAG edges from causal relations
//!   │       │
//!   └─► Adaptive Ontology ──► grow new primitives from experience (Hebbian)
//! ```
//!
//! # Feature Gate
//!
//! All code in this module is compiled unconditionally (no feature gate) because
//! the knowledge types are lightweight. The *cognitive loop wiring* is gated
//! behind `enable_knowledge_engine` in `CognitiveLoopConfig`.

pub mod adaptive_ontology;
pub mod causal_bridge;
pub mod causal_reasoning_bridge;
pub mod encoding;
pub mod extraction;
pub mod graph;
pub mod manager;

pub use adaptive_ontology::{AdaptiveOntology, PrimitiveUsage};
pub use causal_bridge::CausalKnowledgeBridge;
pub use causal_reasoning_bridge::CausalReasoningBridge;
pub use encoding::{FactEncoding, KnowledgeEncoder};
pub use extraction::{
    EntityType, ExtractedEntity, ExtractedFact, ExtractedRelation, KnowledgeExtractor, SemanticRole,
};
pub use graph::{ContradictionAlert, EnhancedKnowledgeGraph, FactId, TemporalFact};
pub use manager::{KnowledgeManager, KnowledgeSignals, KnowledgeTelemetry};

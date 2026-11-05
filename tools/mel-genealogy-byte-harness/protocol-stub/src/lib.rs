//! Minimal serde-compatible subset of the production Muse protocol used by
//! genealogy.rs. This shim is deliberately limited to genealogy DTOs.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GenealogyOrigin {
    MuseGenerated { seed: u64, style_name: String },
    ImportedScore { source_score_sha256: String },
    ImportedRecording { source_audio_sha256: String },
    ManuallyAuthored { initial_recipe_sha256: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GenealogyRelation {
    Root,
    Reharmonization { preserved_melody: bool },
    MotifVariation { motif_ids: Vec<String> },
    Reorchestration,
    NewPerformance,
    ManualRevision,
    ImportedArrangement,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenealogyManifest {
    pub id: i64,
    pub family_id: i64,
    pub parent_id: Option<i64>,
    pub namespace: String,
    pub relation: GenealogyRelation,
    pub origin: GenealogyOrigin,
    pub audio_key: String,
    pub recipe_sha256: String,
    pub score_sha256: Option<String>,
    pub audio_sha256: String,
    pub manifest_sha256: String,
    pub created_at_unix_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenealogyEdgeKindV1 {
    DerivedFrom,
    MotifContribution,
    ArrangementContribution,
    PerformanceSource,
    ReferenceOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenealogyParentInputV1 {
    pub parent_manifest_id: i64,
    pub kind: GenealogyEdgeKindV1,
    pub source_event_id: Option<String>,
    pub contributor_ref: Option<String>,
    pub authority_receipt_ref: Option<String>,
    pub idempotency_key: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenealogyParentEdgeV1 {
    pub id: i64,
    pub parent_manifest_id: i64,
    pub child_manifest_id: i64,
    pub kind: GenealogyEdgeKindV1,
    pub source_event_id: Option<String>,
    pub contributor_ref: Option<String>,
    pub authority_receipt_ref: Option<String>,
    pub idempotency_key: String,
    pub edge_sha256: String,
    pub created_at_unix_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenealogyLineageVerificationV1 {
    pub manifest: GenealogyManifest,
    pub parent_edges: Vec<GenealogyParentEdgeV1>,
    pub lineage_sha256: String,
}

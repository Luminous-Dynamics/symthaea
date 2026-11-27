//! Strict raw-trajectory wire adapter for the standalone MACE BM25 reference.
//!
//! This adapter accepts only candidate-visible source events. It does not accept
//! evaluator labels, loss tables, hidden world state or generator seeds.

#[path = "../../bm25_temporal_reference.rs"]
mod bm25;

pub use bm25::{
    AuthorityTierV1, Bm25ConfigV1, Bm25IndexV1, Bm25ResultV1, EventRelationV1, FusedHitV1,
    RankedChannelV1, RetrievalHitV1, RetrievalQueryV1, RetrievalWorkV1, SourceDocumentV1,
    weighted_rrf_v1,
};

use serde::{Deserialize, Deserializer};

const SCHEMA_VERSION: &str = "mace_raw_trajectory_candidate_v0_1";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateFixtureWireV1 {
    schema_version: String,
    documents: Vec<SourceDocumentWireV1>,
    query: QueryWireV1,
    #[serde(default)]
    ranked_channels: Vec<RankedChannelWireV1>,
    #[serde(default = "default_rrf_k")]
    rrf_k: u32,
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceDocumentWireV1 {
    document_id: String,
    source_id: String,
    scope_id: String,
    observed_at_ms: u64,
    valid_from_ms: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    valid_until_ms: Option<u64>,
    authority_tier: AuthorityTierWireV1,
    uncertainty_ppm: u32,
    relation_to_parent: EventRelationWireV1,
    parent_event_ids: Vec<String>,
    #[serde(default)]
    source_family: Option<String>,
    #[serde(default)]
    source_root_digest_sha256: Option<String>,
    text: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AuthorityTierWireV1 {
    Verified,
    Authenticated,
    UserAsserted,
    Untrusted,
}

impl From<AuthorityTierWireV1> for AuthorityTierV1 {
    fn from(value: AuthorityTierWireV1) -> Self {
        match value {
            AuthorityTierWireV1::Verified => Self::Verified,
            AuthorityTierWireV1::Authenticated => Self::Authenticated,
            AuthorityTierWireV1::UserAsserted => Self::UserAsserted,
            AuthorityTierWireV1::Untrusted => Self::Untrusted,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EventRelationWireV1 {
    Asserts,
    Supersedes,
    Retracts,
    Contradicts,
    DerivedFrom,
}

impl From<EventRelationWireV1> for EventRelationV1 {
    fn from(value: EventRelationWireV1) -> Self {
        match value {
            EventRelationWireV1::Asserts => Self::Asserts,
            EventRelationWireV1::Supersedes => Self::Supersedes,
            EventRelationWireV1::Retracts => Self::Retracts,
            EventRelationWireV1::Contradicts => Self::Contradicts,
            EventRelationWireV1::DerivedFrom => Self::DerivedFrom,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryWireV1 {
    query_text: String,
    scope_id: String,
    as_of_ms: u64,
    top_k: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankedChannelWireV1 {
    channel_id: String,
    weight: f64,
    document_ids: Vec<String>,
}

fn default_rrf_k() -> u32 {
    60
}

fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn check_max_length(field: &str, value: &str, maximum: usize) -> Result<(), AdapterErrorV1> {
    if value.chars().count() > maximum {
        return Err(AdapterErrorV1::InvalidCandidateField(field.to_owned()));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedEvidenceV1 {
    pub document_id: String,
    pub source_id: String,
    pub authority_tier: AuthorityTierV1,
    pub uncertainty_ppm: u32,
    pub relation_to_parent: EventRelationV1,
    pub parent_event_ids: Vec<String>,
    pub source_family: Option<String>,
    pub source_root_digest_sha256: Option<String>,
    pub observed_at_ms: u64,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixtureExecutionV1 {
    pub bm25: Bm25ResultV1,
    pub resolved_hits: Vec<ResolvedEvidenceV1>,
    pub fused_hits: Vec<FusedHitV1>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterErrorV1 {
    InvalidJson(String),
    UnsupportedSchemaVersion,
    InvalidCandidateField(String),
    Index(String),
    Search(String),
    Fusion(String),
    MissingResolvedHit(String),
}

impl SourceDocumentWireV1 {
    fn into_source_document(self) -> SourceDocumentV1 {
        SourceDocumentV1 {
            document_id: self.document_id,
            source_id: self.source_id,
            scope_id: self.scope_id,
            authority_tier: self.authority_tier.into(),
            uncertainty_ppm: self.uncertainty_ppm,
            relation_to_parent: self.relation_to_parent.into(),
            parent_event_ids: self.parent_event_ids,
            source_family: self.source_family,
            source_root_digest_sha256: self.source_root_digest_sha256,
            observed_at_ms: self.observed_at_ms,
            valid_from_ms: self.valid_from_ms,
            valid_until_ms: self.valid_until_ms,
            text: self.text,
        }
    }
}

fn validate_candidate_wire(fixture: &CandidateFixtureWireV1) -> Result<(), AdapterErrorV1> {
    check_max_length("query.query_text", &fixture.query.query_text, 20_000)?;
    check_max_length("query.scope_id", &fixture.query.scope_id, 256)?;
    if fixture.query.top_k > 1_000
        || fixture.top_k.is_some_and(|top_k| top_k > 1_000)
    {
        return Err(AdapterErrorV1::InvalidCandidateField("top_k".to_owned()));
    }

    for document in &fixture.documents {
        check_max_length("document.document_id", &document.document_id, 256)?;
        check_max_length("document.source_id", &document.source_id, 256)?;
        check_max_length("document.scope_id", &document.scope_id, 256)?;
        check_max_length("document.text", &document.text, 200_000)?;
        if let Some(family) = &document.source_family {
            check_max_length("document.source_family", family, 256)?;
        }
        for parent in &document.parent_event_ids {
            check_max_length("document.parent_event_ids", parent, 256)?;
        }
    }

    for channel in &fixture.ranked_channels {
        check_max_length("ranked_channel.channel_id", &channel.channel_id, 128)?;
        for document_id in &channel.document_ids {
            check_max_length("ranked_channel.document_ids", document_id, 256)?;
        }
    }
    Ok(())
}

/// Parse, validate, index, retrieve and resolve a frozen candidate-only fixture.
///
/// Returned IDs are resolved by exact event identity. No text-similarity join
/// is performed between retrieval output and canonical source metadata.
pub fn run_candidate_fixture_v1(json: &str) -> Result<FixtureExecutionV1, AdapterErrorV1> {
    let fixture: CandidateFixtureWireV1 = serde_json::from_str(json)
        .map_err(|error| AdapterErrorV1::InvalidJson(error.to_string()))?;
    if fixture.schema_version != SCHEMA_VERSION {
        return Err(AdapterErrorV1::UnsupportedSchemaVersion);
    }
    validate_candidate_wire(&fixture)?;

    let documents = fixture
        .documents
        .into_iter()
        .map(SourceDocumentWireV1::into_source_document)
        .collect();
    let index = Bm25IndexV1::build(documents, Bm25ConfigV1::default())
        .map_err(|error| AdapterErrorV1::Index(format!("{error:?}")))?;

    let query = RetrievalQueryV1 {
        query_text: fixture.query.query_text,
        scope_id: fixture.query.scope_id.clone(),
        as_of_ms: fixture.query.as_of_ms,
        top_k: fixture.query.top_k,
    };
    let bm25 = index
        .search(&query)
        .map_err(|error| AdapterErrorV1::Search(format!("{error:?}")))?;

    let mut resolved_hits = Vec::with_capacity(bm25.hits.len());
    for hit in &bm25.hits {
        let source = index
            .resolve_source_event(&hit.document_id)
            .ok_or_else(|| AdapterErrorV1::MissingResolvedHit(hit.document_id.clone()))?;
        resolved_hits.push(ResolvedEvidenceV1 {
            document_id: source.document_id.clone(),
            source_id: source.source_id.clone(),
            authority_tier: source.authority_tier,
            uncertainty_ppm: source.uncertainty_ppm,
            relation_to_parent: source.relation_to_parent,
            parent_event_ids: source.parent_event_ids.clone(),
            source_family: source.source_family.clone(),
            source_root_digest_sha256: source.source_root_digest_sha256.clone(),
            observed_at_ms: source.observed_at_ms,
            valid_from_ms: source.valid_from_ms,
            valid_until_ms: source.valid_until_ms,
        });
    }

    let channels: Vec<RankedChannelV1> = fixture
        .ranked_channels
        .into_iter()
        .map(|channel| RankedChannelV1 {
            channel_id: channel.channel_id,
            weight: channel.weight,
            document_ids: channel.document_ids,
        })
        .collect();
    let top_k = fixture.top_k.unwrap_or(query.top_k);
    let fused_hits = weighted_rrf_v1(
        &index,
        &query.scope_id,
        query.as_of_ms,
        &channels,
        fixture.rrf_k,
        top_k,
    )
    .map_err(|error| AdapterErrorV1::Fusion(format!("{error:?}")))?;

    Ok(FixtureExecutionV1 {
        bm25,
        resolved_hits,
        fused_hits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../raw-trajectory-baseline-v0.1.example.json");

    #[test]
    fn candidate_json_fixture_runs_through_index_search_and_exact_lineage_resolution() {
        let report = run_candidate_fixture_v1(FIXTURE).expect("candidate fixture");
        assert!(!report.bm25.hits.is_empty());
        assert!(!report.fused_hits.is_empty());
        assert_eq!(report.resolved_hits.len(), report.bm25.hits.len());
        for (hit, resolved) in report.bm25.hits.iter().zip(&report.resolved_hits) {
            assert_eq!(hit.document_id, resolved.document_id);
            assert!(!resolved.source_id.is_empty());
            assert!(resolved.uncertainty_ppm <= 1_000_000);
        }
        assert!(
            report
                .bm25
                .hits
                .iter()
                .all(|hit| hit.document_id != "late-correction-041")
        );
        assert!(
            report
                .bm25
                .hits
                .iter()
                .all(|hit| hit.document_id != "other-tenant-001")
        );
    }

    #[test]
    fn source_resolution_preserves_authority_family_root_and_parent_relation() {
        let report = run_candidate_fixture_v1(FIXTURE).expect("candidate fixture");
        let current = report
            .resolved_hits
            .iter()
            .find(|hit| hit.document_id == "certificate-041")
            .expect("current certificate should rank for the fixture query");
        assert_eq!(current.authority_tier, AuthorityTierV1::Verified);
        assert_eq!(current.source_family.as_deref(), Some("supplier_registry"));
        assert_eq!(
            current.source_root_digest_sha256.as_deref(),
            Some("ad74977d718be3043ed9ae57f75765a53724767753b8933e6c81bd94f7068019")
        );
    }

    #[test]
    fn unknown_fields_and_schema_versions_are_rejected() {
        let unknown = FIXTURE.replace(
            "\"schema_version\": \"mace_raw_trajectory_candidate_v0_1\"",
            "\"schema_version\": \"mace_raw_trajectory_candidate_v0_1\", \"hidden_oracle\": \"hold\"",
        );
        assert!(matches!(
            run_candidate_fixture_v1(&unknown),
            Err(AdapterErrorV1::InvalidJson(_))
        ));

        let unsupported = FIXTURE.replace(
            "mace_raw_trajectory_candidate_v0_1",
            "mace_raw_trajectory_candidate_v0_2",
        );
        assert_eq!(
            run_candidate_fixture_v1(&unsupported),
            Err(AdapterErrorV1::UnsupportedSchemaVersion)
        );
    }

    #[test]
    fn required_nullable_valid_until_field_cannot_be_omitted() {
        let missing = FIXTURE.replace(
            "            \"valid_until_ms\": 300,\n",
            "",
        );
        assert!(matches!(
            run_candidate_fixture_v1(&missing),
            Err(AdapterErrorV1::InvalidJson(_))
        ));
    }

    #[test]
    fn documented_string_and_top_k_limits_are_enforced_before_indexing() {
        let oversized_text = FIXTURE.replace(
            "Pump P-17 pressure inspection completed.",
            &"x".repeat(200_001),
        );
        assert_eq!(
            run_candidate_fixture_v1(&oversized_text),
            Err(AdapterErrorV1::InvalidCandidateField("document.text".into()))
        );

        let mut value: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        value["query"]["top_k"] = serde_json::json!(1_001);
        let oversized_top_k = serde_json::to_string(&value).unwrap();
        assert_eq!(
            run_candidate_fixture_v1(&oversized_top_k),
            Err(AdapterErrorV1::InvalidCandidateField("top_k".into()))
        );
    }

    #[test]
    fn future_observed_backdated_event_is_not_visible_at_earlier_query_cut() {
        let report = run_candidate_fixture_v1(FIXTURE).expect("candidate fixture");
        assert!(
            report
                .bm25
                .hits
                .iter()
                .all(|hit| hit.document_id != "late-correction-041")
        );
    }
}

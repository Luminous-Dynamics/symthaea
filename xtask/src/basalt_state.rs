use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TIMELINE_ENTRIES: usize = 20_000;
const MAX_INITIAL_RESOURCES: usize = 100_000;
const MAX_PINS_PER_EVIDENCE: usize = 4_096;
const MAX_EVIDENCE_PARENTS: usize = 4_096;
const MAX_TOTAL_EDGES: usize = 250_000;

/// A structurally valid timeline can still contain stale evidence. The valid
/// flag means the event/evidence history is consistent, not that every item
/// remains current or that the issuer of any event has been authenticated.
#[derive(Debug, Serialize)]
pub struct StateAuditReport {
    pub report_version: u32,
    pub run_id: Option<String>,
    pub input_digest: String,
    pub final_state_digest: Option<String>,
    pub valid: bool,
    pub counts: StateAuditCounts,
    pub evidence: Vec<EvidenceAssessment>,
    /// Stale evidence IDs in original topological/creation order.
    pub reverification_plan: Vec<String>,
    pub errors: Vec<StateAuditError>,
}

#[derive(Debug, Default, Serialize)]
pub struct StateAuditCounts {
    pub initial_resources: usize,
    pub timeline_entries: usize,
    pub resource_changes: usize,
    pub unrelated_resource_changes: usize,
    pub evidence_total: usize,
    pub evidence_fresh: usize,
    pub evidence_stale_direct: usize,
    pub evidence_stale_transitive: usize,
    pub evidence_usable_passes: usize,
    pub reverification_required: usize,
}

#[derive(Debug, Serialize)]
pub struct StateAuditError {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct EvidenceAssessment {
    pub evidence_id: String,
    pub check_id: String,
    pub evidence_kind: EvidenceKind,
    pub outcome: EvidenceOutcome,
    pub freshness: EvidenceFreshness,
    pub usable_for_success: bool,
    pub stale_resources: Vec<String>,
    pub stale_ancestor_evidence_ids: Vec<String>,
    pub invalidation_event_ids: Vec<String>,
    pub blocked_by_evidence_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Build,
    Test,
    Typecheck,
    Lint,
    Security,
    Scope,
    Regression,
    Other,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOutcome {
    Pass,
    Fail,
    Unresolved,
    NotRun,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceFreshness {
    Fresh,
    StaleDirect,
    StaleTransitive,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ChangeKind {
    SourceChange,
    DependencyChange,
    ToolchainChange,
    ConfigurationChange,
    TaskContractChange,
    EnvironmentChange,
    SessionChange,
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Timeline {
    schema_version: u32,
    program: String,
    run_id: String,
    initial_resources: Vec<InitialResource>,
    entries: Vec<TimelineEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InitialResource {
    resource_id: String,
    digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "entry_type", rename_all = "snake_case", deny_unknown_fields)]
enum TimelineEntry {
    ResourceChanged {
        sequence: u64,
        event_id: String,
        change_kind: ChangeKind,
        event_digest: String,
        resource_id: String,
        before_digest: String,
        after_digest: String,
    },
    EvidenceRecorded {
        sequence: u64,
        evidence_id: String,
        check_id: String,
        evidence_kind: EvidenceKind,
        outcome: EvidenceOutcome,
        evidence_digest: String,
        resource_pins: Vec<ResourcePin>,
        evidence_dependencies: Vec<String>,
    },
}

impl TimelineEntry {
    fn sequence(&self) -> u64 {
        match self {
            Self::ResourceChanged { sequence, .. }
            | Self::EvidenceRecorded { sequence, .. } => *sequence,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourcePin {
    resource_id: String,
    digest: String,
    generation: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ResourceVersion {
    digest: String,
    generation: u64,
}

#[derive(Serialize)]
struct ResourceChangeCommitment<'a> {
    run_id: &'a str,
    sequence: u64,
    event_id: &'a str,
    change_kind: ChangeKind,
    resource_id: &'a str,
    before_digest: &'a str,
    after_digest: &'a str,
}

fn resource_change_digest(
    run_id: &str,
    sequence: u64,
    event_id: &str,
    change_kind: ChangeKind,
    resource_id: &str,
    before_digest: &str,
    after_digest: &str,
) -> Result<String, String> {
    let commitment = ResourceChangeCommitment {
        run_id,
        sequence,
        event_id,
        change_kind,
        resource_id,
        before_digest,
        after_digest,
    };
    serde_json::to_vec(&commitment)
        .map(|bytes| format!("sha256:{}", sha256_hex(&bytes)))
        .map_err(|error| error.to_string())
}


#[derive(Debug)]
struct EvidenceState {
    assessment: EvidenceAssessment,
    pinned_resource_ids: Vec<String>,
}

#[derive(Debug)]
struct AuditFailure {
    code: &'static str,
    path: String,
    message: String,
}

fn failure(code: &'static str, path: impl Into<String>, message: impl Into<String>) -> AuditFailure {
    AuditFailure {
        code,
        path: path.into(),
        message: message.into(),
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.trim() == value
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        })
}

fn valid_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn audit_file(path: &Path) -> anyhow::Result<StateAuditReport> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    Ok(audit_bytes(&bytes))
}

pub fn audit_bytes(bytes: &[u8]) -> StateAuditReport {
    let input_digest = format!("sha256:{}", sha256_hex(bytes));
    let mut report = StateAuditReport {
        report_version: 1,
        run_id: None,
        input_digest,
        final_state_digest: None,
        valid: false,
        counts: StateAuditCounts::default(),
        evidence: Vec::new(),
        reverification_plan: Vec::new(),
        errors: Vec::new(),
    };

    if bytes.len() as u64 > MAX_INPUT_BYTES {
        push_failure(
            &mut report,
            failure("E_INPUT_TOO_LARGE", "$", "state timeline exceeds the 64 MiB input limit"),
        );
        return report;
    }

    let timeline: Timeline = match serde_json::from_slice(bytes) {
        Ok(value) => value,
        Err(error) => {
            push_failure(
                &mut report,
                failure(
                    "E_TIMELINE_JSON",
                    "$",
                    format!("invalid timeline JSON, duplicate field, unknown field, or missing field: {error}"),
                ),
            );
            return report;
        }
    };

    report.run_id = Some(timeline.run_id.clone());
    match audit_timeline(&timeline) {
        Ok(result) => {
            report.valid = true;
            report.counts = result.counts;
            report.final_state_digest = Some(result.final_state_digest);
            report.evidence = result.evidence;
            report.reverification_plan = result.reverification_plan;
        }
        Err(error) => push_failure(&mut report, error),
    }
    report
}

fn push_failure(report: &mut StateAuditReport, failure: AuditFailure) {
    report.errors.push(StateAuditError {
        code: failure.code.to_string(),
        path: failure.path,
        message: failure.message,
    });
}

struct AuditResult {
    counts: StateAuditCounts,
    final_state_digest: String,
    evidence: Vec<EvidenceAssessment>,
    reverification_plan: Vec<String>,
}

fn audit_timeline(timeline: &Timeline) -> Result<AuditResult, AuditFailure> {
    if timeline.schema_version != 1 {
        return Err(failure("E_SCHEMA_VERSION", "schema_version", "only schema_version=1 is supported"));
    }
    if timeline.program != "BASALT" {
        return Err(failure("E_PROGRAM", "program", "program must be BASALT"));
    }
    if !valid_identifier(&timeline.run_id) {
        return Err(failure("E_RUN_ID", "run_id", "run_id must be a trimmed 1–128 character identifier"));
    }
    if timeline.initial_resources.is_empty() {
        return Err(failure("E_INITIAL_RESOURCES_EMPTY", "initial_resources", "at least one initial resource is required"));
    }
    if timeline.initial_resources.len() > MAX_INITIAL_RESOURCES {
        return Err(failure("E_INITIAL_RESOURCES_LIMIT", "initial_resources", "initial resource count exceeds 100000"));
    }
    if timeline.entries.len() > MAX_TIMELINE_ENTRIES {
        return Err(failure("E_TIMELINE_ENTRIES_LIMIT", "entries", "timeline exceeds 20000 entries"));
    }

    let mut resources = BTreeMap::<String, ResourceVersion>::new();
    for (index, resource) in timeline.initial_resources.iter().enumerate() {
        let path = format!("initial_resources[{index}]");
        if !valid_identifier(&resource.resource_id) {
            return Err(failure("E_RESOURCE_ID", format!("{path}.resource_id"), "resource ID has invalid syntax"));
        }
        if !valid_digest(&resource.digest) {
            return Err(failure("E_RESOURCE_DIGEST", format!("{path}.digest"), "resource digest must be lowercase sha256:<64 hex>"));
        }
        if resources
            .insert(
                resource.resource_id.clone(),
                ResourceVersion {
                    digest: resource.digest.clone(),
                    generation: 0,
                },
            )
            .is_some()
        {
            return Err(failure("E_RESOURCE_DUPLICATE", format!("{path}.resource_id"), "initial resource IDs must be unique"));
        }
    }

    let mut event_ids = BTreeSet::<String>::new();
    let mut evidence = BTreeMap::<String, EvidenceState>::new();
    let mut evidence_order = Vec::<String>::new();
    let mut reverse_resources = BTreeMap::<String, BTreeSet<String>>::new();
    let mut resources_with_evidence = BTreeSet::<String>::new();
    let mut reverse_evidence = BTreeMap::<String, BTreeSet<String>>::new();
    let mut total_edges = 0_usize;
    let mut counts = StateAuditCounts {
        initial_resources: resources.len(),
        timeline_entries: timeline.entries.len(),
        ..StateAuditCounts::default()
    };

    for (index, entry) in timeline.entries.iter().enumerate() {
        let path = format!("entries[{index}]");
        let expected_sequence = (index as u64) + 1;
        if entry.sequence() != expected_sequence {
            return Err(failure(
                "E_SEQUENCE",
                format!("{path}.sequence"),
                format!("timeline sequence must be contiguous; expected {expected_sequence}"),
            ));
        }

        match entry {
            TimelineEntry::ResourceChanged {
                sequence,
                event_id,
                change_kind,
                event_digest,
                resource_id,
                before_digest,
                after_digest,
                ..
            } => {
                if !valid_identifier(event_id) || !event_ids.insert(event_id.clone()) {
                    return Err(failure("E_EVENT_ID", format!("{path}.event_id"), "event IDs must be valid and unique"));
                }
                if !valid_identifier(resource_id) {
                    return Err(failure("E_RESOURCE_ID", format!("{path}.resource_id"), "resource ID has invalid syntax"));
                }
                for (field, digest) in [
                    ("event_digest", event_digest),
                    ("before_digest", before_digest),
                    ("after_digest", after_digest),
                ] {
                    if !valid_digest(digest) {
                        return Err(failure("E_EVENT_DIGEST", format!("{path}.{field}"), "digest must be lowercase sha256:<64 hex>"));
                    }
                }
                let expected_event_digest = resource_change_digest(
                    &timeline.run_id,
                    *sequence,
                    event_id,
                    *change_kind,
                    resource_id,
                    before_digest,
                    after_digest,
                )
                .map_err(|error| failure("E_EVENT_COMMITMENT", &path, error))?;
                if &expected_event_digest != event_digest {
                    return Err(failure(
                        "E_EVENT_DIGEST_MISMATCH",
                        format!("{path}.event_digest"),
                        "event_digest does not bind the canonical run/sequence/change payload",
                    ));
                }
                if before_digest == after_digest {
                    return Err(failure("E_CHANGE_NOOP", &path, "resource-change event must change the pinned digest"));
                }
                let current = resources.get(resource_id).ok_or_else(|| {
                    failure("E_RESOURCE_UNKNOWN", format!("{path}.resource_id"), "resource changes must target a declared resource")
                })?;
                if &current.digest != before_digest {
                    return Err(failure("E_CHANGE_BEFORE_MISMATCH", format!("{path}.before_digest"), "before_digest does not match the current resource version"));
                }
                let next_generation = current.generation.checked_add(1).ok_or_else(|| {
                    failure("E_RESOURCE_GENERATION_OVERFLOW", &path, "resource generation counter overflow")
                })?;
                resources.insert(
                    resource_id.clone(),
                    ResourceVersion {
                        digest: after_digest.clone(),
                        generation: next_generation,
                    },
                );

                let direct_dependents = reverse_resources
                    .get(resource_id)
                    .cloned()
                    .unwrap_or_default();
                if !resources_with_evidence.contains(resource_id) {
                    counts.unrelated_resource_changes += 1;
                }
                counts.resource_changes += 1;
                invalidate_dependents(
                    &direct_dependents,
                    resource_id,
                    event_id,
                    &mut reverse_resources,
                    &reverse_evidence,
                    &mut evidence,
                );
            }
            TimelineEntry::EvidenceRecorded {
                evidence_id,
                check_id,
                evidence_kind,
                outcome,
                evidence_digest,
                resource_pins,
                evidence_dependencies,
                ..
            } => {
                if !valid_identifier(evidence_id) || evidence.contains_key(evidence_id) {
                    return Err(failure("E_EVIDENCE_ID", format!("{path}.evidence_id"), "evidence IDs must be valid and unique"));
                }
                if !valid_identifier(check_id) {
                    return Err(failure("E_CHECK_ID", format!("{path}.check_id"), "check ID has invalid syntax"));
                }
                if !valid_digest(evidence_digest) {
                    return Err(failure("E_EVIDENCE_DIGEST", format!("{path}.evidence_digest"), "evidence digest must be lowercase sha256:<64 hex>"));
                }
                if resource_pins.is_empty() {
                    return Err(failure("E_RESOURCE_PINS_EMPTY", format!("{path}.resource_pins"), "evidence must pin at least one resource"));
                }
                if resource_pins.len() > MAX_PINS_PER_EVIDENCE {
                    return Err(failure("E_RESOURCE_PINS_LIMIT", format!("{path}.resource_pins"), "evidence resource pin count exceeds 4096"));
                }
                if evidence_dependencies.len() > MAX_EVIDENCE_PARENTS {
                    return Err(failure("E_EVIDENCE_PARENTS_LIMIT", format!("{path}.evidence_dependencies"), "evidence dependency count exceeds 4096"));
                }

                let mut pin_ids = BTreeSet::new();
                for (pin_index, pin) in resource_pins.iter().enumerate() {
                    let pin_path = format!("{path}.resource_pins[{pin_index}]");
                    if !valid_identifier(&pin.resource_id) || !pin_ids.insert(pin.resource_id.clone()) {
                        return Err(failure("E_RESOURCE_PIN_ID", format!("{pin_path}.resource_id"), "resource pin IDs must be valid and unique"));
                    }
                    if !valid_digest(&pin.digest) {
                        return Err(failure("E_RESOURCE_PIN_DIGEST", format!("{pin_path}.digest"), "resource pin digest must be lowercase sha256:<64 hex>"));
                    }
                    let current = resources.get(&pin.resource_id).ok_or_else(|| {
                        failure("E_RESOURCE_PIN_UNKNOWN", format!("{pin_path}.resource_id"), "evidence pins an undeclared resource")
                    })?;
                    if pin.digest != current.digest || pin.generation != current.generation {
                        return Err(failure(
                            "E_EVIDENCE_PIN_NOT_CURRENT",
                            &pin_path,
                            format!(
                                "evidence must pin current digest and generation for {}; current generation is {}",
                                pin.resource_id, current.generation
                            ),
                        ));
                    }
                }

                let mut parent_ids = BTreeSet::new();
                let mut stale_ancestors = BTreeSet::new();
                let mut inherited_events = BTreeSet::new();
                let mut blocked_by = BTreeSet::new();
                for parent_id in evidence_dependencies {
                    if !valid_identifier(parent_id) || !parent_ids.insert(parent_id.clone()) {
                        return Err(failure("E_EVIDENCE_PARENT_ID", format!("{path}.evidence_dependencies"), "evidence parent IDs must be valid and unique"));
                    }
                    let parent = evidence.get(parent_id).ok_or_else(|| {
                        failure("E_EVIDENCE_PARENT_UNKNOWN", format!("{path}.evidence_dependencies"), "evidence dependencies must reference an earlier timeline entry")
                    })?;
                    if parent.assessment.freshness != EvidenceFreshness::Fresh {
                        stale_ancestors.insert(parent_id.clone());
                        stale_ancestors.extend(parent.assessment.stale_ancestor_evidence_ids.iter().cloned());
                        inherited_events.extend(parent.assessment.invalidation_event_ids.iter().cloned());
                    }
                    if !parent.assessment.usable_for_success {
                        blocked_by.insert(parent_id.clone());
                        blocked_by.extend(parent.assessment.blocked_by_evidence_ids.iter().cloned());
                    }
                }

                let freshness = if stale_ancestors.is_empty() {
                    EvidenceFreshness::Fresh
                } else {
                    EvidenceFreshness::StaleTransitive
                };
                let assessment = EvidenceAssessment {
                    evidence_id: evidence_id.clone(),
                    check_id: check_id.clone(),
                    evidence_kind: *evidence_kind,
                    outcome: *outcome,
                    freshness,
                    usable_for_success: freshness == EvidenceFreshness::Fresh
                        && *outcome == EvidenceOutcome::Pass
                        && blocked_by.is_empty(),
                    stale_resources: Vec::new(),
                    stale_ancestor_evidence_ids: stale_ancestors.into_iter().collect(),
                    invalidation_event_ids: inherited_events.into_iter().collect(),
                    blocked_by_evidence_ids: blocked_by.into_iter().collect(),
                };

                for pin in resource_pins {
                    resources_with_evidence.insert(pin.resource_id.clone());
                }

                // Only current evidence belongs in reverse invalidation indexes.
                // Evidence born stale through an upstream dependency remains in
                // the audit output but cannot become a propagation source.
                if freshness == EvidenceFreshness::Fresh {
                    for pin in resource_pins {
                        reverse_resources
                            .entry(pin.resource_id.clone())
                            .or_default()
                            .insert(evidence_id.clone());
                    }
                    for parent_id in evidence_dependencies {
                        reverse_evidence
                            .entry(parent_id.clone())
                            .or_default()
                            .insert(evidence_id.clone());
                    }
                }
                total_edges = total_edges
                    .checked_add(resource_pins.len())
                    .and_then(|count| count.checked_add(evidence_dependencies.len()))
                    .ok_or_else(|| failure("E_GRAPH_EDGE_OVERFLOW", &path, "evidence graph edge count overflow"))?;
                if total_edges > MAX_TOTAL_EDGES {
                    return Err(failure("E_GRAPH_EDGE_LIMIT", &path, "evidence graph exceeds 250000 total dependency edges"));
                }

                evidence.insert(
                    evidence_id.clone(),
                    EvidenceState {
                        assessment,
                        pinned_resource_ids: resource_pins.iter().map(|pin| pin.resource_id.clone()).collect(),
                    },
                );
                evidence_order.push(evidence_id.clone());
            }
        }
    }

    let mut assessments = Vec::with_capacity(evidence_order.len());
    let mut reverification_plan = Vec::new();
    for evidence_id in evidence_order {
        let state = evidence
            .remove(&evidence_id)
            .expect("every ordered evidence record remains in the evidence map");
        let assessment = state.assessment;
        if assessment.freshness != EvidenceFreshness::Fresh {
            reverification_plan.push(assessment.evidence_id.clone());
            counts.reverification_required += 1;
        }
        match assessment.freshness {
            EvidenceFreshness::Fresh => counts.evidence_fresh += 1,
            EvidenceFreshness::StaleDirect => counts.evidence_stale_direct += 1,
            EvidenceFreshness::StaleTransitive => counts.evidence_stale_transitive += 1,
        }
        if assessment.usable_for_success {
            counts.evidence_usable_passes += 1;
        }
        assessments.push(assessment);
    }
    counts.evidence_total = assessments.len();

    let final_state_bytes = serde_json::to_vec(&resources).map_err(|error| {
        failure("E_FINAL_STATE_SERIALIZE", "final_state", format!("cannot serialize canonical final resource state: {error}"))
    })?;
    let final_state_digest = format!("sha256:{}", sha256_hex(&final_state_bytes));

    Ok(AuditResult {
        counts,
        final_state_digest,
        evidence: assessments,
        reverification_plan,
    })
}

fn invalidate_dependents(
    direct_dependents: &BTreeSet<String>,
    resource_id: &str,
    event_id: &str,
    reverse_resources: &mut BTreeMap<String, BTreeSet<String>>,
    reverse_evidence: &BTreeMap<String, BTreeSet<String>>,
    evidence: &mut BTreeMap<String, EvidenceState>,
) {
    let mut queue = VecDeque::<String>::new();

    for evidence_id in direct_dependents {
        let pinned_resource_ids = if let Some(state) = evidence.get_mut(evidence_id) {
            if state.assessment.freshness != EvidenceFreshness::Fresh {
                Vec::new()
            } else {
                state.assessment.stale_resources.push(resource_id.to_string());
                state.assessment.invalidation_event_ids.push(event_id.to_string());
                state.assessment.freshness = EvidenceFreshness::StaleDirect;
                state.assessment.usable_for_success = false;
                queue.push_back(evidence_id.clone());
                state.pinned_resource_ids.clone()
            }
        } else {
            Vec::new()
        };
        for pinned_resource_id in pinned_resource_ids {
            if let Some(dependents) = reverse_resources.get_mut(&pinned_resource_id) {
                dependents.remove(evidence_id);
            }
        }
    }

    while let Some(stale_parent_id) = queue.pop_front() {
        let children = reverse_evidence
            .get(&stale_parent_id)
            .cloned()
            .unwrap_or_default();
        let (mut inherited_ancestors, inherited_events) = evidence
            .get(&stale_parent_id)
            .map(|parent| {
                (
                    parent.assessment.stale_ancestor_evidence_ids.clone(),
                    parent.assessment.invalidation_event_ids.clone(),
                )
            })
            .unwrap_or_default();
        inherited_ancestors.push(stale_parent_id.clone());

        for child_id in &children {
            let pinned_resource_ids = if let Some(child) = evidence.get_mut(child_id) {
                for ancestor_id in &inherited_ancestors {
                    if !child.assessment.stale_ancestor_evidence_ids.contains(ancestor_id) {
                        child.assessment.stale_ancestor_evidence_ids.push(ancestor_id.clone());
                    }
                }
                for invalidation_id in &inherited_events {
                    if !child.assessment.invalidation_event_ids.contains(invalidation_id) {
                        child.assessment.invalidation_event_ids.push(invalidation_id.clone());
                    }
                }
                if child.assessment.freshness == EvidenceFreshness::Fresh {
                    child.assessment.freshness = EvidenceFreshness::StaleTransitive;
                    child.assessment.usable_for_success = false;
                    queue.push_back(child_id.clone());
                    child.pinned_resource_ids.clone()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
            for pinned_resource_id in pinned_resource_ids {
                if let Some(dependents) = reverse_resources.get_mut(&pinned_resource_id) {
                    dependents.remove(child_id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const EXAMPLE: &str = include_str!("../../benchmarks/basalt/examples/state-timeline.example.json");

    fn parsed_example() -> Value {
        serde_json::from_str(EXAMPLE).expect("state timeline fixture parses")
    }

    fn audit_value(value: &Value) -> StateAuditReport {
        let bytes = serde_json::to_vec(value).expect("state timeline serializes");
        audit_bytes(&bytes)
    }

    fn has_code(report: &StateAuditReport, code: &str) -> bool {
        report.errors.iter().any(|error| error.code == code)
    }

    fn freshness(report: &StateAuditReport, evidence_id: &str) -> EvidenceFreshness {
        report
            .evidence
            .iter()
            .find(|item| item.evidence_id == evidence_id)
            .expect("evidence assessment exists")
            .freshness
    }

    #[test]
    fn synthetic_timeline_reports_direct_and_transitive_invalidation() {
        let report = audit_bytes(EXAMPLE.as_bytes());
        assert!(report.valid, "{:?}", report.errors);
        assert_eq!(report.counts.evidence_total, 6);
        assert_eq!(report.counts.evidence_stale_direct, 3);
        assert_eq!(report.counts.evidence_stale_transitive, 1);
        assert_eq!(report.counts.evidence_fresh, 2);
        assert_eq!(report.counts.reverification_required, 4);
        assert_eq!(freshness(&report, "build-before-change"), EvidenceFreshness::StaleDirect);
        assert_eq!(freshness(&report, "tests-before-change"), EvidenceFreshness::StaleDirect);
        assert_eq!(freshness(&report, "report-before-change"), EvidenceFreshness::StaleTransitive);
        let report_evidence = report.evidence.iter().find(|item| item.evidence_id == "report-before-change").unwrap();
        assert!(report_evidence.stale_ancestor_evidence_ids.iter().any(|id| id == "tests-before-change"));
        assert!(report_evidence.stale_ancestor_evidence_ids.iter().any(|id| id == "build-before-change"));
        assert!(report_evidence.invalidation_event_ids.iter().any(|id| id == "source-edit-001"));
        assert_eq!(freshness(&report, "lint-before-change"), EvidenceFreshness::StaleDirect);
        assert_eq!(freshness(&report, "build-after-change"), EvidenceFreshness::Fresh);
        assert_eq!(freshness(&report, "tests-after-change"), EvidenceFreshness::Fresh);
        assert!(report.evidence.iter().find(|item| item.evidence_id == "tests-after-change").unwrap().usable_for_success);
    }

    #[test]
    fn unrelated_resource_change_preserves_unaffected_evidence() {
        let mut value = parsed_example();
        value["entries"].as_array_mut().unwrap().remove(5);
        value["entries"].as_array_mut().unwrap().truncate(5);
        let report = audit_value(&value);
        assert!(report.valid, "{:?}", report.errors);
        assert_eq!(report.counts.evidence_fresh, 4);
        assert_eq!(report.counts.reverification_required, 0);
        assert_eq!(report.counts.unrelated_resource_changes, 1);
    }

    #[test]
    fn exact_reverification_restores_current_evidence_without_reusing_stale_parent() {
        let report = audit_bytes(EXAMPLE.as_bytes());
        assert!(report.valid);
        let build = report.evidence.iter().find(|item| item.evidence_id == "build-after-change").unwrap();
        let tests = report.evidence.iter().find(|item| item.evidence_id == "tests-after-change").unwrap();
        assert_eq!(build.freshness, EvidenceFreshness::Fresh);
        assert_eq!(tests.freshness, EvidenceFreshness::Fresh);
        assert!(build.usable_for_success && tests.usable_for_success);
        assert!(!tests.blocked_by_evidence_ids.iter().any(|id| id == "build-before-change"));
    }

    #[test]
    fn digest_reversion_does_not_resurrect_old_generation_evidence() {
        let mut value = parsed_example();
        value["entries"].as_array_mut().unwrap().push(serde_json::json!({
            "entry_type": "resource_changed",
            "sequence": 9,
            "event_id": "source-reverted",
            "change_kind": "source_change",
            "event_digest": format!("sha256:{}", "f".repeat(64)),
            "resource_id": "repo:src/lib.rs",
            "before_digest": format!("sha256:{}", "e".repeat(64)),
            "after_digest": format!("sha256:{}", "a".repeat(64))
        }));
        let report = audit_value(&value);
        assert!(report.valid, "{:?}", report.errors);
        assert_eq!(freshness(&report, "build-after-change"), EvidenceFreshness::StaleDirect);
        assert_eq!(freshness(&report, "tests-after-change"), EvidenceFreshness::StaleDirect);
        assert_eq!(report.counts.unrelated_resource_changes, 1);
    }

    #[test]
    fn event_chain_rejects_wrong_before_digest() {
        let mut value = parsed_example();
        let wrong_before = format!("sha256:{}", "9".repeat(64));
        let after = format!("sha256:{}", "e".repeat(64));
        value["entries"][5]["before_digest"] = Value::String(wrong_before.clone());
        value["entries"][5]["event_digest"] = Value::String(
            resource_change_digest(
                "state-shift-selective-invalidation-001",
                6,
                "source-edit-001",
                ChangeKind::SourceChange,
                "repo:src/lib.rs",
                &wrong_before,
                &after,
            )
            .unwrap(),
        );
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_CHANGE_BEFORE_MISMATCH"));
    }

    #[test]
    fn event_digest_is_recomputed_from_the_full_change_payload() {
        let mut value = parsed_example();
        value["entries"][5]["after_digest"] = Value::String(format!("sha256:{}", "9".repeat(64)));
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_EVENT_DIGEST_MISMATCH"));
    }

    #[test]
    fn no_op_resource_change_is_rejected() {
        let mut value = parsed_example();
        let digest = value["entries"][5]["before_digest"].clone();
        value["entries"][5]["after_digest"] = digest;
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_CHANGE_NOOP"));
    }

    #[test]
    fn evidence_cannot_be_recorded_against_stale_generation() {
        let mut value = parsed_example();
        value["entries"][6]["resource_pins"][0]["generation"] = Value::from(0);
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_EVIDENCE_PIN_NOT_CURRENT"));
    }

    #[test]
    fn evidence_dependencies_must_reference_prior_entries() {
        let mut value = parsed_example();
        value["entries"][1]["evidence_dependencies"][0] = Value::String("missing-build".into());
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_EVIDENCE_PARENT_UNKNOWN"));
    }

    #[test]
    fn evidence_resource_pins_must_be_unique() {
        let mut value = parsed_example();
        let duplicate = value["entries"][0]["resource_pins"][0].clone();
        value["entries"][0]["resource_pins"].as_array_mut().unwrap().push(duplicate);
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_RESOURCE_PIN_ID"));
    }

    #[test]
    fn evidence_cannot_pin_an_undeclared_resource() {
        let mut value = parsed_example();
        value["entries"][0]["resource_pins"][0]["resource_id"] = Value::String("unknown:file".into());
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_RESOURCE_PIN_UNKNOWN"));
    }

    #[test]
    fn evidence_with_nonpassing_parent_is_not_usable_for_success() {
        let mut value = parsed_example();
        value["entries"][0]["outcome"] = Value::String("fail".into());
        let report = audit_value(&value);
        assert!(report.valid, "{:?}", report.errors);
        let child = report.evidence.iter().find(|item| item.evidence_id == "tests-before-change").unwrap();
        assert!(!child.usable_for_success);
        assert!(child.blocked_by_evidence_ids.iter().any(|id| id == "build-before-change"));
    }

    #[test]
    fn timeline_sequence_must_be_contiguous() {
        let mut value = parsed_example();
        value["entries"][2]["sequence"] = Value::from(99);
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_SEQUENCE"));
    }

    #[test]
    fn duplicate_resource_ids_are_rejected() {
        let mut value = parsed_example();
        let first = value["initial_resources"][0].clone();
        value["initial_resources"].as_array_mut().unwrap().push(first);
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_RESOURCE_DUPLICATE"));
    }

    #[test]
    fn timeline_entries_reject_unknown_fields() {
        let mut value = parsed_example();
        value["entries"][0]["unmodeled_field"] = Value::String("must not be ignored".into());
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_TIMELINE_JSON"));
    }

    #[test]
    fn malformed_sha256_is_rejected() {
        let mut value = parsed_example();
        value["initial_resources"][0]["digest"] = Value::String("sha256:ABC".into());
        let report = audit_value(&value);
        assert!(!report.valid);
        assert!(has_code(&report, "E_RESOURCE_DIGEST"));
    }
}

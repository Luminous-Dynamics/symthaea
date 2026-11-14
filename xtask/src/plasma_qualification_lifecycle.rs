use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::plasma_claim_scope::{decide, Disposition as ClaimDisposition, ScopeEvidence};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: String,
    cases: Vec<LifecycleCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LifecycleCase {
    id: String,
    evaluation_time: i64,
    generations: Vec<QualificationGeneration>,
    events: Vec<LifecycleEvent>,
    expected_audit_disposition: String,
    expected_claim_states: Vec<ExpectedClaimState>,
    rationale: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualificationGeneration {
    passport_id: String,
    generation_id: String,
    generation_index: u64,
    predecessor_generation_id: Option<String>,
    profile_sha256: String,
    dependency_closure_ref: String,
    dependency_closure_sha256: String,
    decision_receipt_sha256: String,
    transfer_receipt_sha256: String,
    durability_receipt_sha256: String,
    authority_ref: String,
    initial_decision: String,
    valid_from: i64,
    claims: Vec<PassportClaim>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PassportClaim {
    claim_id: String,
    claim_type: String,
    dependency_keys: Vec<String>,
    dependency_closure_ref: String,
    dependency_closure_sha256: String,
    evidence_refs: Vec<String>,
    method_receipts: Vec<ScopeEvidence>,
    review_due_at: Option<i64>,
    valid_until: Option<i64>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct LifecycleEvent {
    event_id: String,
    sequence: u64,
    previous_event_id: Option<String>,
    event_type: String,
    evidence_status: String,
    subject_id: String,
    target_generation_id: Option<String>,
    successor_generation_id: Option<String>,
    affected_dependency_keys: Vec<String>,
    event_time: i64,
    ingested_at: i64,
    effective_from: i64,
    effective_until: Option<i64>,
    source_ref: String,
    source_sha256: String,
    time_source_ref: String,
    authority_ref: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ExpectedClaimState {
    generation_id: String,
    claim_id: String,
    state: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuditDisposition {
    Pass,
    Blocked,
    Rejected,
}

impl AuditDisposition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Blocked => "BLOCKED",
            Self::Rejected => "REJECTED",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaimState {
    Current,
    Suspended,
    Revoked,
    Expired,
    Superseded,
    Blocked,
}

impl ClaimState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Current => "CURRENT",
            Self::Suspended => "SUSPENDED",
            Self::Revoked => "REVOKED",
            Self::Expired => "EXPIRED",
            Self::Superseded => "SUPERSEDED",
            Self::Blocked => "BLOCKED",
        }
    }

    fn priority(self) -> u8 {
        match self {
            Self::Current => 0,
            Self::Suspended => 1,
            Self::Expired => 2,
            Self::Superseded => 3,
            Self::Revoked => 4,
            Self::Blocked => 5,
        }
    }
}

struct AuditResult {
    disposition: AuditDisposition,
    claim_states: Vec<ExpectedClaimState>,
    reason: &'static str,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn unique_nonempty(values: &[String]) -> bool {
    let mut seen = HashSet::new();
    values
        .iter()
        .all(|value| !value.trim().is_empty() && seen.insert(value.as_str()))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn overlap(
    left_start: i64,
    left_end: Option<i64>,
    right_start: i64,
    right_end: Option<i64>,
) -> bool {
    let left_end = left_end.unwrap_or(i64::MAX);
    let right_end = right_end.unwrap_or(i64::MAX);
    left_start <= right_end && right_start <= left_end
}

fn validate_generation_set<'a>(
    case: &'a LifecycleCase,
) -> std::result::Result<HashMap<&'a str, &'a QualificationGeneration>, AuditDisposition> {
    let mut generations = HashMap::new();
    let mut passport_indices = HashSet::new();

    if case.generations.is_empty() {
        return Err(AuditDisposition::Blocked);
    }

    for generation in &case.generations {
        for value in [
            generation.passport_id.as_str(),
            generation.generation_id.as_str(),
            generation.dependency_closure_ref.as_str(),
            generation.authority_ref.as_str(),
        ] {
            if !nonempty(value) {
                return Err(AuditDisposition::Blocked);
            }
        }

        if !is_sha256(&generation.profile_sha256)
            || !is_sha256(&generation.dependency_closure_sha256)
            || !is_sha256(&generation.decision_receipt_sha256)
            || !is_sha256(&generation.transfer_receipt_sha256)
            || !is_sha256(&generation.durability_receipt_sha256)
        {
            return Err(AuditDisposition::Blocked);
        }

        if generation.initial_decision != "QUALIFIED" {
            return Err(AuditDisposition::Rejected);
        }
        if generation.valid_from > case.evaluation_time {
            return Err(AuditDisposition::Blocked);
        }
        if generation.generation_index == 0 {
            return Err(AuditDisposition::Rejected);
        }
        if !passport_indices.insert((generation.passport_id.as_str(), generation.generation_index)) {
            return Err(AuditDisposition::Rejected);
        }
        if generations
            .insert(generation.generation_id.as_str(), generation)
            .is_some()
        {
            return Err(AuditDisposition::Rejected);
        }

        if generation.claims.is_empty() {
            return Err(AuditDisposition::Blocked);
        }
        let mut claim_ids = HashSet::new();
        for claim in &generation.claims {
            if !nonempty(&claim.claim_id)
                || !nonempty(&claim.claim_type)
                || !nonempty(&claim.dependency_closure_ref)
            {
                return Err(AuditDisposition::Blocked);
            }
            if !matches!(
                claim.claim_type.as_str(),
                "method_measurement" | "survival_reliability" | "transfer_qualification" | "process_observation" | "surface_state" | "material_integrity" | "lifecycle_recovery"
            ) {
                return Err(AuditDisposition::Rejected);
            }
            if !claim_ids.insert(claim.claim_id.as_str()) {
                return Err(AuditDisposition::Rejected);
            }
            if !unique_nonempty(&claim.dependency_keys)
                || !unique_nonempty(&claim.evidence_refs)
            {
                return Err(AuditDisposition::Blocked);
            }
            if !is_sha256(&claim.dependency_closure_sha256) {
                return Err(AuditDisposition::Blocked);
            }
            if claim
                .valid_until
                .is_some_and(|end| end < generation.valid_from)
                || claim
                    .review_due_at
                    .is_some_and(|due| due < generation.valid_from)
            {
                return Err(AuditDisposition::Rejected);
            }
            if claim.claim_type == "method_measurement" && claim.method_receipts.is_empty() {
                return Err(AuditDisposition::Blocked);
            }

            let mut receipt_ids = HashSet::new();
            let mut raw_artifacts = HashSet::new();
            for receipt in &claim.method_receipts {
                if !receipt_ids.insert(receipt.id.as_str()) {
                    return Err(AuditDisposition::Rejected);
                }
                let Some(raw_digest) = receipt.raw_data_sha256.as_deref() else {
                    return Err(AuditDisposition::Blocked);
                };
                if !raw_artifacts.insert(raw_digest) {
                    return Err(AuditDisposition::Rejected);
                }
                match decide(receipt).disposition {
                    ClaimDisposition::Supported => {}
                    ClaimDisposition::Blocked => return Err(AuditDisposition::Blocked),
                    ClaimDisposition::Rejected => return Err(AuditDisposition::Rejected),
                }
            }
        }
    }

    for generation in &case.generations {
        match generation.predecessor_generation_id.as_deref() {
            None if generation.generation_index != 1 => {
                return Err(AuditDisposition::Blocked);
            }
            Some(predecessor_id) => {
                let Some(predecessor) = generations.get(predecessor_id).copied() else {
                    return Err(AuditDisposition::Blocked);
                };
                if predecessor.passport_id != generation.passport_id
                    || predecessor.generation_index + 1 != generation.generation_index
                    || predecessor.generation_id == generation.generation_id
                {
                    return Err(AuditDisposition::Rejected);
                }
            }
            _ => {}
        }
    }

    Ok(generations)
}

fn ordered_events<'a>(
    case: &'a LifecycleCase,
    generations: &HashMap<&str, &'a QualificationGeneration>,
) -> std::result::Result<Vec<&'a LifecycleEvent>, AuditDisposition> {
    let mut by_id = HashMap::<&str, &LifecycleEvent>::new();
    for event in &case.events {
        if let Some(previous) = by_id.get(event.event_id.as_str()) {
            if *previous == event {
                continue; // exact replay of the same immutable event is idempotent
            }
            return Err(AuditDisposition::Rejected);
        }
        by_id.insert(event.event_id.as_str(), event);
    }

    let mut events: Vec<&LifecycleEvent> = by_id.values().copied().collect();
    events.sort_by_key(|event| event.sequence);

    let mut seen_sequences = HashSet::new();
    for event in &events {
        if !seen_sequences.insert(event.sequence) {
            return Err(AuditDisposition::Rejected);
        }
    }

    for (index, event) in events.iter().enumerate() {
        if !nonempty(&event.event_id)
            || !nonempty(&event.subject_id)
            || !nonempty(&event.source_ref)
            || !nonempty(&event.time_source_ref)
            || !nonempty(&event.authority_ref)
        {
            return Err(AuditDisposition::Blocked);
        }
        if !is_sha256(&event.source_sha256) {
            return Err(AuditDisposition::Blocked);
        }
        if event.sequence == 0 || event.event_time > event.ingested_at
            || event.ingested_at > case.evaluation_time
            || event.effective_until.is_some_and(|end| end < event.effective_from)
        {
            return Err(AuditDisposition::Rejected);
        }

        if event.affected_dependency_keys.is_empty() && event.target_generation_id.is_none() {
            return Err(AuditDisposition::Blocked);
        }
        if !event.affected_dependency_keys.is_empty()
            && !unique_nonempty(&event.affected_dependency_keys)
        {
            return Err(AuditDisposition::Rejected);
        }
        if let Some(target) = event.target_generation_id.as_deref() {
            if !generations.contains_key(target) {
                return Err(AuditDisposition::Blocked);
            }
        }

        let known_type = matches!(
            event.event_type.as_str(),
            "calibration_expired"
                | "calibration_out_of_tolerance"
                | "sensor_witness_failed"
                | "process_drift"
                | "material_profile_changed"
                | "method_scope_changed"
                | "adverse_field_failure"
                | "evidence_retracted"
                | "review_due"
                | "qualification_superseded"
        );
        if !known_type {
            return Err(AuditDisposition::Blocked);
        }
        if !matches!(event.evidence_status.as_str(), "suspected" | "confirmed") {
            return Err(AuditDisposition::Blocked);
        }

        if event.event_type == "qualification_superseded" {
            if event.evidence_status != "confirmed" {
                return Err(AuditDisposition::Blocked);
            }
            let (Some(target), Some(successor_id)) = (
                event.target_generation_id.as_deref(),
                event.successor_generation_id.as_deref(),
            ) else {
                return Err(AuditDisposition::Blocked);
            };
            let Some(successor) = generations.get(successor_id).copied() else {
                return Err(AuditDisposition::Blocked);
            };
            if successor.predecessor_generation_id.as_deref() != Some(target)
                || successor.valid_from < event.effective_from
            {
                return Err(AuditDisposition::Rejected);
            }
        } else if event.successor_generation_id.is_some() {
            return Err(AuditDisposition::Rejected);
        }

        if index == 0 {
            if event.sequence != 1 || event.previous_event_id.is_some() {
                return Err(AuditDisposition::Blocked);
            }
        } else {
            let previous = events[index - 1];
            if event.sequence != previous.sequence + 1
                || event.previous_event_id.as_deref() != Some(previous.event_id.as_str())
            {
                return Err(AuditDisposition::Blocked);
            }
        }
    }

    Ok(events)
}

fn event_applies_to_claim(
    event: &LifecycleEvent,
    generation: &QualificationGeneration,
    claim: &PassportClaim,
    evaluation_time: i64,
) -> bool {
    let explicitly_targeted = event.target_generation_id.as_deref()
        == Some(generation.generation_id.as_str());
    let dependency_match = event
        .affected_dependency_keys
        .iter()
        .any(|key| claim.dependency_keys.contains(key));

    if !explicitly_targeted && !dependency_match {
        return false;
    }

    let claim_end = claim
        .valid_until
        .unwrap_or(evaluation_time)
        .min(evaluation_time);
    overlap(
        generation.valid_from,
        Some(claim_end),
        event.effective_from,
        event.effective_until,
    )
}

fn state_for_claim(
    generation: &QualificationGeneration,
    claim: &PassportClaim,
    events: &[&LifecycleEvent],
    generations: &HashMap<&str, &QualificationGeneration>,
    evaluation_time: i64,
) -> ClaimState {
    let mut state = ClaimState::Current;

    if claim.valid_until.is_some_and(|end| end <= evaluation_time)
        || claim.review_due_at.is_some_and(|due| due <= evaluation_time)
    {
        state = ClaimState::Expired;
    }

    for event in events {
        if !event_applies_to_claim(event, generation, claim, evaluation_time) {
            continue;
        }

        let next = match event.event_type.as_str() {
            "evidence_retracted" if event.evidence_status == "confirmed" => {
                ClaimState::Revoked
            }
            "qualification_superseded" => {
                let successor_id = event.successor_generation_id.as_deref().unwrap_or("");
                if generations.contains_key(successor_id) {
                    ClaimState::Superseded
                } else {
                    ClaimState::Blocked
                }
            }
            "review_due" if event.evidence_status == "confirmed" => ClaimState::Expired,
            "calibration_expired"
            | "calibration_out_of_tolerance"
            | "sensor_witness_failed"
            | "process_drift"
            | "material_profile_changed"
            | "method_scope_changed"
            | "adverse_field_failure"
            | "evidence_retracted"
            | "review_due" => ClaimState::Suspended,
            _ => ClaimState::Blocked,
        };

        if next.priority() > state.priority() {
            state = next;
        }
    }

    state
}

fn audit_case(case: &LifecycleCase) -> AuditResult {
    let generations = match validate_generation_set(case) {
        Ok(value) => value,
        Err(disposition) => {
            return AuditResult {
                disposition,
                claim_states: Vec::new(),
                reason: "qualification generation identity, dependency closure or method-qualified claim evidence is incomplete or contradictory",
            };
        }
    };

    let events = match ordered_events(case, &generations) {
        Ok(value) => value,
        Err(disposition) => {
            return AuditResult {
                disposition,
                claim_states: Vec::new(),
                reason: "event stream identity, sequence, time, scope or successor lineage is incomplete or contradictory",
            };
        }
    };

    let mut states = Vec::new();
    for generation in &case.generations {
        for claim in &generation.claims {
            let state = state_for_claim(
                generation,
                claim,
                &events,
                &generations,
                case.evaluation_time,
            );
            if state == ClaimState::Blocked {
                return AuditResult {
                    disposition: AuditDisposition::Blocked,
                    claim_states: Vec::new(),
                    reason: "a lifecycle transition references incomplete successor or dependency evidence",
                };
            }
            states.push(ExpectedClaimState {
                generation_id: generation.generation_id.clone(),
                claim_id: claim.claim_id.clone(),
                state: state.as_str().to_owned(),
            });
        }
    }
    states.sort_by(|a, b| {
        a.generation_id
            .cmp(&b.generation_id)
            .then_with(|| a.claim_id.cmp(&b.claim_id))
    });

    AuditResult {
        disposition: AuditDisposition::Pass,
        claim_states: states,
        reason: "event stream is well-formed and claim states were derived using exact dependency and effective-time scope",
    }
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fixture {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse fixture {}", path.display()))
}

pub fn run(path: &Path) -> Result<()> {
    let fixture = read_fixture(path)?;
    if fixture.schema_version != "plasma-qualification-lifecycle-fixture-v0.1" {
        bail!("unsupported fixture schema version: {}", fixture.schema_version);
    }

    let mut counts = [0usize; 3];
    println!("PLASMA-QUAL-LIFECYCLE S0 — event-reconciled Passport lifecycle audit");
    println!("Claim ceiling: synthetic event identity, dependency scope and state-transition semantics only; no production release or restart authority.");

    for case in &fixture.cases {
        let result = audit_case(case);
        let index = match result.disposition {
            AuditDisposition::Pass => 0,
            AuditDisposition::Blocked => 1,
            AuditDisposition::Rejected => 2,
        };
        counts[index] += 1;
        println!("{}: {} — {}", case.id, result.disposition.as_str(), result.reason);

        if result.disposition.as_str() != case.expected_audit_disposition {
            bail!(
                "case {}: expected audit {}, got {}",
                case.id,
                case.expected_audit_disposition,
                result.disposition.as_str()
            );
        }

        if result.disposition == AuditDisposition::Pass {
            if result.claim_states != case.expected_claim_states {
                bail!("case {}: expected claim states differ from replayed states", case.id);
            }
            for state in &result.claim_states {
                println!("  {} / {} => {}", state.generation_id, state.claim_id, state.state);
            }
        }

        if !nonempty(&case.rationale) {
            bail!("case {} has an empty rationale", case.id);
        }
    }

    println!(
        "Fixture replay: {} cases; PASS={} BLOCKED={} REJECTED={}",
        fixture.cases.len(),
        counts[0],
        counts[1],
        counts[2]
    );
    println!("Historical generations are immutable; recovery requires a new qualified successor generation.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str =
        include_str!("../../data/research/plasma_qualification_lifecycle_fixture_v0_1.json");

    #[test]
    fn frozen_fixture_replays_all_expected_states() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid lifecycle fixture");
        assert_eq!(
            fixture.schema_version,
            "plasma-qualification-lifecycle-fixture-v0.1"
        );
        for case in &fixture.cases {
            let result = audit_case(case);
            assert_eq!(
                result.disposition.as_str(),
                case.expected_audit_disposition,
                "{}",
                case.id
            );
            if result.disposition == AuditDisposition::Pass {
                assert_eq!(result.claim_states, case.expected_claim_states, "{}", case.id);
            }
        }
    }

    #[test]
    fn exact_duplicate_event_replay_is_idempotent() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid lifecycle fixture");
        let case = fixture.cases.iter().find(|case| case.id == "exact-event-replay-is-idempotent").expect("case exists");
        assert_eq!(audit_case(case).disposition, AuditDisposition::Pass);
    }

    #[test]
    fn same_event_id_with_different_digest_is_rejected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid lifecycle fixture");
        let case = fixture.cases.iter().find(|case| case.id == "event-id-different-digest-rejected").expect("case exists");
        assert_eq!(audit_case(case).disposition, AuditDisposition::Rejected);
    }

    #[test]
    fn calibration_event_only_suspends_dependent_claims() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid lifecycle fixture");
        let case = fixture.cases.iter().find(|case| case.id == "calibration-event-is-claim-scoped").expect("case exists");
        let result = audit_case(case);
        assert_eq!(result.disposition, AuditDisposition::Pass);
        assert!(result.claim_states.iter().any(|s| s.claim_id == "bond_strength" && s.state == "SUSPENDED"));
        assert!(result.claim_states.iter().any(|s| s.claim_id == "wettability" && s.state == "CURRENT"));
    }

    #[test]
    fn successor_generation_supersedes_without_rewriting_history() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid lifecycle fixture");
        let case = fixture.cases.iter().find(|case| case.id == "new-generation-supersedes-predecessor").expect("case exists");
        let result = audit_case(case);
        assert_eq!(result.disposition, AuditDisposition::Pass);
        assert!(result.claim_states.iter().any(|s| s.generation_id == "GEN-1" && s.state == "SUPERSEDED"));
        assert!(result.claim_states.iter().any(|s| s.generation_id == "GEN-2" && s.state == "CURRENT"));
    }
}

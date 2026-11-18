// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightshell-specific claim impact and requalification planning.
//!
//! Consumes Mycelix serialized lifecycle events and owner-authored impact
//! policies, then maps changed garment entities to zone qualification claims.
//! This is not a service ledger, maintenance system, test scheduler, return-to-
//! service authority, product passport or safety approval.

use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChangeImpactManifest {
    pub schema_version: String,
    pub decision_id: String,
    pub state: String,
    pub predecessor: ConfigurationGeneration,
    pub successor: ConfigurationGeneration,
    pub change_event: ChangeEvent,
    pub unaffected_claim_policy: UnaffectedClaimPolicy,
    pub claims: Vec<QualificationClaim>,
    pub impact_rules: Vec<ImpactRule>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigurationGeneration {
    pub generation_ref: String,
    pub article_ref: String,
    pub robot_article_revision: String,
    pub geometry_digest: String,
    pub topology_digest: String,
    pub route_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChangeEvent {
    pub event_id: String,
    pub event_ref: String,
    pub event_kind: String,
    pub event_origin: String,
    pub authority_ref: Option<String>,
    pub event_evidence_ref: Option<String>,
    pub predecessor_generation_ref: String,
    pub successor_generation_ref: String,
    pub treatment_state: String,
    pub changed_entities: Vec<ChangedEntity>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChangedEntity {
    pub entity_kind: String,
    pub predecessor_entity_ref: Option<String>,
    pub successor_entity_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimDependency {
    pub entity_kind: String,
    pub entity_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationClaim {
    pub claim_id: String,
    pub claim_family: String,
    pub evidence_generation_ref: String,
    pub evidence_origin: String,
    pub evidence_ref: String,
    pub dependencies: Vec<ClaimDependency>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnaffectedClaimPolicy {
    pub disposition: String,
    pub owner_ref: String,
    pub policy_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImpactRule {
    pub rule_id: String,
    pub event_kind: String,
    pub claim_family: String,
    pub affected_entity_kinds: Vec<String>,
    pub disposition: String,
    pub owner_ref: String,
    pub policy_ref: String,
    pub review_record_ref: Option<String>,
    pub test_plan_ref: Option<String>,
    pub method_selection_ref: Option<String>,
    pub fixture_revision_ref: Option<String>,
    pub acceptance_rule_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub canonical_event_authority_is_external: bool,
    pub historical_evidence_is_immutable: bool,
    pub unaffected_claims_need_explicit_policy: bool,
    pub missing_dependency_mapping_is_not_evaluated: bool,
    pub retest_targets_successor_generation: bool,
    pub no_return_to_service_authority: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimImpactAssessment {
    pub claim_id: String,
    pub disposition: String,
    pub matched_rule_ref: Option<String>,
    pub affected_dependency_refs: Vec<String>,
    pub target_generation_ref: String,
    pub review_owner_ref: Option<String>,
    pub review_record_ref: Option<String>,
    pub test_plan_ref: Option<String>,
    pub method_selection_ref: Option<String>,
    pub fixture_revision_ref: Option<String>,
    pub acceptance_rule_ref: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub claim_count: usize,
    pub changed_entity_count: usize,
    pub preserved_count: usize,
    pub review_required_count: usize,
    pub retest_required_count: usize,
    pub superseded_count: usize,
    pub not_evaluated_count: usize,
    pub assessments: Vec<ClaimImpactAssessment>,
    pub impact_plan_admitted: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// An admitted impact plan routes claims; it does not perform repair,
    /// execute testing or authorize the robot to return to service.
    pub fn can_issue_requalification_plan(&self) -> bool {
        self.is_valid() && self.impact_plan_admitted
    }

    pub fn claims_requiring_retest(&self) -> Vec<&str> {
        self.assessments.iter()
            .filter(|item| item.disposition == "retest_required")
            .map(|item| item.claim_id.as_str())
            .collect()
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: ChangeImpactManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl ChangeImpactManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut claim_ids = HashSet::new();
        let mut rule_ids = HashSet::new();
        let mut changed_entity_keys = HashSet::new();

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell change-impact schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.decision_id) {
            push(&mut findings, "decision_identity_missing", "decision_id",
                "Change-impact decision identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "impact_assessed") {
            push(&mut findings, "invalid_impact_state", "state",
                "Impact state must be proposed or impact_assessed; no service-return or release state exists.");
        }

        validate_generation(&self.predecessor, "predecessor", &mut findings);
        validate_generation(&self.successor, "successor", &mut findings);
        if self.predecessor.generation_ref == self.successor.generation_ref {
            push(&mut findings, "configuration_generation_not_advanced", "predecessor/successor.generation_ref",
                "A service/change event must create a distinct successor configuration generation.");
        }
        if is_blank(&self.change_event.event_id) || is_blank(&self.change_event.event_ref) {
            push(&mut findings, "change_event_identity_missing", "change_event",
                "Change event identity and canonical external event reference are mandatory.");
        }
        if !is_known_event_kind(&self.change_event.event_kind) {
            push(&mut findings, "unknown_change_event_kind", "change_event.event_kind",
                "Unknown Lightshell change-event class.");
        }
        if !matches!(self.change_event.event_origin.as_str(), "physical_event" | "synthetic_fixture" | "supplier_declaration" | "model_inferred") {
            push(&mut findings, "unknown_change_event_origin", "change_event.event_origin",
                "Unknown change event origin.");
        }
        if self.change_event.predecessor_generation_ref != self.predecessor.generation_ref
            || self.change_event.successor_generation_ref != self.successor.generation_ref
        {
            push(&mut findings, "change_event_generation_mismatch", "change_event.predecessor_generation_ref/successor_generation_ref",
                "Change event predecessor/successor identities must match the two bound configuration generations.");
        }
        if !matches!(self.change_event.treatment_state.as_str(),
            "as_manufactured" | "after_cleaning" | "after_flex" | "after_mission_replay" | "after_repair" | "after_replacement")
        {
            push(&mut findings, "invalid_change_treatment_state", "change_event.treatment_state",
                "Unknown post-change treatment state.");
        }
        if self.state == "impact_assessed" && self.change_event.event_origin == "physical_event" {
            for (field, value) in [
                ("authority_ref", self.change_event.authority_ref.as_deref()),
                ("event_evidence_ref", self.change_event.event_evidence_ref.as_deref()),
            ] {
                if value.is_none_or(is_blank) {
                    push(&mut findings, "physical_event_provenance_missing", format!("change_event.{field}"),
                        "An assessed physical event must bind its external authority and event evidence.");
                }
            }
        }
        if self.state == "impact_assessed"
            && self.change_event.event_origin != "physical_event"
        {
            push(&mut findings, "nonphysical_event_promoted", "change_event.event_origin",
                "An impact-assessed plan for a real article cannot be promoted from synthetic, supplier-declared or model-inferred events.");
        }
        if self.change_event.changed_entities.is_empty() {
            push(&mut findings, "change_event_without_affected_entities", "change_event.changed_entities",
                "A change event must identify at least one affected entity; unknown scope is not a blanket preserve.");
        }
        if let Some(required_kind) = required_entity_kind_for_event(&self.change_event.event_kind) {
            if !self.change_event.changed_entities.iter().any(|entity| entity.entity_kind == required_kind) {
                push(&mut findings, "change_event_entity_kind_mismatch", "change_event.changed_entities",
                    format!("Event kind {} requires at least one changed entity of kind {}.", self.change_event.event_kind, required_kind));
            }
        }

        for (index, entity) in self.change_event.changed_entities.iter().enumerate() {
            let path = format!("change_event.changed_entities[{index}]");
            if !is_known_entity_kind(&entity.entity_kind) {
                push(&mut findings, "unknown_changed_entity_kind", format!("{path}.entity_kind"),
                    "Unknown Lightshell entity kind.");
            }
            if is_blank_option(&entity.predecessor_entity_ref) && is_blank_option(&entity.successor_entity_ref) {
                push(&mut findings, "changed_entity_identity_missing", &path,
                    "A changed entity requires at least a predecessor or successor identity.");
            }
            let replacement_requires_new_identity = matches!(
                self.change_event.event_kind.as_str(),
                "panel_replacement" | "seam_replacement" | "attachment_replacement" |
                "sensor_replacement" | "material_lot_change" | "process_revision_change" |
                "robot_geometry_change" | "joint_revision_change"
            ) && entity.entity_kind == required_entity_kind_for_event(&self.change_event.event_kind).unwrap_or("");
            if replacement_requires_new_identity {
                match (entity.predecessor_entity_ref.as_deref(), entity.successor_entity_ref.as_deref()) {
                    (Some(previous), Some(next)) if previous != next => {}
                    _ => push(&mut findings, "replacement_identity_not_advanced", &path,
                        "Replacement/change of an identity-bearing entity requires distinct predecessor and successor entity references."),
                }
            }
            if let Some(previous) = entity.predecessor_entity_ref.as_deref() {
                if is_blank(previous) {
                    push(&mut findings, "blank_predecessor_entity_ref", format!("{path}.predecessor_entity_ref"),
                        "Predecessor entity identity must be non-empty.");
                } else if !changed_entity_keys.insert((entity.entity_kind.as_str(), previous)) {
                    push(&mut findings, "duplicate_changed_entity", &path,
                        "Predecessor entity references must be unique within a change event.");
                }
            }
            if let Some(next) = entity.successor_entity_ref.as_deref() {
                if is_blank(next) {
                    push(&mut findings, "blank_successor_entity_ref", format!("{path}.successor_entity_ref"),
                        "Successor entity identity must be non-empty.");
                }
            }
        }

        if self.claims.is_empty() {
            push(&mut findings, "no_claims_to_assess", "claims",
                "At least one predecessor-generation claim must be assessed.");
        }
        for (index, claim) in self.claims.iter().enumerate() {
            let path = format!("claims[{index}]");
            if is_blank(&claim.claim_id) || !claim_ids.insert(claim.claim_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_claim_id", format!("{path}.claim_id"),
                    "Qualification claim IDs must be non-empty and unique.");
            }
            if !is_known_claim_family(&claim.claim_family) {
                push(&mut findings, "unknown_claim_family", format!("{path}.claim_family"),
                    "Unknown Lightshell qualification claim family.");
            }
            if is_blank(&claim.evidence_generation_ref) || is_blank(&claim.evidence_ref) {
                push(&mut findings, "claim_evidence_identity_missing", &path,
                    "Claim must bind exact historical evidence and its configuration generation.");
            }
            if claim.evidence_generation_ref != self.predecessor.generation_ref {
                push(&mut findings, "claim_not_bound_to_predecessor_generation", format!("{path}.evidence_generation_ref"),
                    "This P0 maps claims from the predecessor generation only; successor claims require newly generated evidence.");
            }
            if !matches!(claim.evidence_origin.as_str(), "physical_measurement" | "calibrated_model" | "supplier_declaration" | "simulation" | "synthetic_fixture") {
                push(&mut findings, "unknown_claim_evidence_origin", format!("{path}.evidence_origin"),
                    "Unknown historical claim evidence origin.");
            }
            if claim.dependencies.is_empty() {
                push(&mut findings, "claim_dependency_scope_missing", format!("{path}.dependencies"),
                    "Claim dependency scope must be explicit; an empty dependency list cannot be treated as unaffected.");
            }
            let mut dependencies = HashSet::new();
            for (dep_index, dependency) in claim.dependencies.iter().enumerate() {
                let dep_path = format!("{path}.dependencies[{dep_index}]");
                if !is_known_entity_kind(&dependency.entity_kind) || is_blank(&dependency.entity_ref) {
                    push(&mut findings, "invalid_claim_dependency", &dep_path,
                        "Claim dependency must bind a known entity kind and non-empty exact entity reference.");
                }
                if !dependencies.insert((dependency.entity_kind.as_str(), dependency.entity_ref.as_str())) {
                    push(&mut findings, "duplicate_claim_dependency", &dep_path,
                        "Claim dependency identities must be unique.");
                }
            }
        }

        for (index, rule) in self.impact_rules.iter().enumerate() {
            let path = format!("impact_rules[{index}]");
            if is_blank(&rule.rule_id) || !rule_ids.insert(rule.rule_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_rule_id", format!("{path}.rule_id"),
                    "Impact rule IDs must be non-empty and unique.");
            }
            if !is_known_event_kind(&rule.event_kind) || !is_known_claim_family(&rule.claim_family) {
                push(&mut findings, "invalid_impact_rule_scope", &path,
                    "Impact rule event kind and claim family must use the known Lightshell vocabulary.");
            }
            if rule.affected_entity_kinds.is_empty() {
                push(&mut findings, "impact_rule_entity_scope_missing", format!("{path}.affected_entity_kinds"),
                    "Impact rule must name the entity kinds that trigger it.");
            }
            for kind in &rule.affected_entity_kinds {
                if !is_known_entity_kind(kind) {
                    push(&mut findings, "unknown_rule_entity_kind", format!("{path}.affected_entity_kinds"),
                        "Impact rule contains an unknown entity kind.");
                }
            }
            if !matches!(rule.disposition.as_str(), "review_required" | "retest_required" | "superseded") {
                push(&mut findings, "invalid_impact_rule_disposition", format!("{path}.disposition"),
                    "Affected-claim rules may require review, retest or supersession; preserve is only allowed by the unaffected-claim policy.");
            }
            if is_blank(&rule.owner_ref) || is_blank(&rule.policy_ref) {
                push(&mut findings, "impact_rule_owner_policy_missing", &path,
                    "Every change-impact disposition must bind its responsible engineering owner and policy revision.");
            }
            if rule.disposition == "review_required" && is_blank_option(&rule.review_record_ref) {
                push(&mut findings, "review_rule_record_slot_missing", format!("{path}.review_record_ref"),
                    "Review-required rules must bind an owner review record slot/reference.");
            }
            if rule.disposition == "retest_required" {
                for (field, value) in [
                    ("test_plan_ref", rule.test_plan_ref.as_deref()),
                    ("method_selection_ref", rule.method_selection_ref.as_deref()),
                    ("fixture_revision_ref", rule.fixture_revision_ref.as_deref()),
                    ("acceptance_rule_ref", rule.acceptance_rule_ref.as_deref()),
                ] {
                    if value.is_none_or(is_blank) {
                        push(&mut findings, "retest_rule_incomplete", format!("{path}.{field}"),
                            "Retest-required rules must bind test plan, applicable method, fixture revision and acceptance rule.");
                    }
                }
            }
        }

        if !matches!(self.unaffected_claim_policy.disposition.as_str(), "preserved" | "not_evaluated")
            || is_blank(&self.unaffected_claim_policy.owner_ref)
            || is_blank(&self.unaffected_claim_policy.policy_ref)
        {
            push(&mut findings, "unaffected_claim_policy_incomplete", "unaffected_claim_policy",
                "Claims with no changed dependency may only be preserved or left not_evaluated under an explicit owner policy.");
        }
        if !self.evidence_policy.canonical_event_authority_is_external
            || !self.evidence_policy.historical_evidence_is_immutable
            || !self.evidence_policy.unaffected_claims_need_explicit_policy
            || !self.evidence_policy.missing_dependency_mapping_is_not_evaluated
            || !self.evidence_policy.retest_targets_successor_generation
            || !self.evidence_policy.no_return_to_service_authority
        {
            push(&mut findings, "unsafe_change_impact_policy", "evidence_policy",
                "External event authority, immutable history, explicit preservation policy, successor retest target and no-service-authority boundaries are mandatory.");
        }

        let mut assessments = Vec::new();
        for claim in &self.claims {
            let mut affected_dependency_refs = Vec::new();
            let mut affected_kinds = HashSet::new();
            for dependency in &claim.dependencies {
                let intersects = self.change_event.changed_entities.iter().any(|changed|
                    changed.entity_kind == dependency.entity_kind
                        && changed.predecessor_entity_ref.as_deref() == Some(dependency.entity_ref.as_str()));
                if intersects {
                    affected_dependency_refs.push(format!("{}:{}", dependency.entity_kind, dependency.entity_ref));
                    affected_kinds.insert(dependency.entity_kind.as_str());
                }
            }
            affected_dependency_refs.sort();

            let mut matched_rules: Vec<&ImpactRule> = self.impact_rules.iter()
                .filter(|rule| {
                    rule.event_kind == self.change_event.event_kind
                        && rule.claim_family == claim.claim_family
                        && rule.affected_entity_kinds.iter().any(|kind| affected_kinds.contains(kind.as_str()))
                })
                .collect();

            let mut assessment = ClaimImpactAssessment {
                claim_id: claim.claim_id.clone(),
                disposition: "not_evaluated".to_string(),
                matched_rule_ref: None,
                affected_dependency_refs,
                target_generation_ref: self.successor.generation_ref.clone(),
                review_owner_ref: None,
                review_record_ref: None,
                test_plan_ref: None,
                method_selection_ref: None,
                fixture_revision_ref: None,
                acceptance_rule_ref: None,
            };

            if assessment.affected_dependency_refs.is_empty() {
                assessment.disposition = self.unaffected_claim_policy.disposition.clone();
                if assessment.disposition == "preserved" {
                    assessment.matched_rule_ref = Some(self.unaffected_claim_policy.policy_ref.clone());
                    assessment.review_owner_ref = Some(self.unaffected_claim_policy.owner_ref.clone());
                }
            } else if matched_rules.is_empty() {
                assessment.disposition = "not_evaluated".to_string();
            } else if matched_rules.len() > 1 {
                push(&mut findings, "conflicting_impact_rules", format!("claims.{}.impact_rules", claim.claim_id),
                    "More than one rule matches this event, claim family and changed dependency scope.");
                assessment.disposition = "not_evaluated".to_string();
            } else {
                let rule = matched_rules.remove(0);
                assessment.disposition = rule.disposition.clone();
                assessment.matched_rule_ref = Some(rule.rule_id.clone());
                assessment.review_owner_ref = Some(rule.owner_ref.clone());
                assessment.review_record_ref = rule.review_record_ref.clone();
                assessment.test_plan_ref = rule.test_plan_ref.clone();
                assessment.method_selection_ref = rule.method_selection_ref.clone();
                assessment.fixture_revision_ref = rule.fixture_revision_ref.clone();
                assessment.acceptance_rule_ref = rule.acceptance_rule_ref.clone();
            }
            assessments.push(assessment);
        }

        let preserved_count = assessments.iter().filter(|item| item.disposition == "preserved").count();
        let review_required_count = assessments.iter().filter(|item| item.disposition == "review_required").count();
        let retest_required_count = assessments.iter().filter(|item| item.disposition == "retest_required").count();
        let superseded_count = assessments.iter().filter(|item| item.disposition == "superseded").count();
        let not_evaluated_count = assessments.iter().filter(|item| item.disposition == "not_evaluated").count();
        let impact_plan_admitted = self.state == "impact_assessed"
            && self.change_event.event_origin == "physical_event"
            && !findings.iter().any(|f| f.code == "conflicting_impact_rules")
            && findings.is_empty();

        ValidationReport {
            findings,
            claim_count: self.claims.len(),
            changed_entity_count: self.change_event.changed_entities.len(),
            preserved_count,
            review_required_count,
            retest_required_count,
            superseded_count,
            not_evaluated_count,
            assessments,
            impact_plan_admitted,
        }
    }
}

fn validate_generation(generation: &ConfigurationGeneration, path: &str, findings: &mut Vec<Finding>) {
    for (field, value) in [
        ("generation_ref", generation.generation_ref.as_str()),
        ("article_ref", generation.article_ref.as_str()),
        ("robot_article_revision", generation.robot_article_revision.as_str()),
    ] {
        if is_blank(value) {
            push(findings, "configuration_generation_identity_missing", format!("{path}.{field}"),
                "Configuration generation requires exact article and robot revision identity.");
        }
    }
    for (field, value) in [
        ("geometry_digest", generation.geometry_digest.as_str()),
        ("topology_digest", generation.topology_digest.as_str()),
        ("route_digest", generation.route_digest.as_str()),
    ] {
        if !is_sha256_digest(value) {
            push(findings, "configuration_generation_digest_invalid", format!("{path}.{field}"),
                "Configuration generation geometry/topology/route identity must bind a SHA-256 digest.");
        }
    }
}

fn required_entity_kind_for_event(value: &str) -> Option<&'static str> {
    match value {
        "panel_replacement" => Some("panel"),
        "seam_replacement" => Some("connection"),
        "attachment_replacement" => Some("connection"),
        "sensor_replacement" | "sensor_recalibration" => Some("sensor_tile"),
        "material_lot_change" => Some("material_lot"),
        "process_revision_change" => Some("process_revision"),
        "robot_geometry_change" => Some("robot_geometry"),
        "joint_revision_change" => Some("robot_joint"),
        _ => None,
    }
}

fn is_known_event_kind(value: &str) -> bool {
    matches!(value,
        "cleaning" | "flex_exposure" | "mission_replay" | "abrasion_exposure" | "impact_event" |
        "repair" | "panel_replacement" | "seam_replacement" | "attachment_replacement" |
        "sensor_replacement" | "sensor_recalibration" | "material_lot_change" |
        "process_revision_change" | "robot_geometry_change" | "joint_revision_change")
}

fn is_known_entity_kind(value: &str) -> bool {
    matches!(value,
        "garment_article" | "component" | "panel" | "connection" | "articulation_feature" |
        "service_feature" | "material_lot" | "process_revision" | "sensor_tile" |
        "sensor_interconnect" | "robot_geometry" | "robot_joint" | "pose_set" | "zone")
}

fn is_known_claim_family(value: &str) -> bool {
    matches!(value,
        "geometry_fit" | "joint_clearance" | "seam_integrity" | "impact_protection" |
        "vibration_isolation" | "airflow" | "thermal" | "dimensional_stability" |
        "cleanability" | "sensor_calibration" | "sensor_continuity" | "attachment_retention" |
        "material_property" | "process_capability" | "mission_profile" | "serviceability")
}

fn push(findings: &mut Vec<Finding>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    findings.push(Finding { code, path: path.into(), message: message.into() });
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_blank_option(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(is_blank)
}

fn is_sha256_digest(value: &str) -> bool {
    value.strip_prefix("sha256:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn base_manifest() -> Value {
        json!({
            "schema_version":"0.1.0",
            "decision_id":"LIGHTSHELL-CHANGE-IMPACT-SYNTHETIC-001",
            "state":"proposed",
            "predecessor":{
                "generation_ref":"GARMENT-GENERATION-01","article_ref":"GARMENT-ARTICLE-01",
                "robot_article_revision":"ROBOT-REV-A",
                "geometry_digest":format!("sha256:{}", "a".repeat(64)),
                "topology_digest":format!("sha256:{}", "b".repeat(64)),
                "route_digest":format!("sha256:{}", "c".repeat(64))
            },
            "successor":{
                "generation_ref":"GARMENT-GENERATION-02","article_ref":"GARMENT-ARTICLE-01",
                "robot_article_revision":"ROBOT-REV-A",
                "geometry_digest":format!("sha256:{}", "d".repeat(64)),
                "topology_digest":format!("sha256:{}", "e".repeat(64)),
                "route_digest":format!("sha256:{}", "f".repeat(64))
            },
            "change_event":{
                "event_id":"REPAIR-EVENT-001","event_ref":"MYCELIX-LIFECYCLE-EVENT-001",
                "event_kind":"panel_replacement","event_origin":"synthetic_fixture",
                "authority_ref":null,"event_evidence_ref":null,
                "predecessor_generation_ref":"GARMENT-GENERATION-01",
                "successor_generation_ref":"GARMENT-GENERATION-02",
                "treatment_state":"after_repair",
                "changed_entities":[
                    {"entity_kind":"panel","predecessor_entity_ref":"FOREARM-PANEL-REV-A","successor_entity_ref":"FOREARM-PANEL-REV-B"},
                    {"entity_kind":"connection","predecessor_entity_ref":"FOREARM-ELBOW-SEAM-REV-A","successor_entity_ref":"FOREARM-ELBOW-SEAM-REV-B"}
                ]
            },
            "unaffected_claim_policy":{
                "disposition":"preserved","owner_ref":"LIGHTSHELL-ENGINEERING-OWNER",
                "policy_ref":"UNaffected-CLAIM-CARRY-FORWARD-REV-A"
            },
            "claims":[
                {
                    "claim_id":"CLAIM-FOREARM-FIT","claim_family":"geometry_fit",
                    "evidence_generation_ref":"GARMENT-GENERATION-01","evidence_origin":"physical_measurement",
                    "evidence_ref":"ASBUILT-METROLOGY-CLAIM-01",
                    "dependencies":[
                        {"entity_kind":"panel","entity_ref":"FOREARM-PANEL-REV-A"},
                        {"entity_kind":"robot_geometry","entity_ref":"ROBOT-GEOMETRY-REV-A"}
                    ]
                },
                {
                    "claim_id":"CLAIM-SEAM-INTEGRITY","claim_family":"seam_integrity",
                    "evidence_generation_ref":"GARMENT-GENERATION-01","evidence_origin":"physical_measurement",
                    "evidence_ref":"SEAM-TEST-CLAIM-01",
                    "dependencies":[
                        {"entity_kind":"connection","entity_ref":"FOREARM-ELBOW-SEAM-REV-A"},
                        {"entity_kind":"panel","entity_ref":"FOREARM-PANEL-REV-A"}
                    ]
                },
                {
                    "claim_id":"CLAIM-SENSOR-CAL","claim_family":"sensor_calibration",
                    "evidence_generation_ref":"GARMENT-GENERATION-01","evidence_origin":"physical_measurement",
                    "evidence_ref":"SENSOR-CALIBRATION-CLAIM-01",
                    "dependencies":[{"entity_kind":"sensor_tile","entity_ref":"SENSOR-TILE-01"}]
                },
                {
                    "claim_id":"CLAIM-MATERIAL-AIRFLOW","claim_family":"airflow",
                    "evidence_generation_ref":"GARMENT-GENERATION-01","evidence_origin":"physical_measurement",
                    "evidence_ref":"COUPON-AIRFLOW-CLAIM-01",
                    "dependencies":[{"entity_kind":"material_lot","entity_ref":"SPACER-LOT-01"}]
                }
            ],
            "impact_rules":[
                {
                    "rule_id":"RULE-PANEL-FIT-RETEST","event_kind":"panel_replacement","claim_family":"geometry_fit",
                    "affected_entity_kinds":["panel"],"disposition":"retest_required",
                    "owner_ref":"LIGHTSHELL-GEOMETRY-OWNER","policy_ref":"PANEL-REPAIR-IMPACT-POLICY-REV-A",
                    "review_record_ref":null,"test_plan_ref":"ASBUILT-POSE-RETEST-PLAN-01",
                    "method_selection_ref":"METROLOGY-METHOD-01","fixture_revision_ref":"FIXTURE-REV-A",
                    "acceptance_rule_ref":"FOREARM-FIT-ACCEPTANCE-01"
                },
                {
                    "rule_id":"RULE-SEAM-REVIEW","event_kind":"panel_replacement","claim_family":"seam_integrity",
                    "affected_entity_kinds":["connection"],"disposition":"review_required",
                    "owner_ref":"LIGHTSHELL-TEXTILE-OWNER","policy_ref":"SEAM-REPAIR-IMPACT-POLICY-REV-A",
                    "review_record_ref":"SEAM-ENGINEERING-REVIEW-SLOT-01","test_plan_ref":null,
                    "method_selection_ref":null,"fixture_revision_ref":null,"acceptance_rule_ref":null
                }
            ],
            "evidence_policy":{
                "canonical_event_authority_is_external":true,
                "historical_evidence_is_immutable":true,
                "unaffected_claims_need_explicit_policy":true,
                "missing_dependency_mapping_is_not_evaluated":true,
                "retest_targets_successor_generation":true,
                "no_return_to_service_authority":true
            }
        })
    }

    #[test]
    fn synthetic_panel_repair_routes_claims_without_promoting_physical_event() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.claim_count, 4);
        assert_eq!(report.retest_required_count, 1);
        assert_eq!(report.review_required_count, 1);
        assert_eq!(report.preserved_count, 2);
        assert!(!report.can_issue_requalification_plan());
        assert_eq!(report.claims_requiring_retest(), vec!["CLAIM-FOREARM-FIT"]);
    }

    #[test]
    fn impact_assessed_physical_event_requires_external_authority_and_event_evidence() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("impact_assessed");
        manifest["change_event"]["event_origin"] = json!("physical_event");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "physical_event_provenance_missing"));
        assert!(!report.can_issue_requalification_plan());
    }

    #[test]
    fn complete_physical_event_can_admit_claim_impact_plan() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("impact_assessed");
        manifest["change_event"]["event_origin"] = json!("physical_event");
        manifest["change_event"]["authority_ref"] = json!("MYCELIX-AUTHORITY-01");
        manifest["change_event"]["event_evidence_ref"] = json!("MYCELIX-PHYSICAL-EVENT-EVIDENCE-01");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.can_issue_requalification_plan());
    }

    #[test]
    fn old_claim_evidence_cannot_be_silently_relabelled_as_successor_generation() {
        let mut manifest = base_manifest();
        manifest["claims"][0]["evidence_generation_ref"] = json!("GARMENT-GENERATION-02");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "claim_not_bound_to_predecessor_generation"));
    }

    #[test]
    fn predecessor_and_successor_generations_must_differ() {
        let mut manifest = base_manifest();
        manifest["successor"]["generation_ref"] = json!("GARMENT-GENERATION-01");
        manifest["change_event"]["successor_generation_ref"] = json!("GARMENT-GENERATION-01");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "configuration_generation_not_advanced"));
    }

    #[test]
    fn missing_impact_rule_for_changed_claim_is_not_evaluated() {
        let mut manifest = base_manifest();
        manifest["impact_rules"].as_array_mut().unwrap().retain(|rule| rule["claim_family"] != "geometry_fit");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        let fit = report.assessments.iter().find(|item| item.claim_id == "CLAIM-FOREARM-FIT").unwrap();
        assert_eq!(fit.disposition, "not_evaluated");
    }

    #[test]
    fn unaffected_claims_require_explicit_owner_policy() {
        let mut manifest = base_manifest();
        manifest["unaffected_claim_policy"]["disposition"] = json!("not_evaluated");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert_eq!(report.preserved_count, 0);
        assert_eq!(report.not_evaluated_count, 2);
    }

    #[test]
    fn retest_rule_must_bind_method_fixture_and_acceptance() {
        let mut manifest = base_manifest();
        manifest["impact_rules"][0]["fixture_revision_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "retest_rule_incomplete"));
    }

    #[test]
    fn retest_is_always_targeted_at_successor_generation() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        let fit = report.assessments.iter().find(|item| item.claim_id == "CLAIM-FOREARM-FIT").unwrap();
        assert_eq!(fit.target_generation_ref, "GARMENT-GENERATION-02");
    }

    #[test]
    fn repair_changes_predecessor_component_identity_and_preserves_history() {
        let mut manifest = base_manifest();
        manifest["change_event"]["changed_entities"][0]["successor_entity_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.claims_requiring_retest(), vec!["CLAIM-FOREARM-FIT"]);
    }

    #[test]
    fn blank_or_duplicate_changed_entities_are_rejected() {
        let mut manifest = base_manifest();
        let duplicate = manifest["change_event"]["changed_entities"][0].clone();
        manifest["change_event"]["changed_entities"].as_array_mut().unwrap().push(duplicate);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_changed_entity"));
    }

    #[test]
    fn empty_claim_dependency_scope_is_not_preserved() {
        let mut manifest = base_manifest();
        manifest["claims"][0]["dependencies"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "claim_dependency_scope_missing"));
    }

    #[test]
    fn changed_sensor_tile_only_routes_sensor_claims() {
        let mut manifest = base_manifest();
        manifest["change_event"]["event_kind"] = json!("sensor_replacement");
        manifest["change_event"]["changed_entities"] = json!([
            {"entity_kind":"sensor_tile","predecessor_entity_ref":"SENSOR-TILE-01","successor_entity_ref":"SENSOR-TILE-02"}
        ]);
        manifest["impact_rules"] = json!([{
            "rule_id":"RULE-SENSOR-RECAL","event_kind":"sensor_replacement","claim_family":"sensor_calibration",
            "affected_entity_kinds":["sensor_tile"],"disposition":"retest_required",
            "owner_ref":"LIGHTSHELL-SENSOR-OWNER","policy_ref":"SENSOR-REPLACEMENT-POLICY-REV-A",
            "review_record_ref":null,"test_plan_ref":"SENSOR-CAL-RETTEST-01",
            "method_selection_ref":"SENSOR-CAL-METHOD-01","fixture_revision_ref":"SENSOR-FIXTURE-REV-A",
            "acceptance_rule_ref":"SENSOR-CAL-ACCEPTANCE-01"
        }]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.retest_required_count, 1);
        assert_eq!(report.assessments.iter().find(|item| item.claim_id == "CLAIM-SENSOR-CAL").unwrap().disposition, "retest_required");
        assert_eq!(report.assessments.iter().find(|item| item.claim_id == "CLAIM-FOREARM-FIT").unwrap().disposition, "preserved");
    }

    #[test]
    fn material_lot_change_can_supersede_only_lot_dependent_property_claims() {
        let mut manifest = base_manifest();
        manifest["change_event"]["event_kind"] = json!("material_lot_change");
        manifest["change_event"]["changed_entities"] = json!([
            {"entity_kind":"material_lot","predecessor_entity_ref":"SPACER-LOT-01","successor_entity_ref":"SPACER-LOT-02"}
        ]);
        manifest["impact_rules"] = json!([{
            "rule_id":"RULE-MATERIAL-LOT-SUPERSEDE","event_kind":"material_lot_change","claim_family":"airflow",
            "affected_entity_kinds":["material_lot"],"disposition":"superseded",
            "owner_ref":"LIGHTSHELL-MATERIAL-OWNER","policy_ref":"MATERIAL-LOT-CHANGE-POLICY-REV-A",
            "review_record_ref":null,"test_plan_ref":null,"method_selection_ref":null,
            "fixture_revision_ref":null,"acceptance_rule_ref":null
        }]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.superseded_count, 1);
    }

    #[test]
    fn conflicting_matching_rules_fail_closed() {
        let mut manifest = base_manifest();
        let duplicate = manifest["impact_rules"][0].clone();
        manifest["impact_rules"].as_array_mut().unwrap().push(duplicate);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "conflicting_impact_rules"));
        assert_eq!(report.assessments.iter().find(|item| item.claim_id == "CLAIM-FOREARM-FIT").unwrap().disposition, "not_evaluated");
    }

    #[test]
    fn panel_replacement_must_bind_panel_predecessor_and_successor() {
        let mut manifest = base_manifest();
        manifest["change_event"]["changed_entities"][0]["successor_entity_ref"] = json!("FOREARM-PANEL-REV-A");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "replacement_identity_not_advanced"));
    }

    #[test]
    fn event_kind_must_match_changed_entity_kind() {
        let mut manifest = base_manifest();
        manifest["change_event"]["event_kind"] = json!("sensor_replacement");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "change_event_entity_kind_mismatch"));
    }

    #[test]
    fn model_inferred_event_cannot_be_promoted_to_assessed_physical_plan() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("impact_assessed");
        manifest["change_event"]["event_origin"] = json!("model_inferred");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "nonphysical_event_promoted"));
    }

    #[test]
    fn unknown_entity_kind_is_rejected() {
        let mut manifest = base_manifest();
        manifest["change_event"]["changed_entities"][0]["entity_kind"] = json!("unknown_magic_entity");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unknown_changed_entity_kind"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["invented_return_to_service"] = json!(true);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}

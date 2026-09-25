use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_manufacturing_process::ProcessDefinitionId;

const HISTORY_DOMAIN: &str = "symthaea-manufacturing-process::capability-history-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ProcessStabilityDispositionV1 {
    NotEvaluated,
    EvidenceInsufficient,
    StableUnderProfile,
    InstabilityDetected,
    MixedOrNonstationary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum YieldEvidenceKindV1 {
    Unknown,
    ObservedConformanceFraction,
    ModelEstimatedConformance,
    QualifiedYieldUnderProfile,
    ProductionYieldObservation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct OutcomeCountsV1 {
    first_pass_conforming: u64,
    reworked_then_conforming: u64,
    nonconforming: u64,
    scrapped: u64,
    aborted: u64,
    missing: u64,
    censored: u64,
}

impl OutcomeCountsV1 {
    fn total(&self) -> Option<u64> {
        [
            self.first_pass_conforming,
            self.reworked_then_conforming,
            self.nonconforming,
            self.scrapped,
            self.aborted,
            self.missing,
            self.censored,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct YieldEvidenceSummaryV1 {
    kind: YieldEvidenceKindV1,
    numerator: Option<u64>,
    denominator: Option<u64>,
    outcome_counts: OutcomeCountsV1,
    summary_method_ref: String,
    statistical_model_ref: Option<String>,
    production_scope_ref: Option<String>,
}

impl YieldEvidenceSummaryV1 {
    fn validate(&self) -> Result<(), &'static str> {
        canonical(&self.summary_method_ref)?;
        if let Some(model) = &self.statistical_model_ref {
            canonical(model)?;
        }
        if let Some(scope) = &self.production_scope_ref {
            canonical(scope)?;
        }
        let total = self.outcome_counts.total().ok_or("outcome count overflow")?;
        if total == 0 {
            return Err("yield evidence requires at least one classified outcome");
        }
        match self.kind {
            YieldEvidenceKindV1::Unknown => {
                if self.numerator.is_some() || self.denominator.is_some() {
                    return Err("unknown yield cannot carry an observed fraction");
                }
            }
            YieldEvidenceKindV1::ObservedConformanceFraction
            | YieldEvidenceKindV1::ProductionYieldObservation => {
                let numerator = self.numerator.ok_or("observed yield requires numerator")?;
                let denominator = self.denominator.ok_or("observed yield requires denominator")?;
                if denominator == 0 || numerator > denominator || denominator > total {
                    return Err("invalid observed yield numerator/denominator");
                }
                if matches!(self.kind, YieldEvidenceKindV1::ProductionYieldObservation)
                    && self.production_scope_ref.is_none()
                {
                    return Err("production yield observation requires production scope");
                }
            }
            YieldEvidenceKindV1::ModelEstimatedConformance => {
                if self.statistical_model_ref.is_none() {
                    return Err("model-estimated conformance requires statistical model ref");
                }
                if self.numerator.is_some() || self.denominator.is_some() {
                    return Err("model estimate must not masquerade as observed fraction");
                }
            }
            YieldEvidenceKindV1::QualifiedYieldUnderProfile => {
                if self.production_scope_ref.is_none() {
                    return Err("qualified yield requires qualification/production scope");
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct CapabilityHistorySubjectV1 {
    process_id: ProcessDefinitionId,
    resource_ref: String,
    capability_profile_ref: String,
    recipe_or_commitment_ref: Option<String>,
    configuration_ref: String,
    material_input_profile_ref: String,
    environment_profile_ref: String,
    measurement_system_profile_ref: String,
    specification_profile_ref: String,
    sampling_plan_ref: String,
    statistical_method_profile_ref: String,
    observation_set_refs: Vec<String>,
    window_start_unix_s: u64,
    window_end_unix_s: u64,
    stability: ProcessStabilityDispositionV1,
    stability_method_ref: Option<String>,
    yield_evidence: YieldEvidenceSummaryV1,
    display_label: Option<String>,
}

impl CapabilityHistorySubjectV1 {
    fn validate(&self) -> Result<(), &'static str> {
        for value in [
            self.process_id.0.as_str(),
            self.resource_ref.as_str(),
            self.capability_profile_ref.as_str(),
            self.configuration_ref.as_str(),
            self.material_input_profile_ref.as_str(),
            self.environment_profile_ref.as_str(),
            self.measurement_system_profile_ref.as_str(),
            self.specification_profile_ref.as_str(),
            self.sampling_plan_ref.as_str(),
            self.statistical_method_profile_ref.as_str(),
        ] {
            canonical(value)?;
        }
        if let Some(recipe) = &self.recipe_or_commitment_ref {
            canonical(recipe)?;
        }
        if self.window_start_unix_s >= self.window_end_unix_s {
            return Err("capability-history time window must be ordered");
        }
        if self.observation_set_refs.is_empty() {
            return Err("capability history requires observation evidence");
        }
        let mut seen = BTreeSet::new();
        for reference in &self.observation_set_refs {
            canonical(reference)?;
            if !seen.insert(reference) {
                return Err("duplicate observation-set reference");
            }
        }
        if let Some(method) = &self.stability_method_ref {
            canonical(method)?;
        }
        if matches!(self.stability, ProcessStabilityDispositionV1::StableUnderProfile)
            && self.stability_method_ref.is_none()
        {
            return Err("stable disposition requires explicit stability method/profile");
        }
        self.yield_evidence.validate()
    }

    fn history_id(&self) -> Result<String, &'static str> {
        self.validate()?;
        let mut observations = self.observation_set_refs.clone();
        observations.sort();
        let mut hasher = blake3::Hasher::new();
        for field in [
            HISTORY_DOMAIN,
            self.process_id.0.as_str(),
            self.resource_ref.as_str(),
            self.capability_profile_ref.as_str(),
            self.recipe_or_commitment_ref.as_deref().unwrap_or("<none>"),
            self.configuration_ref.as_str(),
            self.material_input_profile_ref.as_str(),
            self.environment_profile_ref.as_str(),
            self.measurement_system_profile_ref.as_str(),
            self.specification_profile_ref.as_str(),
            self.sampling_plan_ref.as_str(),
            self.statistical_method_profile_ref.as_str(),
        ] {
            hash_field(&mut hasher, field);
        }
        hasher.update(&self.window_start_unix_s.to_le_bytes());
        hasher.update(&self.window_end_unix_s.to_le_bytes());
        hash_field(&mut hasher, &format!("{:?}", self.stability));
        hash_field(
            &mut hasher,
            self.stability_method_ref.as_deref().unwrap_or("<none>"),
        );
        for observation in observations {
            hash_field(&mut hasher, &observation);
        }
        hash_field(
            &mut hasher,
            &serde_json::to_string(&self.yield_evidence).unwrap(),
        );
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum CapabilityDriftKindV1 {
    CalibrationChanged,
    MaintenanceOrRebuild,
    ToolingChanged,
    ControlRevisionChanged,
    RecipeChanged,
    MaterialOrSupplierChanged,
    EnvironmentShift,
    ProcessDriftDetected,
    MeasurementSystemChanged,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct CapabilityDriftEventV1 {
    prior_history_id: String,
    event_time_unix_s: u64,
    kind: CapabilityDriftKindV1,
    evidence_ref: String,
}

impl CapabilityDriftEventV1 {
    fn validate(&self) -> Result<(), &'static str> {
        canonical(&self.prior_history_id)?;
        canonical(&self.evidence_ref)?;
        if self.event_time_unix_s == 0 {
            return Err("drift event requires nonzero time");
        }
        Ok(())
    }
}

fn canonical(value: &str) -> Result<(), &'static str> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err("non-canonical ref");
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn fixture() -> CapabilityHistorySubjectV1 {
    CapabilityHistorySubjectV1 {
        process_id: ProcessDefinitionId("process:precision-milling-v1".into()),
        resource_ref: "eng-catalog:resource:mill-a".into(),
        capability_profile_ref: "mfg:capability:mill-a-rev4".into(),
        recipe_or_commitment_ref: Some("mfg:recipe-commitment:abc".into()),
        configuration_ref: "mfg:configuration:fixture-7-toolset-2".into(),
        material_input_profile_ref: "materials:state:al6061-t6-lot-profile".into(),
        environment_profile_ref: "field:environment:shop-profile-2026q3".into(),
        measurement_system_profile_ref: "field:msa:cmm-17-cal-2026-09".into(),
        specification_profile_ref: "design:spec:feature-envelope-r7".into(),
        sampling_plan_ref: "quality:sampling:subgroup-v2".into(),
        statistical_method_profile_ref: "statistics:spc:method-profile-v1".into(),
        observation_set_refs: vec![
            "field:observations:batch-001".into(),
            "field:observations:batch-002".into(),
        ],
        window_start_unix_s: 1_000,
        window_end_unix_s: 2_000,
        stability: ProcessStabilityDispositionV1::StableUnderProfile,
        stability_method_ref: Some("statistics:stability:profile-v1".into()),
        yield_evidence: YieldEvidenceSummaryV1 {
            kind: YieldEvidenceKindV1::ObservedConformanceFraction,
            numerator: Some(94),
            denominator: Some(100),
            outcome_counts: OutcomeCountsV1 {
                first_pass_conforming: 90,
                reworked_then_conforming: 4,
                nonconforming: 2,
                scrapped: 2,
                aborted: 1,
                missing: 1,
                censored: 0,
            },
            summary_method_ref: "statistics:summary:observed-fraction-v1".into(),
            statistical_model_ref: None,
            production_scope_ref: None,
        },
        display_label: Some("Mill A Q3 capability history".into()),
    }
}

#[test]
fn display_label_does_not_change_history_identity() {
    let a = fixture();
    let mut b = a.clone();
    b.display_label = Some("renamed UI label".into());
    assert_eq!(a.history_id().unwrap(), b.history_id().unwrap());
}

#[test]
fn engineering_and_measurement_context_changes_history_identity() {
    let base = fixture();
    let mutations: [fn(&mut CapabilityHistorySubjectV1); 4] = [
        |h| h.resource_ref = "eng-catalog:resource:mill-b".into(),
        |h| h.measurement_system_profile_ref = "field:msa:cmm-18".into(),
        |h| h.specification_profile_ref = "design:spec:feature-envelope-r8".into(),
        |h| h.configuration_ref = "mfg:configuration:fixture-8".into(),
    ];
    for mutate in mutations {
        let mut changed = base.clone();
        mutate(&mut changed);
        assert_ne!(base.history_id().unwrap(), changed.history_id().unwrap());
    }
}

#[test]
fn stable_disposition_requires_explicit_stability_method() {
    let mut history = fixture();
    history.stability_method_ref = None;
    assert!(history.validate().is_err());
}

#[test]
fn observed_yield_preserves_failures_and_censoring() {
    let history = fixture();
    let counts = &history.yield_evidence.outcome_counts;
    assert_eq!(counts.total(), Some(100));
    assert_eq!(counts.scrapped, 2);
    assert_eq!(counts.aborted, 1);
    assert_eq!(counts.missing, 1);
    assert_eq!(history.yield_evidence.numerator, Some(94));
    assert_eq!(history.yield_evidence.denominator, Some(100));
}

#[test]
fn invalid_or_zero_denominator_rejects() {
    let mut history = fixture();
    history.yield_evidence.denominator = Some(0);
    assert!(history.validate().is_err());
}

#[test]
fn model_estimate_cannot_masquerade_as_observed_fraction() {
    let mut history = fixture();
    history.yield_evidence.kind = YieldEvidenceKindV1::ModelEstimatedConformance;
    history.yield_evidence.statistical_model_ref = Some("statistics:model:normal-fit-v1".into());
    assert!(history.validate().is_err());

    history.yield_evidence.numerator = None;
    history.yield_evidence.denominator = None;
    assert!(history.validate().is_ok());
}

#[test]
fn production_yield_requires_explicit_scope() {
    let mut history = fixture();
    history.yield_evidence.kind = YieldEvidenceKindV1::ProductionYieldObservation;
    assert!(history.validate().is_err());
    history.yield_evidence.production_scope_ref =
        Some("mfg:production-scope:line-a-week-39".into());
    assert!(history.validate().is_ok());
}

#[test]
fn duplicate_observation_batches_reject() {
    let mut history = fixture();
    history
        .observation_set_refs
        .push(history.observation_set_refs[0].clone());
    assert!(history.validate().is_err());
}

#[test]
fn drift_event_is_append_only_and_does_not_mutate_prior_history_identity() {
    let history = fixture();
    let before = history.history_id().unwrap();
    let event = CapabilityDriftEventV1 {
        prior_history_id: before.clone(),
        event_time_unix_s: 2_001,
        kind: CapabilityDriftKindV1::ToolingChanged,
        evidence_ref: "mycelix:evidence:tool-change-42".into(),
    };
    assert!(event.validate().is_ok());
    assert_eq!(history.history_id().unwrap(), before);
}

#[test]
fn serde_round_trip_preserves_stability_yield_and_counts() {
    let history = fixture();
    let encoded = serde_json::to_string(&history).unwrap();
    let decoded: CapabilityHistorySubjectV1 = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, history);
    assert_eq!(decoded.history_id().unwrap(), history.history_id().unwrap());
}

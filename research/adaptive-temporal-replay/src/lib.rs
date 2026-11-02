//! Deterministic synthetic regime generation and bitemporal feedback replay.
//!
//! Evaluator-side research substrate. This crate does not implement SAOCP,
//! UP-OCP, a formal conformal guarantee, or a validated Symthaea model.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

pub const SCALE_PPM: i64 = 1_000_000;
pub const REGISTRY_JSON: &str = include_str!("../../../crates/domains/symthaea-capability-closure/fixtures/adaptive_temporal_calibration_benchmark_v0.1.json");

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
const U32_CARDINALITY: u64 = 1u64 << 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayError(pub String);
impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for ReplayError {}
fn invalid(message: impl Into<String>) -> ReplayError { ReplayError(message.into()) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum StreamTag { Residual = 1, Covariate = 2, TailDecision = 3, TailSign = 4, FeedbackDelay = 5 }

/// SplitMix64 as specified by the benchmark registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 { state: u64 }
impl SplitMix64 {
    pub fn new(seed: u64) -> Self { Self { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_M1);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_M2);
        z ^ (z >> 31)
    }
    pub fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in bytes { hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME); }
    hash
}
pub fn derive_substream_seed(seed_id: u64, scenario_id: &str, tag: StreamTag) -> u64 {
    seed_id ^ fnv1a64(scenario_id.as_bytes()) ^ (tag as u64)
}
pub fn map_u32_to_signed_ppm(value: u32) -> i64 {
    let numerator = 2_000_000u64 * u64::from(value);
    (numerator / u64::from(u32::MAX)) as i64 - SCALE_PPM
}

/// Round n/d to nearest integer, with exact halves away from zero.
pub fn div_round_away_from_zero(n: i128, d: i128) -> Result<i128, ReplayError> {
    if d <= 0 { return Err(invalid("rounding denominator must be positive")); }
    let q = n / d;
    let r = n % d;
    if r == 0 { return Ok(q); }
    let twice = r.abs().checked_mul(2).ok_or_else(|| invalid("rounding overflow"))?;
    if twice >= d {
        if n >= 0 { q.checked_add(1).ok_or_else(|| invalid("rounding overflow")) }
        else { q.checked_sub(1).ok_or_else(|| invalid("rounding overflow")) }
    } else { Ok(q) }
}
fn interpolate(start: i64, end: i64, offset: u64, length: u64) -> Result<i64, ReplayError> {
    if length == 0 || offset >= length { return Err(invalid("interpolation offset outside segment")); }
    if length == 1 { return Ok(start); }
    let numerator = (i128::from(end) - i128::from(start)) * i128::from(offset);
    let delta = div_round_away_from_zero(numerator, i128::from(length - 1))?;
    i64::try_from(i128::from(start) + delta).map_err(|_| invalid("interpolated value exceeds i64"))
}
fn add_step(step: u64, delay: u64) -> Result<u64, ReplayError> {
    step.checked_add(delay).ok_or_else(|| invalid("step plus delay overflow"))
}
fn uniform_integer(rng: &mut SplitMix64, minimum: u64, maximum: u64) -> Result<u64, ReplayError> {
    if minimum > maximum { return Err(invalid("uniform integer lower bound exceeds upper bound")); }
    let span = maximum.checked_sub(minimum).and_then(|x| x.checked_add(1))
        .ok_or_else(|| invalid("uniform integer span overflow"))?;
    if span == 1 { return Ok(minimum); }
    let threshold = span.wrapping_neg() % span;
    loop {
        let value = rng.next_u64();
        if value >= threshold {
            return minimum.checked_add(value % span).ok_or_else(|| invalid("sampled delay overflow"));
        }
    }
}
fn tail_event(rng: &mut SplitMix64, probability_ppm: u32) -> bool {
    u64::from(rng.next_u32()) * 1_000_000u64
        < u64::from(probability_ppm) * U32_CARDINALITY
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct FeedbackDelaySpec { pub mode: String, pub minimum_steps: u64, pub maximum_steps: u64 }
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct RegimeSegment {
    pub start_step: u64, pub end_step_exclusive: u64,
    pub location_start_milli: i64, pub location_end_milli: i64,
    pub scale_start_milli: i64, pub scale_end_milli: i64,
    pub tail_contamination_ppm: u32, pub tail_scale_milli: i64,
    pub covariate_slope_start_milli: i64, pub covariate_slope_end_milli: i64,
    pub source_availability_delay_steps: u64,
}
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct LabelRevisionSpec {
    pub target_step: u64, pub initial_available_step: u64,
    pub revision_available_step: u64, pub revision_kind: String,
}
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SourceOutageSpec {
    pub source_id: u64, pub start_step: u64,
    pub end_step_exclusive: u64, pub recovery_delay_steps: u64,
}
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ScenarioSpec {
    pub id: String, pub kind: String, pub length_steps: u64,
    pub preregistered_change_points: Vec<u64>,
    pub feedback_delay: FeedbackDelaySpec,
    pub regime_segments: Vec<RegimeSegment>,
    pub label_revisions: Vec<LabelRevisionSpec>,
    pub source_outages: Vec<SourceOutageSpec>,
    pub required_source_ids: Vec<u64>,
    pub notes: String,
}
impl ScenarioSpec {
    pub fn validate(&self) -> Result<(), ReplayError> {
        if self.id.is_empty() || self.length_steps == 0 { return Err(invalid("scenario ID/length invalid")); }
        if self.required_source_ids.is_empty() { return Err(invalid(format!("{} has no required sources", self.id))); }
        let mut sources = BTreeSet::new();
        for source in &self.required_source_ids {
            if *source == 0 || !sources.insert(*source) { return Err(invalid(format!("{} has zero/duplicate source IDs", self.id))); }
        }
        if self.feedback_delay.minimum_steps > self.feedback_delay.maximum_steps
            || !matches!(self.feedback_delay.mode.as_str(), "constant" | "uniform_integer") {
            return Err(invalid(format!("{} has unsupported/inverted feedback delay", self.id)));
        }
        if self.feedback_delay.mode == "constant" && self.feedback_delay.minimum_steps != self.feedback_delay.maximum_steps {
            return Err(invalid(format!("{} constant delay bounds differ", self.id)));
        }
        if self.regime_segments.is_empty() || self.regime_segments[0].start_step != 0
            || self.regime_segments.last().map(|x| x.end_step_exclusive) != Some(self.length_steps) {
            return Err(invalid(format!("{} segments do not cover [0,length)", self.id)));
        }
        for (i, segment) in self.regime_segments.iter().enumerate() {
            if segment.start_step >= segment.end_step_exclusive || segment.end_step_exclusive > self.length_steps
                || segment.scale_start_milli <= 0 || segment.scale_end_milli <= 0
                || segment.tail_scale_milli <= 0 || segment.tail_contamination_ppm > 1_000_000 {
                return Err(invalid(format!("{} segment {} invalid", self.id, i)));
            }
            if i > 0 && self.regime_segments[i - 1].end_step_exclusive != segment.start_step {
                return Err(invalid(format!("{} segment gap/overlap at {}", self.id, i)));
            }
        }
        let mut previous = 0;
        for point in &self.preregistered_change_points {
            if *point == 0 || *point >= self.length_steps || *point <= previous {
                return Err(invalid(format!("{} change points invalid", self.id)));
            }
            previous = *point;
        }
        let mut revision_targets = BTreeSet::new();
        for revision in &self.label_revisions {
            if !(revision.target_step < revision.initial_available_step
                && revision.initial_available_step < revision.revision_available_step
                && revision.revision_available_step < self.length_steps)
                || !revision_targets.insert(revision.target_step)
                || !matches!(revision.revision_kind.as_str(), "supersede" | "retract") {
                return Err(invalid(format!("{} label revision invalid", self.id)));
            }
        }
        for outage in &self.source_outages {
            if !sources.contains(&outage.source_id) || outage.start_step >= outage.end_step_exclusive
                || outage.end_step_exclusive > self.length_steps {
                return Err(invalid(format!("{} source outage invalid", self.id)));
            }
        }
        Ok(())
    }
}

pub fn load_registry(json: &str) -> Result<(Vec<u64>, Vec<ScenarioSpec>), ReplayError> {
    let root: serde_json::Value = serde_json::from_str(json).map_err(|e| invalid(format!("registry JSON: {e}")))?;
    let generator = root.get("generator_contract").ok_or_else(|| invalid("missing generator_contract"))?;
    if generator.get("prng").and_then(|v| v.as_str()) != Some("SplitMix64")
        || generator.get("substream_seed_derivation").and_then(|v| v.as_str())
            != Some("seed_id XOR FNV1A64(UTF8(scenario_id)) XOR stream_tag_u64") {
        return Err(invalid("registry PRNG/seed derivation does not match implementation"));
    }
    let seeds: Vec<u64> = serde_json::from_value(root.get("seed_ids").cloned().ok_or_else(|| invalid("missing seed_ids"))?)
        .map_err(|e| invalid(format!("seed_ids: {e}")))?;
    if seeds.len() < 5 { return Err(invalid("registry must pin at least five seeds")); }
    let tags = generator.get("stream_tags").ok_or_else(|| invalid("missing stream_tags"))?;
    let actual = ["residual", "covariate", "tail_decision", "tail_sign", "feedback_delay"]
        .map(|name| tags.get(name).and_then(|v| v.as_u64()).unwrap_or(0));
    if actual != [1, 2, 3, 4, 5] { return Err(invalid("stream tags differ from implementation")); }
    let scenarios: Vec<ScenarioSpec> = serde_json::from_value(
        root.get("scenarios").cloned().ok_or_else(|| invalid("missing scenarios"))?
    ).map_err(|e| invalid(format!("scenario list: {e}")))?;
    if scenarios.len() < 10 { return Err(invalid("registry must include at least ten scenarios")); }
    let mut ids = BTreeSet::new();
    for scenario in &scenarios {
        scenario.validate()?;
        if !ids.insert(scenario.id.as_str()) { return Err(invalid(format!("duplicate scenario {}", scenario.id))); }
    }
    Ok((seeds, scenarios))
}
pub fn load_builtin_registry() -> Result<(Vec<u64>, Vec<ScenarioSpec>), ReplayError> {
    load_registry(REGISTRY_JSON)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedOutcome {
    pub decision_id: u64, pub target_step: u64,
    pub covariate_milli: i64, pub residual_milli: i64,
    pub source_available_step: u64, pub feedback_available_step: u64,
}
fn source_ready_step(spec: &ScenarioSpec, source_id: u64, step: u64, base_delay: u64) -> Result<u64, ReplayError> {
    let mut ready = add_step(step, base_delay)?;
    for outage in spec.source_outages.iter().filter(|x| x.source_id == source_id) {
        if step >= outage.start_step && step < outage.end_step_exclusive {
            ready = ready.max(add_step(outage.end_step_exclusive, outage.recovery_delay_steps)?);
        }
    }
    Ok(ready)
}
pub fn generate_outcomes(spec: &ScenarioSpec, seed_id: u64) -> Result<Vec<GeneratedOutcome>, ReplayError> {
    spec.validate()?;
    let mut residual_rng = SplitMix64::new(derive_substream_seed(seed_id, &spec.id, StreamTag::Residual));
    let mut covariate_rng = SplitMix64::new(derive_substream_seed(seed_id, &spec.id, StreamTag::Covariate));
    let mut tail_rng = SplitMix64::new(derive_substream_seed(seed_id, &spec.id, StreamTag::TailDecision));
    let mut sign_rng = SplitMix64::new(derive_substream_seed(seed_id, &spec.id, StreamTag::TailSign));
    let mut delay_rng = SplitMix64::new(derive_substream_seed(seed_id, &spec.id, StreamTag::FeedbackDelay));
    let mut segment_index = 0usize;
    let capacity = usize::try_from(spec.length_steps).map_err(|_| invalid("scenario too large for this platform"))?;
    let mut outcomes = Vec::with_capacity(capacity);
    let revision_by_target: BTreeMap<u64, &LabelRevisionSpec> =
        spec.label_revisions.iter().map(|x| (x.target_step, x)).collect();

    for step in 0..spec.length_steps {
        while segment_index + 1 < spec.regime_segments.len()
            && step >= spec.regime_segments[segment_index].end_step_exclusive { segment_index += 1; }
        let segment = &spec.regime_segments[segment_index];
        let len = segment.end_step_exclusive - segment.start_step;
        let offset = step - segment.start_step;
        let location = interpolate(segment.location_start_milli, segment.location_end_milli, offset, len)?;
        let scale = interpolate(segment.scale_start_milli, segment.scale_end_milli, offset, len)?;
        let slope = interpolate(segment.covariate_slope_start_milli, segment.covariate_slope_end_milli, offset, len)?;
        let z_ppm = map_u32_to_signed_ppm(residual_rng.next_u32());
        let x_ppm = map_u32_to_signed_ppm(covariate_rng.next_u32());
        let stochastic = if tail_event(&mut tail_rng, segment.tail_contamination_ppm) {
            let sign = if sign_rng.next_u64() & 1 == 0 { -1i128 } else { 1i128 };
            sign * i128::from(segment.tail_scale_milli)
        } else {
            div_round_away_from_zero(i128::from(scale) * i128::from(z_ppm), i128::from(SCALE_PPM))?
        };
        let covariate = div_round_away_from_zero(i128::from(slope) * i128::from(x_ppm), i128::from(SCALE_PPM))?;
        let residual_milli = i64::try_from(i128::from(location) + stochastic + covariate)
            .map_err(|_| invalid("generated residual exceeds i64"))?;
        let source_available_step = spec.required_source_ids.iter().try_fold(step, |latest, source| {
            Ok::<u64, ReplayError>(latest.max(source_ready_step(spec, *source, step, segment.source_availability_delay_steps)?))
        })?;
        let sampled_delay = match spec.feedback_delay.mode.as_str() {
            "constant" => spec.feedback_delay.minimum_steps,
            "uniform_integer" => uniform_integer(&mut delay_rng, spec.feedback_delay.minimum_steps, spec.feedback_delay.maximum_steps)?,
            other => return Err(invalid(format!("unsupported delay mode {other}"))),
        };
        let ordinary_feedback = add_step(step, sampled_delay)?.max(source_available_step);
        let feedback_available_step = revision_by_target.get(&step)
            .map(|r| r.initial_available_step).unwrap_or(ordinary_feedback);
        if feedback_available_step < step { return Err(invalid("feedback precedes target event")); }
        outcomes.push(GeneratedOutcome {
            decision_id: step + 1, target_step: step, covariate_milli: x_ppm,
            residual_milli, source_available_step, feedback_available_step,
        });
    }
    Ok(outcomes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionKind { Assert, Supersede, Retract }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedbackEvent {
    pub decision_id: u64, pub target_step: u64, pub revision: u32,
    pub available_at_step: u64, pub kind: RevisionKind, pub label_milli: Option<i64>,
}
pub fn build_feedback_events(spec: &ScenarioSpec, outcomes: &[GeneratedOutcome]) -> Result<Vec<FeedbackEvent>, ReplayError> {
    let revisions: BTreeMap<u64, &LabelRevisionSpec> = spec.label_revisions.iter().map(|x| (x.target_step, x)).collect();
    let mut events = Vec::with_capacity(outcomes.len() + spec.label_revisions.len());
    for outcome in outcomes {
        events.push(FeedbackEvent {
            decision_id: outcome.decision_id, target_step: outcome.target_step, revision: 1,
            available_at_step: outcome.feedback_available_step, kind: RevisionKind::Assert,
            label_milli: Some(outcome.residual_milli),
        });
        if let Some(revision) = revisions.get(&outcome.target_step) {
            let (kind, value) = match revision.revision_kind.as_str() {
                "supersede" => (RevisionKind::Supersede, Some(outcome.residual_milli.checked_neg()
                    .ok_or_else(|| invalid("superseded label negation overflow"))?)),
                "retract" => (RevisionKind::Retract, None),
                other => return Err(invalid(format!("unsupported revision kind {other}"))),
            };
            events.push(FeedbackEvent {
                decision_id: outcome.decision_id, target_step: outcome.target_step, revision: 2,
                available_at_step: revision.revision_available_step, kind, label_milli: value,
            });
        }
    }
    events.sort_by_key(|e| (e.available_at_step, e.decision_id, e.revision));
    Ok(events)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelState {
    NotYetAvailable,
    Active { revision: u32, value_milli: i64 },
    Retracted { revision: u32 },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelRevisionRecord {
    pub decision_id: u64, pub target_step: u64, pub revision: u32,
    pub available_at_step: u64, pub recorded_at_step: u64,
    pub kind: RevisionKind, pub label_milli: Option<i64>,
}
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BitemporalLabelLedger {
    records: Vec<LabelRevisionRecord>,
    latest_by_decision: BTreeMap<u64, LabelRevisionRecord>,
}
impl BitemporalLabelLedger {
    pub fn append(&mut self, record: LabelRevisionRecord) -> Result<(), ReplayError> {
        if record.decision_id == 0 || record.revision == 0 || record.recorded_at_step < record.available_at_step {
            return Err(invalid("label identity/revision invalid or recorded before availability"));
        }
        if let Some(last) = self.records.last() {
            if record.available_at_step < last.available_at_step || record.recorded_at_step < last.recorded_at_step {
                return Err(invalid("ledger append is not in non-decreasing knowledge-time order"));
            }
        }
        match self.latest_by_decision.get(&record.decision_id) {
            None if record.revision == 1 && record.kind == RevisionKind::Assert && record.label_milli.is_some() => {}
            None => return Err(invalid("first label revision must be revision-1 Assert with a value")),
            Some(previous) => {
                if record.revision != previous.revision + 1 || record.target_step != previous.target_step {
                    return Err(invalid("label revision sequence/target is not contiguous"));
                }
                match record.kind {
                    RevisionKind::Supersede if record.label_milli.is_some() => {}
                    RevisionKind::Retract if record.label_milli.is_none() => {}
                    _ => return Err(invalid("later label must be Supersede(value) or Retract")),
                }
            }
        }
        self.latest_by_decision.insert(record.decision_id, record.clone());
        self.records.push(record);
        Ok(())
    }
    pub fn append_feedback_event(&mut self, event: &FeedbackEvent) -> Result<(), ReplayError> {
        self.append(LabelRevisionRecord {
            decision_id: event.decision_id, target_step: event.target_step,
            revision: event.revision, available_at_step: event.available_at_step,
            recorded_at_step: event.available_at_step, kind: event.kind, label_milli: event.label_milli,
        })
    }
    pub fn state_as_known(&self, decision_id: u64, cutoff_step: u64) -> LabelState {
        let latest = self.records.iter()
            .filter(|r| r.decision_id == decision_id && r.available_at_step <= cutoff_step && r.recorded_at_step <= cutoff_step)
            .max_by_key(|r| (r.available_at_step, r.revision));
        match latest {
            None => LabelState::NotYetAvailable,
            Some(r) if r.kind == RevisionKind::Retract => LabelState::Retracted { revision: r.revision },
            Some(r) => LabelState::Active { revision: r.revision, value_milli: r.label_milli.unwrap_or_default() },
        }
    }
    pub fn active_labels_as_known(&self, cutoff_step: u64) -> BTreeMap<u64, i64> {
        let mut result = BTreeMap::new();
        for id in self.latest_by_decision.keys() {
            if let LabelState::Active { value_milli, .. } = self.state_as_known(*id, cutoff_step) {
                result.insert(*id, value_milli);
            }
        }
        result
    }
    pub fn records(&self) -> &[LabelRevisionRecord] { &self.records }
    pub fn verify(&self) -> Result<(), ReplayError> {
        let mut replay = Self::default();
        for record in &self.records { replay.append(record.clone())?; }
        if replay.latest_by_decision != self.latest_by_decision {
            return Err(invalid("latest revision index differs from append-only history"));
        }
        Ok(())
    }
}
pub fn replay_feedback_events(events: &[FeedbackEvent]) -> Result<BitemporalLabelLedger, ReplayError> {
    let mut ledger = BitemporalLabelLedger::default();
    for event in events { ledger.append_feedback_event(event)?; }
    ledger.verify()?;
    Ok(ledger)
}

/// Scalar ACI recursion applied only when feedback is delivered. This is a
/// baseline update, not the fixed-horizon delayed-ACI theorem or SAOCP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AciState { target_alpha_ppm: i64, gamma_ppm: i64, current_alpha_ppm: i64 }
impl AciState {
    pub fn new(target_alpha_ppm: i64, gamma_ppm: i64) -> Result<Self, ReplayError> {
        if !(1..SCALE_PPM).contains(&target_alpha_ppm) || !(1..=SCALE_PPM).contains(&gamma_ppm) {
            return Err(invalid("ACI alpha must be in (0,1) and gamma in (0,1]"));
        }
        Ok(Self { target_alpha_ppm, gamma_ppm, current_alpha_ppm: target_alpha_ppm })
    }
    pub fn current_alpha_ppm(&self) -> i64 { self.current_alpha_ppm }
    pub fn update_on_delivered_feedback(&mut self, miscovered: bool) -> Result<i64, ReplayError> {
        let observed = if miscovered { SCALE_PPM } else { 0 };
        let delta = div_round_away_from_zero(
            i128::from(self.gamma_ppm) * i128::from(self.target_alpha_ppm - observed),
            i128::from(SCALE_PPM),
        )?;
        self.current_alpha_ppm = i64::try_from(i128::from(self.current_alpha_ppm) + delta)
            .map_err(|_| invalid("ACI update overflow"))?.clamp(1, SCALE_PPM - 1);
        Ok(self.current_alpha_ppm)
    }
}

/// Split-conformal order statistic. None means the finite-sample order
/// statistic exceeds the observed sample; callers must not silently use max.
pub fn finite_sample_radius(scores: &[i64], alpha_ppm: i64) -> Result<Option<i64>, ReplayError> {
    if !(1..SCALE_PPM).contains(&alpha_ppm) { return Err(invalid("alpha must be in (0,1)")); }
    if scores.is_empty() { return Ok(None); }
    let n_plus_one = i128::try_from(scores.len()).map_err(|_| invalid("score count exceeds i128"))? + 1;
    let numerator = n_plus_one * i128::from(SCALE_PPM - alpha_ppm);
    let k = (numerator + i128::from(SCALE_PPM) - 1) / i128::from(SCALE_PPM);
    if k > i128::try_from(scores.len()).map_err(|_| invalid("score count exceeds i128"))? { return Ok(None); }
    let mut ordered = scores.to_vec();
    ordered.sort_unstable();
    let index = usize::try_from(k - 1).map_err(|_| invalid("quantile index overflow"))?;
    Ok(ordered.get(index).copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_matches_reference_vector() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    }
    #[test]
    fn fnv1a_matches_known_hello_vector() {
        assert_eq!(fnv1a64(b"hello"), 0xa430_d846_80aa_bd0b);
    }
    #[test]
    fn ppm_mapping_has_inclusive_endpoints() {
        assert_eq!(map_u32_to_signed_ppm(0), -1_000_000);
        assert_eq!(map_u32_to_signed_ppm(u32::MAX), 1_000_000);
    }
    #[test]
    fn rounding_is_symmetric_and_ties_away_from_zero() {
        assert_eq!(div_round_away_from_zero(15, 10).unwrap(), 2);
        assert_eq!(div_round_away_from_zero(-15, 10).unwrap(), -2);
        assert_eq!(div_round_away_from_zero(14, 10).unwrap(), 1);
        assert_eq!(div_round_away_from_zero(-14, 10).unwrap(), -1);
    }
    #[test]
    fn fixture_loads_ten_scenarios_and_five_pinned_seeds() {
        let (seeds, scenarios) = load_builtin_registry().unwrap();
        assert_eq!(seeds, vec![11, 23, 47, 83, 20_261_102]);
        assert_eq!(scenarios.len(), 10);
    }
    #[test]
    fn stream_tags_derive_distinct_substreams() {
        let tags = [StreamTag::Residual, StreamTag::Covariate, StreamTag::TailDecision, StreamTag::TailSign, StreamTag::FeedbackDelay];
        let derived: BTreeSet<_> = tags.iter().map(|t| derive_substream_seed(20_261_102, "abrupt_location_shift", *t)).collect();
        assert_eq!(derived.len(), tags.len());
    }
    #[test]
    fn generation_is_repeatable() {
        let (_, scenarios) = load_builtin_registry().unwrap();
        let spec = scenarios.iter().find(|s| s.id == "abrupt_location_shift").unwrap();
        assert_eq!(generate_outcomes(spec, 20_261_102).unwrap(), generate_outcomes(spec, 20_261_102).unwrap());
    }
    #[test]
    fn all_scenarios_generate_full_streams_and_non_anticipating_feedback() {
        let (_, scenarios) = load_builtin_registry().unwrap();
        for spec in scenarios {
            let outcomes = generate_outcomes(&spec, 47).unwrap();
            assert_eq!(outcomes.len() as u64, spec.length_steps);
            assert!(outcomes.iter().all(|x| x.feedback_available_step >= x.target_step));
            assert!(outcomes.windows(2).all(|w| w[0].target_step < w[1].target_step));
        }
    }
    #[test]
    fn feedback_and_revision_events_are_sorted_by_availability() {
        let (_, scenarios) = load_builtin_registry().unwrap();
        let spec = scenarios.iter().find(|s| s.id == "superseded_outcome_feedback").unwrap();
        let outcomes = generate_outcomes(spec, 83).unwrap();
        let events = build_feedback_events(spec, &outcomes).unwrap();
        assert!(events.windows(2).all(|w| (w[0].available_at_step, w[0].decision_id, w[0].revision)
            <= (w[1].available_at_step, w[1].decision_id, w[1].revision)));
        let target: Vec<_> = events.iter().filter(|e| e.target_step == 3000).collect();
        assert_eq!(target.len(), 2);
        assert_eq!(target[0].available_at_step, 3010);
        assert_eq!(target[1].available_at_step, 3400);
        assert_eq!(target[1].kind, RevisionKind::Supersede);
    }
    #[test]
    fn source_outage_delays_feedback_until_recovery() {
        let (_, scenarios) = load_builtin_registry().unwrap();
        let spec = scenarios.iter().find(|s| s.id == "asynchronous_source_outage").unwrap();
        let outcomes = generate_outcomes(spec, 11).unwrap();
        assert_eq!(outcomes[5_000].source_available_step, 7_040);
        assert!(outcomes[5_000].feedback_available_step >= 7_040);
        assert_eq!(outcomes[7_000].source_available_step, 7_000);
    }
    #[test]
    fn ledger_hides_future_labels_and_preserves_prior_as_known_views() {
        let mut ledger = BitemporalLabelLedger::default();
        assert_eq!(ledger.state_as_known(1, 14), LabelState::NotYetAvailable);
        ledger.append(LabelRevisionRecord {
            decision_id: 1, target_step: 10, revision: 1, available_at_step: 15,
            recorded_at_step: 15, kind: RevisionKind::Assert, label_milli: Some(42),
        }).unwrap();
        assert_eq!(ledger.state_as_known(1, 14), LabelState::NotYetAvailable);
        assert_eq!(ledger.state_as_known(1, 15), LabelState::Active { revision: 1, value_milli: 42 });
        ledger.append(LabelRevisionRecord {
            decision_id: 1, target_step: 10, revision: 2, available_at_step: 20,
            recorded_at_step: 20, kind: RevisionKind::Supersede, label_milli: Some(7),
        }).unwrap();
        assert_eq!(ledger.state_as_known(1, 19), LabelState::Active { revision: 1, value_milli: 42 });
        assert_eq!(ledger.state_as_known(1, 20), LabelState::Active { revision: 2, value_milli: 7 });
        ledger.verify().unwrap();
    }
    #[test]
    fn retract_changes_only_future_as_known_views() {
        let mut ledger = BitemporalLabelLedger::default();
        ledger.append(LabelRevisionRecord {
            decision_id: 9, target_step: 3, revision: 1, available_at_step: 4,
            recorded_at_step: 4, kind: RevisionKind::Assert, label_milli: Some(100),
        }).unwrap();
        ledger.append(LabelRevisionRecord {
            decision_id: 9, target_step: 3, revision: 2, available_at_step: 8,
            recorded_at_step: 8, kind: RevisionKind::Retract, label_milli: None,
        }).unwrap();
        assert_eq!(ledger.state_as_known(9, 7), LabelState::Active { revision: 1, value_milli: 100 });
        assert_eq!(ledger.state_as_known(9, 8), LabelState::Retracted { revision: 2 });
    }
    #[test]
    fn ledger_rejects_out_of_order_or_noncontiguous_revisions() {
        let mut ledger = BitemporalLabelLedger::default();
        assert!(ledger.append(LabelRevisionRecord {
            decision_id: 1, target_step: 1, revision: 1, available_at_step: 4,
            recorded_at_step: 3, kind: RevisionKind::Assert, label_milli: Some(2),
        }).is_err());
        ledger.append(LabelRevisionRecord {
            decision_id: 1, target_step: 1, revision: 1, available_at_step: 4,
            recorded_at_step: 4, kind: RevisionKind::Assert, label_milli: Some(2),
        }).unwrap();
        assert!(ledger.append(LabelRevisionRecord {
            decision_id: 1, target_step: 1, revision: 3, available_at_step: 5,
            recorded_at_step: 5, kind: RevisionKind::Supersede, label_milli: Some(3),
        }).is_err());
    }
    #[test]
    fn delayed_aci_moves_alpha_in_correct_direction_on_feedback() {
        let mut state = AciState::new(100_000, 50_000).unwrap();
        assert_eq!(state.update_on_delivered_feedback(true).unwrap(), 55_000);
        assert_eq!(state.update_on_delivered_feedback(false).unwrap(), 57_250);
    }
    #[test]
    fn finite_sample_quantile_keeps_unbounded_case_explicit() {
        assert_eq!(finite_sample_radius(&[1,2,3,4,5,6,7,8], 100_000).unwrap(), None);
        assert_eq!(finite_sample_radius(&[8,1,7,2,6,3,5,4,9], 100_000).unwrap(), Some(9));
    }
    #[test]
    fn complete_replay_builds_verified_bitemporal_ledger() {
        let (_, scenarios) = load_builtin_registry().unwrap();
        let spec = scenarios.iter().find(|s| s.id == "superseded_outcome_feedback").unwrap();
        let outcomes = generate_outcomes(spec, 2026).unwrap();
        let events = build_feedback_events(spec, &outcomes).unwrap();
        let ledger = replay_feedback_events(&events).unwrap();
        assert_eq!(ledger.records().len(), outcomes.len() + spec.label_revisions.len());
        assert!(matches!(ledger.state_as_known(3001, 3100), LabelState::Active { revision: 1, .. }));
        assert!(matches!(ledger.state_as_known(3001, 3400), LabelState::Active { revision: 2, .. }));
    }
}

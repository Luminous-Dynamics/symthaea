//! Deterministic synthetic regime generation and bitemporal feedback replay.
//!
//! Evaluator-side research substrate. This crate does not implement SAOCP,
//! UP-OCP, a formal conformal guarantee, or a validated Symthaea model.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SCALE_PPM: i64 = 1_000_000;
pub const ALPHA_DENOMINATOR_1E12: i128 = 1_000_000_000_000;
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
    let difference = i128::from(end) - i128::from(start);
    let numerator = difference.checked_mul(i128::from(offset))
        .ok_or_else(|| invalid("interpolation multiplication overflow"))?;
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
            != Some("seed_id XOR FNV1A64(UTF8(scenario_id)) XOR stream_tag_u64")
        || generator.get("distribution_transform").and_then(|v| v.as_str())
            != Some("u32=next_u32(); signed_ppm=floor(2000000*u32/(2^32-1))-1000000; use z_ppm=signed_ppm for residual innovations")
        || generator.get("covariate_stream").and_then(|v| v.as_str())
            != Some("independent SplitMix64 substream with its own committed seed tag; x_ppm uses the identical integer u32-to-signed-PPM mapping; no floating-point conversion")
        || generator.get("residual_equation").and_then(|v| v.as_str())
            != Some("ordinary residual = location_milli + round_away(scale_milli*z_ppm/1000000) + round_away(covariate_slope_milli*x_ppm/1000000); on a tail event replace the scale*z term with +/- tail_scale_milli")
        || generator.get("tail_draw").and_then(|v| v.as_str())
            != Some("tail occurs when next_u32*1000000 < tail_contamination_ppm*2^32; the separate tail_sign stream selects +/- tail_scale_milli")
        || generator.get("feedback_delay_draw").and_then(|v| v.as_str())
            != Some("uniform_integer modes use rejection sampling over the inclusive [minimum_steps,maximum_steps] range") {
        return Err(invalid("registry PRNG, stream, or integer transform contract differs from implementation"));
    }
    let seeds: Vec<u64> = serde_json::from_value(root.get("seed_ids").cloned().ok_or_else(|| invalid("missing seed_ids"))?)
        .map_err(|e| invalid(format!("seed_ids: {e}")))?;
    let mut unique_seeds = BTreeSet::new();
    if seeds.len() < 5 || seeds.iter().any(|seed| !unique_seeds.insert(*seed)) {
        return Err(invalid("registry must pin at least five unique seed IDs"));
    }
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
    pub covariate_ppm: i64, pub residual_milli: i64,
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
        let feedback_available_step = if let Some(revision) = revision_by_target.get(&step) {
            if revision.initial_available_step < ordinary_feedback {
                return Err(invalid("revision schedule backdates initial label before normal/source availability"));
            }
            revision.initial_available_step
        } else {
            ordinary_feedback
        };
        if feedback_available_step < step { return Err(invalid("feedback precedes target event")); }
        outcomes.push(GeneratedOutcome {
            decision_id: step + 1, target_step: step, covariate_ppm: x_ppm,
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
                let expected_revision = previous.revision.checked_add(1)
                    .ok_or_else(|| invalid("label revision counter overflow"))?;
                if record.revision != expected_revision || record.target_step != previous.target_step {
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
    /// Domain-separated digest of the ordered revision history. The digest
    /// binds availability/knowledge times and distinguishes Supersede/Retract.
    pub fn sha256_hex(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"SYMTHAEA-ATCB-BITEMPORAL-LABEL-LEDGER-V1");
        hasher.update((self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            hasher.update(record.decision_id.to_le_bytes());
            hasher.update(record.target_step.to_le_bytes());
            hasher.update(record.revision.to_le_bytes());
            hasher.update(record.available_at_step.to_le_bytes());
            hasher.update(record.recorded_at_step.to_le_bytes());
            hasher.update([match record.kind {
                RevisionKind::Assert => 0,
                RevisionKind::Supersede => 1,
                RevisionKind::Retract => 2,
            }]);
            match record.label_milli {
                Some(value) => {
                    hasher.update([1]);
                    hasher.update(value.to_le_bytes());
                }
                None => hasher.update([0]),
            }
        }
        let digest = hasher.finalize();
        let mut output = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }
}
pub fn replay_feedback_events(events: &[FeedbackEvent]) -> Result<BitemporalLabelLedger, ReplayError> {
    let mut ledger = BitemporalLabelLedger::default();
    for event in events { ledger.append_feedback_event(event)?; }
    ledger.verify()?;
    Ok(ledger)
}

/// Interpretation of the level produced by the delayed-ACI recursion.
/// The original method allows levels outside [0,1]: signed-residual intervals
/// become all-real when alpha < 0 and empty when alpha > 1. Callers must map
/// these regimes explicitly rather than silently clipping alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AciSetRegime {
    InUnitInterval,
    AllReal,
    Empty,
}

pub fn aci_set_regime(alpha_numerator_1e12: i128) -> AciSetRegime {
    if alpha_numerator_1e12 < 0 { AciSetRegime::AllReal }
    else if alpha_numerator_1e12 > ALPHA_DENOMINATOR_1E12 { AciSetRegime::Empty }
    else { AciSetRegime::InUnitInterval }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingDelayedAciPrediction {
    origin_step: u64,
    target_step: u64,
    phase_index: usize,
    alpha_used_numerator_1e12: i128,
    prediction_set_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayedAciFeedbackReceipt {
    pub target_step: u64,
    pub prediction_origin_step: u64,
    pub phase_index: usize,
    pub alpha_used_ppm: i64,
    pub prediction_set_sha256: [u8; 32],
    pub miscovered: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayedAciStepReceipt {
    pub origin_step: u64,
    pub issued_target_step: u64,
    pub phase_index: usize,
    pub alpha_used_ppm: i64,
    pub updated_alpha_ppm: i64,
    pub prediction_set_sha256: [u8; 32],
    pub consumed_feedback: Option<DelayedAciFeedbackReceipt>,
}

/// Fixed-horizon tau-delayed Adaptive Conformal Inference (tau-DACI).
///
/// One call corresponds to one consecutive forecast origin. At origin t:
/// - if t < tau, no feedback is due and the phase starts at target alpha;
/// - if t >= tau, the caller must supply the miss/cover result for target t;
/// - the queued prediction issued at t-tau is consumed and its stored alpha
///   is updated using alpha_next = alpha_used + gamma * (target_alpha - err);
/// - the resulting phase-specific level is used for the new prediction t+tau.
///
/// This creates tau interleaved ACI recursions. It deliberately rejects missing,
/// variable-delay, or out-of-order feedback: those are different methods and
/// need a different filtration/validity analysis.
///
/// The arithmetic is integer PPM with nearest rounding (ties away from zero).
/// This is a deterministic software approximation to the paper's real-valued
/// recursion; rounding means this implementation alone does not inherit the
/// paper's exact theorem. Alpha is not clipped to [0,1].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayedAciBank {
    horizon_steps: u64,
    target_alpha_ppm: i64,
    gamma_ppm: i64,
    phase_alpha_numerator_1e12: Vec<i128>,
    pending: BTreeMap<u64, PendingDelayedAciPrediction>,
    next_origin_step: u64,
    history: Vec<DelayedAciStepReceipt>,
}

impl DelayedAciBank {
    pub fn new(horizon_steps: u64, target_alpha_ppm: i64, gamma_ppm: i64) -> Result<Self, ReplayError> {
        if horizon_steps == 0 { return Err(invalid("delayed-ACI horizon must be positive")); }
        if horizon_steps > 1_000_000 { return Err(invalid("delayed-ACI horizon exceeds configured cap")); }
        if !(1..SCALE_PPM).contains(&target_alpha_ppm) {
            return Err(invalid("delayed-ACI target alpha must be in (0,1)"));
        }
        if !(1..=SCALE_PPM).contains(&gamma_ppm) {
            return Err(invalid("delayed-ACI gamma must be in (0,1]"));
        }
        let phase_count = usize::try_from(horizon_steps)
            .map_err(|_| invalid("delayed-ACI horizon does not fit usize"))?;
        Ok(Self {
            horizon_steps,
            target_alpha_ppm,
            gamma_ppm,
            phase_alpha_numerator_1e12: vec![i128::from(target_alpha_ppm) * i128::from(SCALE_PPM); phase_count],
            pending: BTreeMap::new(),
            next_origin_step: 0,
            history: Vec::new(),
        })
    }

    pub fn horizon_steps(&self) -> u64 { self.horizon_steps }
    pub fn next_origin_step(&self) -> u64 { self.next_origin_step }
    pub fn pending_count(&self) -> usize { self.pending.len() }
    pub fn phase_alpha_numerator_1e12(&self, phase_index: usize) -> Option<i128> {
        self.phase_alpha_numerator_1e12.get(phase_index).copied()
    }

    pub fn phase_alpha_ppm_rounded(&self, phase_index: usize) -> Result<Option<i64>, ReplayError> {
        self.phase_alpha_numerator_1e12.get(phase_index).map(|value| {
            let rounded = div_round_away_from_zero(*value, i128::from(SCALE_PPM))?;
            i64::try_from(rounded).map_err(|_| invalid("rounded alpha PPM exceeds i64"))
        }).transpose()
    }
    pub fn history(&self) -> &[DelayedAciStepReceipt] { &self.history }

    /// Process feedback for the current target (when due) and issue a new
    /// prediction for origin_step + horizon_steps. Feedback is None only in
    /// the initial horizon warm-up. The digest must bind the issued prediction
    /// set and its target timestamp.
    pub fn advance(
        &mut self,
        origin_step: u64,
        miscovered: Option<bool>,
        prediction_set_sha256: [u8; 32],
    ) -> Result<DelayedAciStepReceipt, ReplayError> {
        if origin_step != self.next_origin_step {
            return Err(invalid(format!(
                "delayed-ACI expected origin {}, received {}",
                self.next_origin_step, origin_step
            )));
        }
        if prediction_set_sha256 == [0; 32] {
            return Err(invalid("prediction-set digest must be nonzero"));
        }
        let phase_index = usize::try_from(origin_step % self.horizon_steps)
            .map_err(|_| invalid("delayed-ACI phase index overflow"))?;
        let mut consumed_feedback = None;
        let mut alpha_used_numerator_1e12 = self.phase_alpha_numerator_1e12[phase_index];
        let mut updated_alpha_numerator_1e12 = self.phase_alpha_numerator_1e12[phase_index];

        if origin_step < self.horizon_steps {
            if miscovered.is_some() {
                return Err(invalid("feedback supplied before first delayed target is due"));
            }
        } else {
            let observed_miss = miscovered
                .ok_or_else(|| invalid("fixed-horizon delayed-ACI feedback is missing"))?;
            let pending = self.pending.get(&origin_step)
                .ok_or_else(|| invalid("no queued prediction exists for the due target"))?;
            let expected_origin = origin_step.checked_sub(self.horizon_steps)
                .ok_or_else(|| invalid("delayed-ACI target underflow"))?;
            if pending.origin_step != expected_origin
                || pending.target_step != origin_step
                || pending.phase_index != phase_index
            {
                return Err(invalid("queued prediction identity/phase does not match due target"));
            }
            if self.phase_alpha_numerator_1e12[phase_index] != pending.alpha_used_numerator_1e12 {
                return Err(invalid("phase alpha state differs from alpha stored with due prediction"));
            }
            alpha_used_numerator_1e12 = pending.alpha_used_numerator_1e12;
            let error_ppm = if observed_miss { SCALE_PPM } else { 0 };
            // Exact rational update: alpha and gamma are PPM rationals, so
            // denominator 10^12 is fixed and no per-step rounding is needed.
            let delta_numerator = i128::from(self.gamma_ppm)
                * i128::from(self.target_alpha_ppm - error_ppm);
            updated_alpha_numerator_1e12 = pending.alpha_used_numerator_1e12
                .checked_add(delta_numerator)
                .ok_or_else(|| invalid("delayed-ACI exact rational alpha overflow"))?;
            consumed_feedback = Some(DelayedAciFeedbackReceipt {
                target_step: origin_step,
                prediction_origin_step: pending.origin_step,
                phase_index: pending.phase_index,
                alpha_used_numerator_1e12: pending.alpha_used_numerator_1e12,
                prediction_set_sha256: pending.prediction_set_sha256,
                miscovered: observed_miss,
            });
        }

        let issued_target_step = origin_step.checked_add(self.horizon_steps)
            .ok_or_else(|| invalid("delayed-ACI target step overflow"))?;
        if self.pending.contains_key(&issued_target_step) {
            return Err(invalid("duplicate delayed-ACI pending target"));
        }

        // Commit only after every validation and arithmetic operation succeeds.
        if consumed_feedback.is_some() { self.pending.remove(&origin_step); }
        self.phase_alpha_numerator_1e12[phase_index] = updated_alpha_numerator_1e12;
        self.pending.insert(issued_target_step, PendingDelayedAciPrediction {
            origin_step,
            target_step: issued_target_step,
            phase_index,
            alpha_used_numerator_1e12: updated_alpha_numerator_1e12,
            prediction_set_sha256,
        });
        let receipt = DelayedAciStepReceipt {
            origin_step,
            issued_target_step,
            phase_index,
            alpha_used_numerator_1e12,
            updated_alpha_numerator_1e12,
            prediction_set_sha256,
            consumed_feedback,
        };
        self.history.push(receipt.clone());
        self.next_origin_step = self.next_origin_step.checked_add(1)
            .ok_or_else(|| invalid("delayed-ACI origin counter overflow"))?;
        Ok(receipt)
    }

    /// Replay all ordered step receipts and compare the derived phase/pending
    /// state. This is deterministic replay, not proof of external data truth.
    pub fn verify(&self) -> Result<(), ReplayError> {
        let mut replay = Self::new(self.horizon_steps, self.target_alpha_ppm, self.gamma_ppm)?;
        for record in &self.history {
            let feedback = record.consumed_feedback.as_ref().map(|x| x.miscovered);
            let reconstructed = replay.advance(
                record.origin_step,
                feedback,
                record.prediction_set_sha256,
            )?;
            if reconstructed != *record {
                return Err(invalid("delayed-ACI receipt differs from deterministic replay"));
            }
        }
        if replay.phase_alpha_numerator_1e12 != self.phase_alpha_numerator_1e12
            || replay.pending != self.pending
            || replay.next_origin_step != self.next_origin_step
        {
            return Err(invalid("delayed-ACI phase or pending state differs from replay"));
        }
        Ok(())
    }

    /// SHA-256 over the domain-separated ordered step-receipt history.
    pub fn history_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"SYMTHAEA-TAU-DELAYED-ACI-REPLAY-V1");
        hasher.update(self.horizon_steps.to_le_bytes());
        hasher.update(self.target_alpha_ppm.to_le_bytes());
        hasher.update(self.gamma_ppm.to_le_bytes());
        hasher.update((self.history.len() as u64).to_le_bytes());
        for record in &self.history {
            hasher.update(record.origin_step.to_le_bytes());
            hasher.update(record.issued_target_step.to_le_bytes());
            hasher.update((record.phase_index as u64).to_le_bytes());
            hasher.update(record.alpha_used_numerator_1e12.to_le_bytes());
            hasher.update(record.updated_alpha_numerator_1e12.to_le_bytes());
            hasher.update(record.prediction_set_sha256);
            match &record.consumed_feedback {
                None => hasher.update([0]),
                Some(feedback) => {
                    hasher.update([1]);
                    hasher.update(feedback.target_step.to_le_bytes());
                    hasher.update(feedback.prediction_origin_step.to_le_bytes());
                    hasher.update((feedback.phase_index as u64).to_le_bytes());
                    hasher.update(feedback.alpha_used_numerator_1e12.to_le_bytes());
                    hasher.update(feedback.prediction_set_sha256);
                    hasher.update([u8::from(feedback.miscovered)]);
                }
            }
        }
        let digest = hasher.finalize();
        let mut output = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }
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

/// Maximum absolute deviation between target miscoverage and empirical
/// miscoverage over every contiguous window of the requested size.
/// Returns None when fewer than window_steps observations are available.
/// The result is rounded to integer PPM with ties away from zero.
pub fn worst_case_local_coverage_error_ppm(
    miscoverage: &[bool],
    target_alpha_ppm: i64,
    window_steps: usize,
) -> Result<Option<i64>, ReplayError> {
    if !(1..SCALE_PPM).contains(&target_alpha_ppm) {
        return Err(invalid("local-coverage target alpha must be in (0,1)"));
    }
    if window_steps == 0 {
        return Err(invalid("local-coverage window must be positive"));
    }
    if miscoverage.len() < window_steps {
        return Ok(None);
    }
    let mut count = miscoverage[..window_steps].iter().filter(|x| **x).count() as i128;
    let mut maximum_numerator = (count * i128::from(SCALE_PPM)
        - i128::from(target_alpha_ppm) * window_steps as i128).abs();
    for index in window_steps..miscoverage.len() {
        if miscoverage[index - window_steps] { count -= 1; }
        if miscoverage[index] { count += 1; }
        let numerator = (count * i128::from(SCALE_PPM)
            - i128::from(target_alpha_ppm) * window_steps as i128).abs();
        maximum_numerator = maximum_numerator.max(numerator);
    }
    let rounded = div_round_away_from_zero(maximum_numerator, window_steps as i128)?;
    let result = i64::try_from(rounded).map_err(|_| invalid("local-coverage error exceeds i64"))?;
    Ok(Some(result))
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
        assert_eq!(state.update_on_delivered_feedback(false).unwrap(), 60_000);
    }
    #[test]
    fn finite_sample_quantile_keeps_unbounded_case_explicit() {
        assert_eq!(finite_sample_radius(&[1,2,3,4,5,6,7,8], 100_000).unwrap(), None);
        assert_eq!(finite_sample_radius(&[8,1,7,2,6,3,5,4,9], 100_000).unwrap(), Some(9));
    }
    #[test]
    fn zero_and_full_tail_probabilities_are_exact() {
        let mut rng = SplitMix64::new(123);
        assert!(!(0..64).any(|_| tail_event(&mut rng, 0)));
        assert!((0..64).all(|_| tail_event(&mut rng, 1_000_000)));
    }
    #[test]
    fn fixture_rejects_float_transform_or_stream_contract_drift() {
        let altered = REGISTRY_JSON.replace(
            "u32=next_u32(); signed_ppm=floor(2000000*u32/(2^32-1))-1000000; use z_ppm=signed_ppm for residual innovations",
            "u=(next_u64+0.5)/2^64",
        );
        assert!(load_registry(&altered).is_err());
    }
    #[test]
    fn revision_schedule_cannot_backdate_initial_label_before_feedback() {
        let (_, mut scenarios) = load_builtin_registry().unwrap();
        let spec = scenarios.iter_mut().find(|s| s.id == "superseded_outcome_feedback").unwrap();
        spec.label_revisions[0].initial_available_step = 3005;
        assert!(generate_outcomes(spec, 83).is_err());
    }

    #[test]
    fn ledger_digest_is_stable_and_binds_revision_changes() {
        let mut ledger = BitemporalLabelLedger::default();
        ledger.append(LabelRevisionRecord {
            decision_id: 2, target_step: 1, revision: 1, available_at_step: 4,
            recorded_at_step: 4, kind: RevisionKind::Assert, label_milli: Some(19),
        }).unwrap();
        let original = ledger.sha256_hex();
        assert_eq!(original, ledger.clone().sha256_hex());
        ledger.append(LabelRevisionRecord {
            decision_id: 2, target_step: 1, revision: 2, available_at_step: 8,
            recorded_at_step: 8, kind: RevisionKind::Supersede, label_milli: Some(-19),
        }).unwrap();
        assert_ne!(original, ledger.sha256_hex());
        ledger.verify().unwrap();
    }
    fn digest(byte: u8) -> [u8; 32] { [byte; 32] }

    #[test]
    fn local_coverage_error_matches_all_contiguous_windows() {
        let errors = [true, false, false, true];
        assert_eq!(worst_case_local_coverage_error_ppm(&errors, 250_000, 2).unwrap(), Some(250_000));
        assert_eq!(worst_case_local_coverage_error_ppm(&errors, 250_000, 4).unwrap(), Some(250_000));
    }

    #[test]
    fn local_coverage_error_handles_insufficient_and_invalid_windows() {
        assert_eq!(worst_case_local_coverage_error_ppm(&[true, false], 100_000, 3).unwrap(), None);
        assert!(worst_case_local_coverage_error_ppm(&[true], 100_000, 0).is_err());
        assert!(worst_case_local_coverage_error_ppm(&[true], 1_000_000, 1).is_err());
    }

    #[test]
    fn delayed_aci_tau_one_matches_scalar_aci_inside_unit_interval() {
        let mut delayed = DelayedAciBank::new(1, 100_000, 10_000).unwrap();
        let mut scalar = AciState::new(100_000, 10_000).unwrap();
        let errors = [true, false, false, true, true, false];
        for (step, error) in errors.iter().enumerate() {
            let feedback = if step == 0 { None } else { Some(errors[step - 1]) };
            let alpha_used = i128::from(scalar.current_alpha_ppm()) * i128::from(SCALE_PPM);
            let record = delayed.advance(step as u64, feedback, digest(step as u8 + 1)).unwrap();
            assert_eq!(record.alpha_used_numerator_1e12, alpha_used);
            if step > 0 { scalar.update_on_delivered_feedback(errors[step - 1]).unwrap(); }
            assert_eq!(
                record.updated_alpha_numerator_1e12,
                i128::from(scalar.current_alpha_ppm()) * i128::from(SCALE_PPM)
            );
        }
        delayed.verify().unwrap();
    }

    #[test]
    fn delayed_aci_uses_tau_independent_phase_chains() {
        let mut state = DelayedAciBank::new(3, 100_000, 50_000).unwrap();
        for step in 0..3 {
            let receipt = state.advance(step, None, digest(step as u8 + 1)).unwrap();
            assert_eq!(receipt.alpha_used_ppm, 100_000);
        }
        let r3 = state.advance(3, Some(true), digest(4)).unwrap();
        let r4 = state.advance(4, Some(false), digest(5)).unwrap();
        let r5 = state.advance(5, Some(true), digest(6)).unwrap();
        assert_eq!((r3.phase_index, r3.alpha_used_numerator_1e12, r3.updated_alpha_numerator_1e12),
            (0, 100_000_000_000, 55_000_000_000));
        assert_eq!((r4.phase_index, r4.alpha_used_numerator_1e12, r4.updated_alpha_numerator_1e12),
            (1, 100_000_000_000, 105_000_000_000));
        assert_eq!((r5.phase_index, r5.alpha_used_numerator_1e12, r5.updated_alpha_numerator_1e12),
            (2, 100_000_000_000, 55_000_000_000));
        let r6 = state.advance(6, Some(false), digest(7)).unwrap();
        assert_eq!((r6.phase_index, r6.alpha_used_numerator_1e12, r6.updated_alpha_numerator_1e12),
            (0, 55_000_000_000, 60_000_000_000));
        assert_eq!(state.phase_alpha_numerator_1e12(1), Some(105_000_000_000));
        assert_eq!(state.phase_alpha_numerator_1e12(2), Some(55_000_000_000));
        state.verify().unwrap();
    }

    #[test]
    fn delayed_aci_requires_fixed_horizon_feedback_and_is_transactional() {
        let mut state = DelayedAciBank::new(2, 100_000, 20_000).unwrap();
        state.advance(0, None, digest(1)).unwrap();
        state.advance(1, None, digest(2)).unwrap();
        let before = state.clone();
        assert!(state.advance(2, None, digest(3)).is_err());
        assert_eq!(state, before);
        assert!(state.advance(4, Some(true), digest(3)).is_err());
        assert_eq!(state, before);
        state.advance(2, Some(true), digest(3)).unwrap();
        state.verify().unwrap();
    }

    #[test]
    fn delayed_aci_replay_detects_tampered_receipt_and_digest_binds_history() {
        let mut state = DelayedAciBank::new(2, 100_000, 20_000).unwrap();
        for step in 0..8 {
            let feedback = if step < 2 { None } else { Some(step % 2 == 0) };
            state.advance(step, feedback, digest(step as u8 + 1)).unwrap();
        }
        let original = state.history_sha256();
        state.verify().unwrap();
        let mut tampered = state.clone();
        tampered.history[3].alpha_used_ppm += 1;
        assert!(tampered.verify().is_err());
        assert_ne!(original, tampered.history_sha256());
    }

    #[test]
    fn delayed_aci_preserves_out_of_unit_levels_and_degenerate_set_semantics() {
        let mut state = DelayedAciBank::new(1, 100_000, 1_000_000).unwrap();
        state.advance(0, None, digest(1)).unwrap();
        let r1 = state.advance(1, Some(true), digest(2)).unwrap();
        assert_eq!((r1.alpha_used_numerator_1e12, r1.updated_alpha_numerator_1e12),
            (100_000_000_000, -800_000_000_000));
        assert_eq!(aci_set_regime(r1.updated_alpha_numerator_1e12), AciSetRegime::AllReal);
        let r2 = state.advance(2, Some(false), digest(3)).unwrap();
        assert_eq!((r2.alpha_used_numerator_1e12, r2.updated_alpha_numerator_1e12),
            (-800_000_000_000, -700_000_000_000));
        assert_eq!(aci_set_regime(ALPHA_DENOMINATOR_1E12 + 1), AciSetRegime::Empty);
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

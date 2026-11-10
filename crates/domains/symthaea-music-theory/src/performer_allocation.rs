// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! V2 performer / instrument / part allocation.
//!
//! The time-indexed orchestration timeline says which instrument instance
//! realizes a persistent PartId. This layer separately identifies the human
//! performer(s), declared setup, and evidence-qualified simultaneous workload.
//!
//! V1 supports explicit topology for:
//! - one performer carrying multiple parts on one instrument;
//! - multiple performers sharing one instrument across distinct parts;
//! - one part realized in unison by a declared performer group.
//!
//! No topology is inferred from VoiceRole, register, note overlap or MIDI
//! channel. Instrument polyphony stays in InstrumentCapabilityV1; performer
//! capacity is a separate, source-bound claim.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::orchestration::{InstrumentCapabilityV1, InstrumentKindV1};
use crate::orchestration_timeline::{
    validate_orchestration_timeline, OrchestrationTimelineV1,
    OrchestrationTimelineReportV1, ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1,
};
use crate::rhythm::Duration;
use crate::score::{PartId, Score};

pub const PERFORMANCE_ALLOCATION_SCHEMA_V2: &str = "performance-allocation-v2";
pub const MAX_PERFORMANCE_UNITS_V2: usize = 4_096;
pub const MAX_PERFORMER_PROFILES_V2: usize = 1_024;
pub const MAX_PERFORMERS_PER_UNIT_V2: usize = 64;
pub const MAX_USED_PERFORMERS_V2: usize = 1_024;
const MAX_BEAT_NUMERATOR: i128 = 1_000_000_000;
const MAX_BEAT_DENOMINATOR: i128 = 1_000_000;

/// Only Qualified evidence may satisfy a hard performer-capacity gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformerQualificationV2 {
    Qualified,
    Conditional,
    ResearchOnly,
    Unqualified,
}

/// A source-bound capacity claim scoped to a performer, instrument kind and
/// concrete setup (e.g. two-hand piano, four-hand piano or organ).
///
/// These are performer limits, not instrument polyphony limits. The latter
/// remain attached to the instrument instance in InstrumentCapabilityV1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformerCapabilityV2 {
    pub performer_id: String,
    pub instrument_kind: InstrumentKindV1,
    pub setup_id: String,
    pub max_simultaneous_parts: u8,
    pub max_simultaneous_notes: u8,
    pub evidence_sha256: String,
    pub qualification: PerformerQualificationV2,
}

/// Explicit staffing topology for one orchestration timeline segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceAllocationModeV2 {
    /// One performer realizes one PartId on one instrument instance.
    Solo,
    /// One performer carries multiple distinct PartIds on the same
    /// instrument/setup within the same allocation group.
    SharedPerformerMultiPart,
    /// Different performers realize distinct parts on one shared physical
    /// instrument instance, e.g. four-hand piano.
    SharedInstrumentMultiPerformer,
    /// One PartId is realized in unison by a declared performer group.
    /// Divisi is deliberately not represented by this v2 mode.
    EnsembleUnison,
}

/// Staffing overlay for exactly one existing orchestration timeline segment.
/// PartId, instrument ID and time boundaries remain owned by that segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceUnitV2 {
    pub id: String,
    pub orchestration_segment_id: String,
    pub performer_ids: Vec<String>,
    pub setup_id: String,
    pub mode: PerformanceAllocationModeV2,
    pub allocation_group_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceAllocationPlanV2 {
    pub schema: String,
    pub score_digest_schema: String,
    pub score_sha256: String,
    pub units: Vec<PerformanceUnitV2>,
    pub performer_capabilities: Vec<PerformerCapabilityV2>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceAllocationIssueCodeV2 {
    InvalidSchema,
    InvalidScoreDigestSchema,
    InvalidScoreDigest,
    InvalidExpectedScoreDigest,
    ScoreDigestMismatch,
    InvalidBaseTimeline,
    MissingPartIdentity,
    ResourceLimitExceeded,
    InvalidUnitId,
    DuplicateUnitId,
    DuplicateSegmentAllocation,
    MissingSegmentAllocation,
    UnknownTimelineSegment,
    InvalidPerformerId,
    DuplicatePerformerInUnit,
    InvalidPerformerRoster,
    PerformerResourceLimitExceeded,
    InvalidSetupId,
    InvalidAllocationGroupId,
    InvalidModeShape,
    InvalidPerformerProfile,
    DuplicatePerformerProfile,
    MissingPerformerProfile,
    UnqualifiedPerformerProfile,
    InvalidPerformerCapacity,
    PerformerDoubleBooked,
    SharedInstrumentWithoutDeclaredGroup,
    InvalidSharedPerformerGroup,
    InvalidSharedInstrumentGroup,
    PerformerPolyphonyExceeded,
    PerformerPartCapacityExceeded,
    InvalidScoreTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceAllocationIssueV2 {
    pub code: PerformanceAllocationIssueCodeV2,
    pub unit_id: Option<String>,
    pub performer_id: Option<String>,
    pub part_id: Option<u16>,
    pub note_index: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformerPeakLoadV2 {
    pub performer_id: String,
    pub instrument_kind: InstrumentKindV1,
    pub setup_id: String,
    pub peak_simultaneous_notes: u8,
    pub declared_note_budget: u8,
    pub peak_simultaneous_parts: u8,
    pub declared_part_budget: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceAllocationReportV2 {
    pub valid: bool,
    pub unit_count: usize,
    pub performer_count: usize,
    pub orchestration_report: OrchestrationTimelineReportV1,
    pub performer_peak_load: Vec<PerformerPeakLoadV2>,
    pub issues: Vec<PerformanceAllocationIssueV2>,
}

impl PerformanceAllocationReportV2 {
    fn push(
        &mut self,
        code: PerformanceAllocationIssueCodeV2,
        unit_id: Option<&str>,
        performer_id: Option<&str>,
        part_id: Option<u16>,
        note_index: Option<usize>,
        message: impl Into<String>,
    ) {
        self.issues.push(PerformanceAllocationIssueV2 {
            code,
            unit_id: unit_id.map(str::to_owned),
            performer_id: performer_id.map(str::to_owned),
            part_id,
            note_index,
            message: message.into(),
        });
    }

    fn finish(mut self) -> Self {
        self.valid = self.issues.is_empty() && self.orchestration_report.valid;
        self.performer_peak_load.sort_by(|a, b| {
            a.performer_id.cmp(&b.performer_id)
                .then_with(|| a.setup_id.cmp(&b.setup_id))
                .then_with(|| format!("{:?}", a.instrument_kind).cmp(&format!("{:?}", b.instrument_kind)))
        });
        // load_events is a HashMap for its non-Ord instrument-kind key.
        // Normalize all findings before returning so identical frozen inputs
        // produce byte-stable report ordering.
        self.issues.sort_by(|a, b| {
            a.code.cmp(&b.code)
                .then_with(|| a.unit_id.cmp(&b.unit_id))
                .then_with(|| a.performer_id.cmp(&b.performer_id))
                .then_with(|| a.part_id.cmp(&b.part_id))
                .then_with(|| a.note_index.cmp(&b.note_index))
                .then_with(|| a.message.cmp(&b.message))
        });
        self
    }
}

#[derive(Debug, Clone, Copy)]
struct Moment {
    num: i128,
    den: i128,
}

impl Moment {
    fn from_pair(num: i128, den: i128) -> Option<Self> {
        if den <= 0 || den > MAX_BEAT_DENOMINATOR || num < 0 || num > MAX_BEAT_NUMERATOR {
            return None;
        }
        Some(Self { num, den })
    }

    fn from_wire(value: crate::orchestration_timeline::RationalBeatV1) -> Option<Self> {
        Self::from_pair(i128::from(value.numerator), i128::from(value.denominator))
    }

    fn from_duration(value: Duration) -> Option<Self> {
        Self::from_pair(i128::from(value.num()), i128::from(value.den()))
    }

    fn checked_add(self, rhs: Self) -> Option<Self> {
        let left = self.num.checked_mul(rhs.den)?;
        let right = rhs.num.checked_mul(self.den)?;
        let num = left.checked_add(right)?;
        let den = self.den.checked_mul(rhs.den)?;
        Self::from_pair(num, den)
    }

    fn compare(self, rhs: Self) -> Ordering {
        (self.num * rhs.den).cmp(&(rhs.num * self.den))
    }

    fn less_than(self, rhs: Self) -> bool {
        self.compare(rhs) == Ordering::Less
    }

    fn greater_than(self, rhs: Self) -> bool {
        self.compare(rhs) == Ordering::Greater
    }

    fn equal(self, rhs: Self) -> bool {
        self.compare(rhs) == Ordering::Equal
    }
}

#[derive(Debug, Clone, Copy)]
struct UnitWindow {
    start: Moment,
    end: Moment,
}

#[derive(Debug, Clone, Copy)]
struct LoadBucketKey<'a> {
    performer_id: &'a str,
    instrument_kind: InstrumentKindV1,
    setup_id: &'a str,
}

#[derive(Debug, Clone, Copy)]
struct LoadEvent {
    when: Moment,
    delta: i8,
    part_id: u16,
    note_index: usize,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 128
}

fn windows_overlap(left: UnitWindow, right: UnitWindow) -> bool {
    left.start.less_than(right.end) && left.end.greater_than(right.start)
}

fn same_window(left: UnitWindow, right: UnitWindow) -> bool {
    left.start.equal(right.start) && left.end.equal(right.end)
}

/// Validate performer staffing against an already source-bound
/// PartId-to-instrument timeline and its qualified instrument-instance
/// profiles. Instrument polyphony/range is delegated to the timeline verifier;
/// this layer adds performer IDs, setup, sharing topology and aggregate load.
pub fn validate_performance_allocation_v2(
    score: &Score,
    timeline: &OrchestrationTimelineV1,
    instruments: &[InstrumentCapabilityV1],
    plan: &PerformanceAllocationPlanV2,
    expected_score_sha256: &str,
) -> PerformanceAllocationReportV2 {
    let orchestration_report = validate_orchestration_timeline(
        score,
        timeline,
        instruments,
        expected_score_sha256,
    );
    let mut report = PerformanceAllocationReportV2 {
        valid: false,
        unit_count: plan.units.len(),
        performer_count: 0,
        orchestration_report,
        performer_peak_load: Vec::new(),
        issues: Vec::new(),
    };

    if plan.schema != PERFORMANCE_ALLOCATION_SCHEMA_V2 {
        report.push(PerformanceAllocationIssueCodeV2::InvalidSchema, None, None, None, None,
            "unsupported performance-allocation schema");
    }
    if plan.score_digest_schema != ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1
        || timeline.score_digest_schema != ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1
    {
        report.push(PerformanceAllocationIssueCodeV2::InvalidScoreDigestSchema, None, None, None, None,
            "allocation and timeline must use the exact score-demand digest schema");
    }
    if !valid_digest(&plan.score_sha256) {
        report.push(PerformanceAllocationIssueCodeV2::InvalidScoreDigest, None, None, None, None,
            "allocation score_sha256 must be 64 lowercase hexadecimal characters");
    }
    if !valid_digest(expected_score_sha256) {
        report.push(PerformanceAllocationIssueCodeV2::InvalidExpectedScoreDigest, None, None, None, None,
            "expected score digest must come from trusted score-demand/provenance");
    } else if valid_digest(&plan.score_sha256)
        && (plan.score_sha256 != expected_score_sha256
            || timeline.score_sha256 != expected_score_sha256)
    {
        report.push(PerformanceAllocationIssueCodeV2::ScoreDigestMismatch, None, None, None, None,
            "allocation, timeline and trusted expected score digests must match");
    }
    if !score.has_part_identity() {
        report.push(PerformanceAllocationIssueCodeV2::MissingPartIdentity, None, None, None, None,
            "performer allocation requires complete generator-authored PartIds");
    }
    if plan.units.len() > MAX_PERFORMANCE_UNITS_V2
        || plan.performer_capabilities.len() > MAX_PERFORMER_PROFILES_V2
    {
        report.push(PerformanceAllocationIssueCodeV2::ResourceLimitExceeded, None, None, None, None,
            "performance units or performer profiles exceed v2 resource limits");
        return report.finish();
    }
    if !report.orchestration_report.valid {
        report.push(PerformanceAllocationIssueCodeV2::InvalidBaseTimeline, None, None, None, None,
            "underlying score-to-instrument timeline is invalid");
    }

    let instrument_by_id: BTreeMap<&str, &InstrumentCapabilityV1> =
        instruments.iter().map(|item| (item.id.as_str(), item)).collect();
    let segment_by_id: BTreeMap<&str, &crate::orchestration_timeline::OrchestrationSegmentV1> =
        timeline.segments.iter().map(|item| (item.id.as_str(), item)).collect();
    let mut windows = BTreeMap::new();
    for segment in &timeline.segments {
        if let (Some(start), Some(end)) = (
            Moment::from_wire(segment.start),
            Moment::from_wire(segment.end),
        ) {
            if start.less_than(end) {
                windows.insert(segment.id.as_str(), UnitWindow { start, end });
            }
        }
    }

    // InstrumentKindV1 is hashed but not ordered, so the profile index is a
    // HashMap rather than a BTreeMap. Output ordering is normalized separately.
    let mut profile_by_key: HashMap<(&str, InstrumentKindV1, &str), &PerformerCapabilityV2> =
        HashMap::new();
    for profile in &plan.performer_capabilities {
        if !valid_identifier(&profile.performer_id)
            || !valid_identifier(&profile.setup_id)
            || !valid_digest(&profile.evidence_sha256)
            || profile.max_simultaneous_parts == 0
            || profile.max_simultaneous_notes == 0
            || profile.max_simultaneous_parts > 16
            || profile.max_simultaneous_notes > 64
        {
            report.push(PerformanceAllocationIssueCodeV2::InvalidPerformerProfile, None,
                Some(&profile.performer_id), None, None,
                "performer profile requires bounded capacities, valid evidence digest and non-empty IDs/setup");
            continue;
        }
        let key = (
            profile.performer_id.as_str(),
            profile.instrument_kind,
            profile.setup_id.as_str(),
        );
        if profile_by_key.insert(key, profile).is_some() {
            report.push(PerformanceAllocationIssueCodeV2::DuplicatePerformerProfile, None,
                Some(&profile.performer_id), None, None,
                "performer/instrument-kind/setup profile must be unique");
        }
        if profile.qualification != PerformerQualificationV2::Qualified {
            report.push(PerformanceAllocationIssueCodeV2::UnqualifiedPerformerProfile, None,
                Some(&profile.performer_id), None, None,
                "only Qualified source-bound performer evidence can satisfy a hard gate");
        }
    }

    let mut seen_unit_ids = BTreeSet::new();
    let mut segment_allocations = BTreeSet::new();
    let mut units_by_performer: BTreeMap<&str, Vec<&PerformanceUnitV2>> = BTreeMap::new();
    let mut units_by_instrument: BTreeMap<&str, Vec<&PerformanceUnitV2>> = BTreeMap::new();
    let mut shared_performer_groups: BTreeMap<(&str, &str, &str), Vec<&PerformanceUnitV2>> =
        BTreeMap::new();
    let mut shared_instrument_groups: BTreeMap<(&str, &str), Vec<&PerformanceUnitV2>> =
        BTreeMap::new();
    let mut used_performers = BTreeSet::new();

    for unit in &plan.units {
        if !valid_identifier(&unit.id) {
            report.push(PerformanceAllocationIssueCodeV2::InvalidUnitId, Some(&unit.id), None, None, None,
                "performance unit ID must be non-empty and bounded");
        } else if !seen_unit_ids.insert(unit.id.as_str()) {
            report.push(PerformanceAllocationIssueCodeV2::DuplicateUnitId, Some(&unit.id), None, None, None,
                "performance unit IDs must be unique");
        }
        if !valid_identifier(&unit.setup_id) {
            report.push(PerformanceAllocationIssueCodeV2::InvalidSetupId, Some(&unit.id), None, None, None,
                "setup_id must be non-empty and bounded");
        }

        let Some(segment) = segment_by_id.get(unit.orchestration_segment_id.as_str()) else {
            report.push(PerformanceAllocationIssueCodeV2::UnknownTimelineSegment, Some(&unit.id), None, None, None,
                "unit must reference an existing orchestration segment");
            continue;
        };
        if !segment_allocations.insert(unit.orchestration_segment_id.as_str()) {
            report.push(PerformanceAllocationIssueCodeV2::DuplicateSegmentAllocation, Some(&unit.id), None,
                Some(segment.part_id), None,
                "each orchestration segment must have exactly one performance allocation unit");
        }
        if unit.performer_ids.is_empty() || unit.performer_ids.len() > MAX_PERFORMERS_PER_UNIT_V2 {
            report.push(PerformanceAllocationIssueCodeV2::InvalidPerformerRoster, Some(&unit.id), None,
                Some(segment.part_id), None,
                "performer roster must contain 1..=64 explicit performer IDs");
            continue;
        }
        let Some(instrument) = instrument_by_id.get(segment.instrument_id.as_str()) else {
            continue;
        };
        units_by_instrument.entry(segment.instrument_id.as_str()).or_default().push(unit);
        let mut local_ids = BTreeSet::new();
        for performer_id in &unit.performer_ids {
            if !valid_identifier(performer_id) {
                report.push(PerformanceAllocationIssueCodeV2::InvalidPerformerId, Some(&unit.id),
                    Some(performer_id), Some(segment.part_id), None,
                    "performer ID must be non-empty and bounded");
                continue;
            }
            if !local_ids.insert(performer_id.as_str()) {
                report.push(PerformanceAllocationIssueCodeV2::DuplicatePerformerInUnit, Some(&unit.id),
                    Some(performer_id), Some(segment.part_id), None,
                    "performer roster cannot contain duplicate IDs");
                continue;
            }
            let key = (performer_id.as_str(), instrument.kind, unit.setup_id.as_str());
            if !profile_by_key.contains_key(&key) {
                report.push(PerformanceAllocationIssueCodeV2::MissingPerformerProfile, Some(&unit.id),
                    Some(performer_id), Some(segment.part_id), None,
                    "no performer profile matches performer, instrument kind and setup");
            }
            used_performers.insert(performer_id.as_str());
            units_by_performer.entry(performer_id.as_str()).or_default().push(unit);
        }

        if unit.allocation_group_id.as_ref().is_some_and(|id| !valid_identifier(id)) {
            report.push(PerformanceAllocationIssueCodeV2::InvalidAllocationGroupId, Some(&unit.id), None,
                Some(segment.part_id), None,
                "allocation_group_id must be absent or non-empty and bounded");
        }
        let shape_valid = match unit.mode {
            PerformanceAllocationModeV2::Solo =>
                unit.performer_ids.len() == 1 && unit.allocation_group_id.is_none(),
            PerformanceAllocationModeV2::SharedPerformerMultiPart
            | PerformanceAllocationModeV2::SharedInstrumentMultiPerformer =>
                unit.performer_ids.len() == 1
                    && unit.allocation_group_id.as_ref().is_some_and(|id| valid_identifier(id)),
            PerformanceAllocationModeV2::EnsembleUnison =>
                unit.performer_ids.len() >= 2
                    && unit.allocation_group_id.as_ref().is_some_and(|id| valid_identifier(id)),
        };
        if !shape_valid {
            report.push(PerformanceAllocationIssueCodeV2::InvalidModeShape, Some(&unit.id), None,
                Some(segment.part_id), None,
                "performer roster and allocation_group_id do not match the declared mode");
        }
        match unit.mode {
            PerformanceAllocationModeV2::SharedPerformerMultiPart => {
                if let (Some(performer), Some(group)) =
                    (unit.performer_ids.first(), unit.allocation_group_id.as_deref())
                {
                    shared_performer_groups.entry((group, performer.as_str(), segment.instrument_id.as_str()))
                        .or_default().push(unit);
                }
            }
            PerformanceAllocationModeV2::SharedInstrumentMultiPerformer => {
                if let Some(group) = unit.allocation_group_id.as_deref() {
                    shared_instrument_groups.entry((group, segment.instrument_id.as_str()))
                        .or_default().push(unit);
                }
            }
            _ => {}
        }
    }
    if used_performers.len() > MAX_USED_PERFORMERS_V2 {
        report.push(
            PerformanceAllocationIssueCodeV2::PerformerResourceLimitExceeded,
            None,
            None,
            None,
            None,
            "distinct performer count exceeds the v2 resource limit",
        );
    }
    report.performer_count = used_performers.len();

    for segment in &timeline.segments {
        if !segment_allocations.contains(segment.id.as_str()) {
            report.push(PerformanceAllocationIssueCodeV2::MissingSegmentAllocation, None, None,
                Some(segment.part_id), None,
                "every orchestration segment must have one performer allocation unit");
        }
    }

    // One human may carry multiple parts only under one explicitly named
    // shared-performer group, same instrument/setup and exact time window.
    for ((group, performer, instrument_id), units) in &shared_performer_groups {
        let mut parts = BTreeSet::new();
        let mut first_window: Option<UnitWindow> = None;
        let mut first_setup: Option<&str> = None;
        for unit in units {
            if let Some(segment) = segment_by_id.get(unit.orchestration_segment_id.as_str()) {
                parts.insert(segment.part_id);
                if segment.instrument_id != *instrument_id {
                    report.push(PerformanceAllocationIssueCodeV2::InvalidSharedPerformerGroup,
                        Some(&unit.id), Some(*performer), Some(segment.part_id), None,
                        "shared-performer parts must use the same instrument instance");
                }
            }
            if let Some(window) = windows.get(unit.orchestration_segment_id.as_str()).copied() {
                if let Some(first) = first_window {
                    if !same_window(first, window) {
                        report.push(PerformanceAllocationIssueCodeV2::InvalidSharedPerformerGroup,
                            Some(&unit.id), Some(*performer), None, None,
                            "shared-performer parts must have the same exact score-time window");
                    }
                } else {
                    first_window = Some(window);
                }
            }
            if let Some(setup) = first_setup {
                if setup != unit.setup_id {
                    report.push(PerformanceAllocationIssueCodeV2::InvalidSharedPerformerGroup,
                        Some(&unit.id), Some(*performer), None, None,
                        "shared-performer parts must use the same setup");
                }
            } else {
                first_setup = Some(&unit.setup_id);
            }
        }
        if units.len() < 2 || parts.len() < 2 {
            report.push(PerformanceAllocationIssueCodeV2::InvalidSharedPerformerGroup,
                None, Some(*performer), None, None,
                format!("shared-performer group {group} must contain at least two distinct parts"));
        }
    }

    // Multiple players can share one physical instrument only in an explicit
    // shared-instrument group, same setup/window and distinct part identities.
    for ((group, instrument_id), units) in &shared_instrument_groups {
        let mut performers = BTreeSet::new();
        let mut parts = BTreeSet::new();
        let mut first_window: Option<UnitWindow> = None;
        let mut first_setup: Option<&str> = None;
        for unit in units {
            if let Some(performer) = unit.performer_ids.first() {
                performers.insert(performer.as_str());
            }
            if let Some(segment) = segment_by_id.get(unit.orchestration_segment_id.as_str()) {
                parts.insert(segment.part_id);
                if segment.instrument_id != *instrument_id {
                    report.push(PerformanceAllocationIssueCodeV2::InvalidSharedInstrumentGroup,
                        Some(&unit.id), None, Some(segment.part_id), None,
                        "shared-instrument group references different instrument instances");
                }
            }
            if let Some(window) = windows.get(unit.orchestration_segment_id.as_str()).copied() {
                if let Some(first) = first_window {
                    if !same_window(first, window) {
                        report.push(PerformanceAllocationIssueCodeV2::InvalidSharedInstrumentGroup,
                            Some(&unit.id), None, None, None,
                            "shared-instrument group requires identical score-time windows");
                    }
                } else {
                    first_window = Some(window);
                }
            }
            if let Some(setup) = first_setup {
                if setup != unit.setup_id {
                    report.push(PerformanceAllocationIssueCodeV2::InvalidSharedInstrumentGroup,
                        Some(&unit.id), None, None, None,
                        "shared-instrument group requires the same setup");
                }
            } else {
                first_setup = Some(&unit.setup_id);
            }
        }
        if units.len() < 2 || performers.len() < 2 || parts.len() < 2 {
            report.push(PerformanceAllocationIssueCodeV2::InvalidSharedInstrumentGroup,
                None, None, None, None,
                format!("shared-instrument group {group} requires multiple performers and distinct parts"));
        }
    }

    // A physical instrument instance cannot be implicitly shared by
    // different performers. All concurrent part allocations for one instance
    // must opt into the same explicit shared-instrument group and setup.
    for (instrument_id, units) in &units_by_instrument {
        for (index, left) in units.iter().enumerate() {
            let (Some(left_segment), Some(left_window)) = (
                segment_by_id.get(left.orchestration_segment_id.as_str()),
                windows.get(left.orchestration_segment_id.as_str()).copied(),
            ) else { continue };
            for right in units.iter().skip(index + 1) {
                let (Some(right_segment), Some(right_window)) = (
                    segment_by_id.get(right.orchestration_segment_id.as_str()),
                    windows.get(right.orchestration_segment_id.as_str()).copied(),
                ) else { continue };
                if !windows_overlap(left_window, right_window) {
                    continue;
                }
                let performers_are_distinct = left.performer_ids.first()
                    != right.performer_ids.first();
                let same_group = left.allocation_group_id.is_some()
                    && left.allocation_group_id == right.allocation_group_id
                    && left.setup_id == right.setup_id
                    && left_segment.part_id != right_segment.part_id;
                let shared_performer =
                    left.mode == PerformanceAllocationModeV2::SharedPerformerMultiPart
                    && right.mode == PerformanceAllocationModeV2::SharedPerformerMultiPart
                    && !performers_are_distinct
                    && same_group;
                let shared_instrument =
                    left.mode == PerformanceAllocationModeV2::SharedInstrumentMultiPerformer
                    && right.mode == PerformanceAllocationModeV2::SharedInstrumentMultiPerformer
                    && performers_are_distinct
                    && same_group;
                let explicitly_shared = shared_performer || shared_instrument;
                if !explicitly_shared {
                    report.push(
                        PerformanceAllocationIssueCodeV2::SharedInstrumentWithoutDeclaredGroup,
                        Some(&right.id),
                        right.performer_ids.first().map(String::as_str),
                        Some(right_segment.part_id),
                        None,
                        format!(
                            "overlapping uses of instrument instance {instrument_id} require one explicit shared-instrument group, distinct performers, and one setup"
                        ),
                    );
                }
            }
        }
    }

    // Any overlap for one performer must be covered by one explicit shared
    // performer group; switching instruments while simultaneous is forbidden.
    for (performer, units) in &units_by_performer {
        for (index, left) in units.iter().enumerate() {
            let (Some(left_segment), Some(left_window)) = (
                segment_by_id.get(left.orchestration_segment_id.as_str()),
                windows.get(left.orchestration_segment_id.as_str()).copied(),
            ) else { continue };
            for right in units.iter().skip(index + 1) {
                let (Some(right_segment), Some(right_window)) = (
                    segment_by_id.get(right.orchestration_segment_id.as_str()),
                    windows.get(right.orchestration_segment_id.as_str()).copied(),
                ) else { continue };
                if !windows_overlap(left_window, right_window) {
                    continue;
                }
                let explicitly_shared =
                    left.mode == PerformanceAllocationModeV2::SharedPerformerMultiPart
                    && right.mode == PerformanceAllocationModeV2::SharedPerformerMultiPart
                    && left.allocation_group_id == right.allocation_group_id
                    && left_segment.instrument_id == right_segment.instrument_id
                    && left.setup_id == right.setup_id;
                if !explicitly_shared {
                    report.push(PerformanceAllocationIssueCodeV2::PerformerDoubleBooked,
                        Some(&right.id), Some(*performer), Some(right_segment.part_id), None,
                        "overlapping performer assignments require one explicit shared-performer group on the same instrument/setup");
                }
            }
        }
    }

    // Compute the actual simultaneous note and PartId workload per performer.
    // End events sort before start events at equal rational score time.
    let mut load_events: HashMap<(&str, InstrumentKindV1, &str), Vec<LoadEvent>> = HashMap::new();
    for unit in &plan.units {
        let (Some(segment), Some(window)) = (
            segment_by_id.get(unit.orchestration_segment_id.as_str()),
            windows.get(unit.orchestration_segment_id.as_str()).copied(),
        ) else { continue };
        let Some(instrument) = instrument_by_id.get(segment.instrument_id.as_str()) else { continue };
        for (note_index, note) in score.notes.iter().enumerate() {
            if note.part.0 != segment.part_id { continue; }
            let (Some(onset), Some(duration)) = (
                Moment::from_duration(note.onset),
                Moment::from_duration(note.duration),
            ) else {
                report.push(PerformanceAllocationIssueCodeV2::InvalidScoreTime,
                    Some(&unit.id), None, Some(segment.part_id), Some(note_index),
                    "note timing is not representable as bounded rational score time");
                continue;
            };
            let Some(end) = onset.checked_add(duration) else {
                report.push(PerformanceAllocationIssueCodeV2::InvalidScoreTime,
                    Some(&unit.id), None, Some(segment.part_id), Some(note_index),
                    "note end exceeds bounded rational arithmetic");
                continue;
            };
            if onset.less_than(window.start) || end.greater_than(window.end) { continue; }
            for performer in &unit.performer_ids {
                let key = (performer.as_str(), instrument.kind, unit.setup_id.as_str());
                let events = load_events.entry(key).or_default();
                events.push(LoadEvent { when: end, delta: -1, part_id: segment.part_id, note_index });
                events.push(LoadEvent { when: onset, delta: 1, part_id: segment.part_id, note_index });
            }
        }
    }

    let mut peak_by_scope = Vec::new();
    for ((performer, instrument_kind, setup_id), events) in &mut load_events {
        events.sort_by(|a, b| {
            a.when.compare(b.when)
                .then_with(|| a.delta.cmp(&b.delta))
                .then_with(|| a.part_id.cmp(&b.part_id))
                .then_with(|| a.note_index.cmp(&b.note_index))
        });
        let mut active_notes = 0u16;
        let mut active_parts = BTreeMap::<u16, u16>::new();
        let mut peak_notes = 0u16;
        let mut peak_parts = 0u16;
        for event in events.iter() {
            if event.delta < 0 {
                active_notes = active_notes.saturating_sub(1);
                if let Some(count) = active_parts.get_mut(&event.part_id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 { active_parts.remove(&event.part_id); }
                }
            } else {
                active_notes = active_notes.saturating_add(1);
                *active_parts.entry(event.part_id).or_default() += 1;
            }
            peak_notes = peak_notes.max(active_notes);
            peak_parts = peak_parts.max(active_parts.len() as u16);
        }

        let Some(profile) = profile_by_key.get(&(*performer, *instrument_kind, *setup_id)) else {
            continue;
        };
        let peak = PerformerPeakLoadV2 {
            performer_id: (*performer).to_owned(),
            instrument_kind: *instrument_kind,
            setup_id: (*setup_id).to_owned(),
            peak_simultaneous_notes: peak_notes.min(u16::from(u8::MAX)) as u8,
            declared_note_budget: profile.max_simultaneous_notes,
            peak_simultaneous_parts: peak_parts.min(u16::from(u8::MAX)) as u8,
            declared_part_budget: profile.max_simultaneous_parts,
        };
        if peak_notes > u16::from(profile.max_simultaneous_notes) {
            report.push(PerformanceAllocationIssueCodeV2::PerformerPolyphonyExceeded,
                None, Some(*performer), None, None,
                format!("performer simultaneous-note demand {} exceeds qualified budget {}",
                    peak_notes, profile.max_simultaneous_notes));
        }
        if peak_parts > u16::from(profile.max_simultaneous_parts) {
            report.push(PerformanceAllocationIssueCodeV2::PerformerPartCapacityExceeded,
                None, Some(*performer), None, None,
                format!("performer simultaneous-part demand {} exceeds qualified budget {}",
                    peak_parts, profile.max_simultaneous_parts));
        }
        peak_by_scope.push(peak);
    }
    peak_by_scope.sort_by(|a, b| {
        a.performer_id.cmp(&b.performer_id)
            .then_with(|| a.setup_id.cmp(&b.setup_id))
            .then_with(|| format!("{:?}", a.instrument_kind).cmp(&format!("{:?}", b.instrument_kind)))
    });
    report.performer_peak_load = peak_by_scope;
    report.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::orchestration_timeline::{
        OrchestrationSegmentV1, RationalBeatV1, ORCHESTRATION_TIMELINE_SCHEMA_V1,
    };
    use crate::pitch::{Pitch, PitchClass};
    use crate::score::{Emphasis, ScoreNote, VoiceRole};

    fn digest() -> String { "a".repeat(64) }

    fn score() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        for (part, midi, onset, duration) in [
            (0u16, 60u8, 0i64, 4i64),
            (1, 48, 0, 4),
        ] {
            score.push(ScoreNote {
                part: PartId(part),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(duration, 1),
                velocity: 0.6,
                role: if part == 1 { VoiceRole::Bass } else { VoiceRole::Melody },
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    fn instrument() -> InstrumentCapabilityV1 {
        InstrumentCapabilityV1 {
            id: "piano-1".into(),
            kind: InstrumentKindV1::Piano,
            sounding_range: crate::orchestration::SoundingRangeV1 {
                lowest_midi: 21,
                highest_midi: 108,
            },
            max_polyphony: 16,
        }
    }

    fn segment(id: &str, part_id: u16) -> crate::orchestration_timeline::OrchestrationSegmentV1 {
        OrchestrationSegmentV1 {
            id: id.into(),
            part_id,
            instrument_id: "piano-1".into(),
            start: RationalBeatV1 { numerator: 0, denominator: 1 },
            end: RationalBeatV1 { numerator: 4, denominator: 1 },
            musical_function: "authored_line".into(),
            handoff: None,
        }
    }

    fn timeline() -> OrchestrationTimelineV1 {
        OrchestrationTimelineV1 {
            schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
            score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: digest(),
            segments: vec![segment("melody", 0), segment("bass", 1)],
        }
    }

    fn profile(id: &str, setup: &str, note_budget: u8, part_budget: u8) -> PerformerCapabilityV2 {
        PerformerCapabilityV2 {
            performer_id: id.into(),
            instrument_kind: InstrumentKindV1::Piano,
            setup_id: setup.into(),
            max_simultaneous_parts: part_budget,
            max_simultaneous_notes: note_budget,
            evidence_sha256: digest(),
            qualification: PerformerQualificationV2::Qualified,
        }
    }

    fn unit(id: &str, segment_id: &str, performer: &str,
        setup: &str, mode: PerformanceAllocationModeV2, group: Option<&str>) -> PerformanceUnitV2 {
        PerformanceUnitV2 {
            id: id.into(),
            orchestration_segment_id: segment_id.into(),
            performer_ids: vec![performer.into()],
            setup_id: setup.into(),
            mode,
            allocation_group_id: group.map(str::to_owned),
        }
    }

    fn plan(units: Vec<PerformanceUnitV2>, profiles: Vec<PerformerCapabilityV2>) -> PerformanceAllocationPlanV2 {
        PerformanceAllocationPlanV2 {
            schema: PERFORMANCE_ALLOCATION_SCHEMA_V2.into(),
            score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: digest(),
            units,
            performer_capabilities: profiles,
        }
    }

    fn validate(score: &Score, units: Vec<PerformanceUnitV2>, profiles: Vec<PerformerCapabilityV2>) -> PerformanceAllocationReportV2 {
        validate_performance_allocation_v2(score, &timeline(), &[instrument()], &plan(units, profiles), &digest())
    }

    #[test]
    fn accepts_one_pianist_carrying_two_parts_on_one_piano() {
        let report = validate(&score(), vec![
            unit("melody-unit", "melody", "pianist", "two-hand",
                PerformanceAllocationModeV2::SharedPerformerMultiPart, Some("hands")),
            unit("bass-unit", "bass", "pianist", "two-hand",
                PerformanceAllocationModeV2::SharedPerformerMultiPart, Some("hands")),
        ], vec![profile("pianist", "two-hand", 10, 2)]);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.performer_peak_load[0].peak_simultaneous_parts, 2);
        assert_eq!(report.performer_peak_load[0].peak_simultaneous_notes, 2);
    }

    #[test]
    fn accepts_two_performers_sharing_one_four_hand_piano() {
        let report = validate(&score(), vec![
            unit("left-player", "melody", "pianist-a", "four-hand",
                PerformanceAllocationModeV2::SharedInstrumentMultiPerformer, Some("four-hands")),
            unit("right-player", "bass", "pianist-b", "four-hand",
                PerformanceAllocationModeV2::SharedInstrumentMultiPerformer, Some("four-hands")),
        ], vec![
            profile("pianist-a", "four-hand", 8, 1),
            profile("pianist-b", "four-hand", 8, 1),
        ]);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.performer_count, 2);
    }

    #[test]
    fn accepts_one_part_realized_by_a_declared_unison_performer_group() {
        let mut solo_score = Score::new(Key::major(PitchClass::C), 100.0, 4);
        solo_score.push(ScoreNote {
            part: PartId(0),
            pitch: Pitch::from_midi(60),
            onset: Duration::zero(),
            duration: Duration::new(4, 1),
            velocity: 0.6,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        let one_segment = OrchestrationTimelineV1 {
            schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
            score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: digest(),
            segments: vec![segment("theme", 0)],
        };
        let mut group = unit("section", "theme", "player-a", "section-unison",
            PerformanceAllocationModeV2::EnsembleUnison, Some("strings-unison"));
        group.performer_ids.push("player-b".into());
        let report = validate_performance_allocation_v2(
            &solo_score, &one_segment, &[instrument()],
            &plan(vec![group], vec![
                profile("player-a", "section-unison", 4, 1),
                profile("player-b", "section-unison", 4, 1),
            ]),
            &digest(),
        );
        assert!(report.valid, "{:?}", report.issues);
    }

    #[test]
    fn performer_load_is_qualified_per_setup_not_from_another_sequential_setup() {
        let mut source = score();
        source.notes[0].duration = Duration::new(2, 1);
        source.notes[1].onset = Duration::new(2, 1);
        source.notes[1].duration = Duration::new(2, 1);
        let timed = OrchestrationTimelineV1 {
            schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
            score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: digest(),
            segments: vec![
                OrchestrationSegmentV1 {
                    id: "melody".into(),
                    part_id: 0,
                    instrument_id: "piano-1".into(),
                    start: RationalBeatV1 { numerator: 0, denominator: 1 },
                    end: RationalBeatV1 { numerator: 2, denominator: 1 },
                    musical_function: "melody".into(),
                    handoff: None,
                },
                OrchestrationSegmentV1 {
                    id: "bass".into(),
                    part_id: 1,
                    instrument_id: "piano-1".into(),
                    start: RationalBeatV1 { numerator: 2, denominator: 1 },
                    end: RationalBeatV1 { numerator: 4, denominator: 1 },
                    musical_function: "bass".into(),
                    handoff: None,
                },
            ],
        };
        let report = validate_performance_allocation_v2(
            &source,
            &timed,
            &[instrument()],
            &plan(
                vec![
                    unit("melody-unit", "melody", "performer", "two-hand",
                        PerformanceAllocationModeV2::Solo, None),
                    unit("bass-unit", "bass", "performer", "one-hand",
                        PerformanceAllocationModeV2::Solo, None),
                ],
                vec![
                    profile("performer", "two-hand", 8, 1),
                    profile("performer", "one-hand", 1, 1),
                ],
            ),
            &digest(),
        );
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.performer_peak_load.len(), 2);
        assert!(report.performer_peak_load.iter().any(|load| {
            load.setup_id == "two-hand" && load.declared_note_budget == 8
        }));
        assert!(report.performer_peak_load.iter().any(|load| {
            load.setup_id == "one-hand" && load.declared_note_budget == 1
        }));
    }

    #[test]
    fn rejects_one_performer_double_booked_on_different_instruments() {
        let mut two_instrument_timeline = timeline();
        two_instrument_timeline.segments[1].instrument_id = "second-piano".into();
        let report = validate_performance_allocation_v2(
            &score(),
            &two_instrument_timeline,
            &[instrument(), InstrumentCapabilityV1 {
                id: "second-piano".into(),
                ..instrument()
            }],
            &plan(vec![
                unit("melody-unit", "melody", "pianist", "two-hand",
                    PerformanceAllocationModeV2::Solo, None),
                unit("bass-unit", "bass", "pianist", "two-hand",
                    PerformanceAllocationModeV2::Solo, None),
            ], vec![profile("pianist", "two-hand", 10, 2)]),
            &digest(),
        );
        assert!(report.issues.iter().any(|issue| {
            issue.code == PerformanceAllocationIssueCodeV2::PerformerDoubleBooked
        }));
    }

    #[test]
    fn rejects_concurrent_physical_instrument_sharing_without_declared_group() {
        let report = validate(&score(), vec![
            unit("melody-unit", "melody", "pianist-a", "four-hand",
                PerformanceAllocationModeV2::Solo, None),
            unit("bass-unit", "bass", "pianist-b", "four-hand",
                PerformanceAllocationModeV2::Solo, None),
        ], vec![
            profile("pianist-a", "four-hand", 8, 1),
            profile("pianist-b", "four-hand", 8, 1),
        ]);
        assert!(report.issues.iter().any(|issue| {
            issue.code == PerformanceAllocationIssueCodeV2::SharedInstrumentWithoutDeclaredGroup
        }));
    }

    #[test]
    fn rejects_missing_or_unqualified_performer_evidence() {
        let report = validate(&score(), vec![
            unit("melody-unit", "melody", "pianist", "two-hand", PerformanceAllocationModeV2::Solo, None),
            unit("bass-unit", "bass", "unknown", "two-hand", PerformanceAllocationModeV2::Solo, None),
        ], vec![{
            let mut profile = profile("pianist", "two-hand", 10, 2);
            profile.qualification = PerformerQualificationV2::ResearchOnly;
            profile
        }]);
        assert!(report.issues.iter().any(|issue| {
            issue.code == PerformanceAllocationIssueCodeV2::UnqualifiedPerformerProfile
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == PerformanceAllocationIssueCodeV2::MissingPerformerProfile
        }));
    }

    #[test]
    fn rejects_performer_note_and_part_capacity_overcommitment() {
        let report = validate(&score(), vec![
            unit("melody-unit", "melody", "pianist", "two-hand",
                PerformanceAllocationModeV2::SharedPerformerMultiPart, Some("hands")),
            unit("bass-unit", "bass", "pianist", "two-hand",
                PerformanceAllocationModeV2::SharedPerformerMultiPart, Some("hands")),
        ], vec![profile("pianist", "two-hand", 1, 1)]);
        assert!(report.issues.iter().any(|issue| {
            issue.code == PerformanceAllocationIssueCodeV2::PerformerPolyphonyExceeded
        }));
        assert!(report.issues.iter().any(|issue| {
            issue.code == PerformanceAllocationIssueCodeV2::PerformerPartCapacityExceeded
        }));
    }
}

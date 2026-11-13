// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! ZA-CLICK-001: synthetic multi-source click articulatory gesture compiler.
//!
//! The output is a symbolic gesture timeline, not a voice recording and not
//! a physically faithful click waveform. It keeps velaric oral gestures
//! separate from optional pulmonic/laryngeal voicing, velic nasal coupling,
//! post-release aspiration and the vowel's independent tone/melody contour.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

pub const SCHEMA_VERSION: &str = "za-click-gesture-bench-1.0.0";
pub const CONTENT_CLASS: &str = "synthetic_symbolic_gesture_only";
pub const PROFILE_CLAIM: &str =
    "symbolic gesture timing/constraint validation only; no click acoustics or language competence claim";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ClickPlace {
    Dental,
    Postalveolar,
    Lateral,
}

impl ClickPlace {
    pub fn orthographic_hint(self) -> &'static str {
        match self {
            Self::Dental => "c",
            Self::Postalveolar => "q",
            Self::Lateral => "x",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Dental => "dental",
            Self::Postalveolar => "post-alveolar",
            Self::Lateral => "lateral",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AirstreamMechanism {
    VelaricIngressive,
    PulmonicEgressive,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccompanimentProfile {
    Plain,
    Aspirated,
    Voiced,
    Nasalized,
    Prenasalized,
    VoicedNasalized,
}

impl AccompanimentProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Aspirated => "aspirated",
            Self::Voiced => "voiced",
            Self::Nasalized => "nasalized",
            Self::Prenasalized => "prenasalized",
            Self::VoicedNasalized => "voiced_nasalized",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TimeIntervalMs {
    pub start: f64,
    pub end: f64,
}

impl TimeIntervalMs {
    pub fn duration(self) -> f64 {
        self.end - self.start
    }

    pub fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    pub fn overlap_duration(self, other: Self) -> f64 {
        (self.end.min(other.end) - self.start.max(other.start)).max(0.0)
    }

    pub fn contains(self, other: Self) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    pub fn is_valid(self) -> bool {
        self.start.is_finite() && self.end.is_finite() && self.start >= 0.0 && self.end > self.start
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TonePoint {
    pub time_ms: f64,
    pub f0_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VowelNucleus {
    pub event_id: String,
    pub interval: TimeIntervalMs,
    /// F0/tone belongs to the vowel nucleus, not to the click release.
    pub tone_contour: Vec<TonePoint>,
    /// Optional melodic score target; independent of the click transient.
    pub melody_note_hz: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClickGesture {
    pub event_id: String,
    pub source_case_id: String,
    pub place: ClickPlace,
    /// Alias is annotation metadata only; it is not inferred from arbitrary text.
    pub orthographic_hint: String,
    pub accompaniment: AccompanimentProfile,
    pub airstream: AirstreamMechanism,

    /// Two distinct closures must overlap in time.
    pub front_closure: TimeIntervalMs,
    pub posterior_closure: TimeIntervalMs,

    /// Tongue-body rarefaction occurs while the two closures are maintained.
    pub rarefaction: TimeIntervalMs,

    /// Front closure opens before the ingressive burst is measured.
    pub front_release: TimeIntervalMs,
    pub ingressive_burst: TimeIntervalMs,

    /// Optional source mechanisms: kept distinct from the velaric airstream.
    pub laryngeal_voicing: Option<TimeIntervalMs>,
    pub velic_nasal_coupling: Option<TimeIntervalMs>,
    pub post_release_aspiration: Option<TimeIntervalMs>,
    pub prenasal_lead: Option<TimeIntervalMs>,

    /// The vowel and its tone/melody are independent events.
    pub vowel: VowelNucleus,

    /// Explicitly synthetic geometry/timing policy identity.
    pub gesture_profile_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GestureViolation {
    pub code: String,
    pub field: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GestureValidation {
    pub accepted: bool,
    pub violations: Vec<GestureViolation>,
}

fn violation(code: &str, field: &str, message: impl Into<String>) -> GestureViolation {
    GestureViolation {
        code: code.into(),
        field: field.into(),
        message: message.into(),
    }
}

fn check_interval(
    value: Option<TimeIntervalMs>,
    field: &str,
    required: bool,
    violations: &mut Vec<GestureViolation>,
) {
    match value {
        Some(interval) if !interval.is_valid() => violations.push(violation(
            "INVALID_INTERVAL",
            field,
            "interval must have finite, non-negative start and strictly later end",
        )),
        None if required => violations.push(violation(
            "MISSING_REQUIRED_INTERVAL",
            field,
            "required gesture interval is absent",
        )),
        _ => {}
    }
}

/// Validate the gesture's structural and temporal contract.
///
/// Timing limits are symbolic fixture policy values, not measured universal
/// phonetic durations. The validator deliberately avoids asserting that a
/// structurally valid gesture is a correct native-language pronunciation.
pub fn validate_gesture(gesture: &ClickGesture) -> GestureValidation {
    let mut violations = Vec::new();

    if gesture.event_id.trim().is_empty() || gesture.source_case_id.trim().is_empty() {
        violations.push(violation(
            "MISSING_IDENTITY",
            "event_id/source_case_id",
            "stable event and source-case identities are required",
        ));
    }
    if gesture.gesture_profile_id.trim().is_empty() {
        violations.push(violation(
            "MISSING_PROFILE",
            "gesture_profile_id",
            "synthetic gesture profile identity is required",
        ));
    }
    if gesture.orthographic_hint != gesture.place.orthographic_hint() {
        violations.push(violation(
            "ORTHOGRAPHIC_METADATA_MISMATCH",
            "orthographic_hint",
            "explicit alias does not match the declared click-place hint",
        ));
    }
    if gesture.airstream != AirstreamMechanism::VelaricIngressive {
        violations.push(violation(
            "WRONG_CLICK_AIRSTREAM",
            "airstream",
            "click profile requires explicit velaric ingressive oral airstream",
        ));
    }

    check_interval(Some(gesture.front_closure), "front_closure", true, &mut violations);
    check_interval(
        Some(gesture.posterior_closure),
        "posterior_closure",
        true,
        &mut violations,
    );
    check_interval(Some(gesture.rarefaction), "rarefaction", true, &mut violations);
    check_interval(Some(gesture.front_release), "front_release", true, &mut violations);
    check_interval(
        Some(gesture.ingressive_burst),
        "ingressive_burst",
        true,
        &mut violations,
    );
    check_interval(
        gesture.laryngeal_voicing,
        "laryngeal_voicing",
        false,
        &mut violations,
    );
    check_interval(
        gesture.velic_nasal_coupling,
        "velic_nasal_coupling",
        false,
        &mut violations,
    );
    check_interval(
        gesture.post_release_aspiration,
        "post_release_aspiration",
        false,
        &mut violations,
    );
    check_interval(gesture.prenasal_lead, "prenasal_lead", false, &mut violations);
    check_interval(
        Some(gesture.vowel.interval),
        "vowel.interval",
        true,
        &mut violations,
    );

    let closure_overlap = gesture.front_closure.overlap_duration(gesture.posterior_closure);
    let shorter_closure = gesture
        .front_closure
        .duration()
        .min(gesture.posterior_closure.duration());
    if shorter_closure <= 0.0 || closure_overlap < shorter_closure * 0.50 {
        violations.push(violation(
            "INSUFFICIENT_DOUBLE_CLOSURE_OVERLAP",
            "front_closure/posterior_closure",
            "front and posterior closures must overlap for at least half the shorter closure in this fixture profile",
        ));
    }

    let closure_overlap_interval = TimeIntervalMs {
        start: gesture.front_closure.start.max(gesture.posterior_closure.start),
        end: gesture.front_closure.end.min(gesture.posterior_closure.end),
    };
    if closure_overlap_interval.is_valid()
        && !closure_overlap_interval.contains(gesture.rarefaction)
    {
        violations.push(violation(
            "RAREFACTION_OUTSIDE_DOUBLE_CLOSURE",
            "rarefaction",
            "rarefaction interval must lie inside the two-closure overlap",
        ));
    }

    if gesture.front_release.start < gesture.front_closure.end {
        violations.push(violation(
            "FRONT_RELEASE_BEFORE_CLOSURE_END",
            "front_release",
            "front release cannot begin before front closure ends",
        ));
    }
    if gesture.front_release.start < gesture.posterior_closure.end {
        violations.push(violation(
            "POSTERIOR_CLOSURE_RELEASE_ORDER",
            "front_release",
            "the synthetic v1 profile requires posterior closure to remain through front release onset",
        ));
    }
    if gesture.front_release.start < gesture.rarefaction.end {
        violations.push(violation(
            "RELEASE_BEFORE_RAREFACTION_COMPLETE",
            "front_release",
            "front release begins before rarefaction has completed",
        ));
    }
    if gesture.ingressive_burst.start < gesture.front_release.start
        || gesture.ingressive_burst.end <= gesture.front_release.end
    {
        violations.push(violation(
            "BURST_RELEASE_ORDER",
            "ingressive_burst",
            "ingressive burst must begin at/after front release onset and extend beyond its release interval",
        ));
    }

    let click_end = gesture.ingressive_burst.end;
    let vowel_start = gesture.vowel.interval.start;

    match gesture.accompaniment {
        AccompanimentProfile::Plain => {
            if gesture.post_release_aspiration.is_some()
                || gesture.laryngeal_voicing.is_some()
                || gesture.velic_nasal_coupling.is_some()
                || gesture.prenasal_lead.is_some()
            {
                violations.push(violation(
                    "PLAIN_ACCOMPANIMENT_CONTRADICTION",
                    "accompaniment",
                    "plain profile must not carry aspiration, voicing, nasal coupling or prenasal lead",
                ));
            }
        }
        AccompanimentProfile::Aspirated => {
            if gesture.post_release_aspiration.is_none() {
                violations.push(violation(
                    "MISSING_ASPIRATION",
                    "post_release_aspiration",
                    "aspirated profile requires an explicit post-release aspiration interval",
                ));
            }
            if gesture.laryngeal_voicing.is_some() || gesture.velic_nasal_coupling.is_some()
                || gesture.prenasal_lead.is_some()
            {
                violations.push(violation(
                    "ASPIRATED_ACCOMPANIMENT_CONTRADICTION",
                    "accompaniment",
                    "v1 aspirated profile is kept distinct from voiced/nasal/prenasalized profiles",
                ));
            }
        }
        AccompanimentProfile::Voiced => {
            if gesture.laryngeal_voicing.is_none() {
                violations.push(violation(
                    "MISSING_LARYNGEAL_VOICING",
                    "laryngeal_voicing",
                    "voiced accompaniment requires an explicit pulmonic/laryngeal voicing interval",
                ));
            }
            if gesture.velic_nasal_coupling.is_some() || gesture.prenasal_lead.is_some()
                || gesture.post_release_aspiration.is_some()
            {
                violations.push(violation(
                    "VOICED_ACCOMPANIMENT_CONTRADICTION",
                    "accompaniment",
                    "voiced-only profile must not be silently treated as nasalized or aspirated",
                ));
            }
        }
        AccompanimentProfile::Nasalized => {
            if gesture.velic_nasal_coupling.is_none() {
                violations.push(violation(
                    "MISSING_VELIC_NASAL_COUPLING",
                    "velic_nasal_coupling",
                    "nasalized profile requires an explicit velic coupling interval",
                ));
            }
            if gesture.prenasal_lead.is_some() || gesture.post_release_aspiration.is_some() {
                violations.push(violation(
                    "NASALIZED_ACCOMPANIMENT_CONTRADICTION",
                    "accompaniment",
                    "nasalized profile is distinct from prenasalized and aspirated profiles",
                ));
            }
        }
        AccompanimentProfile::Prenasalized => {
            if gesture.prenasal_lead.is_none() {
                violations.push(violation(
                    "MISSING_PRENASAL_LEAD",
                    "prenasal_lead",
                    "prenasalized profile requires an explicit nasal lead interval",
                ));
            }
            if gesture.velic_nasal_coupling.is_none() {
                violations.push(violation(
                    "MISSING_VELIC_NASAL_COUPLING",
                    "velic_nasal_coupling",
                    "prenasalized profile requires velic nasal coupling",
                ));
            }
            if gesture.post_release_aspiration.is_some() {
                violations.push(violation(
                    "PRENASALIZED_ASPIRATION_CONTRADICTION",
                    "accompaniment",
                    "v1 prenasalized profile is kept distinct from aspirated profile",
                ));
            }
        }
        AccompanimentProfile::VoicedNasalized => {
            if gesture.laryngeal_voicing.is_none() || gesture.velic_nasal_coupling.is_none() {
                violations.push(violation(
                    "MISSING_VOICED_NASAL_FEATURES",
                    "laryngeal_voicing/velic_nasal_coupling",
                    "voiced-nasalized profile requires independent laryngeal and velic intervals",
                ));
            }
            if gesture.post_release_aspiration.is_some() || gesture.prenasal_lead.is_some() {
                violations.push(violation(
                    "VOICED_NASALIZED_ACCOMPANIMENT_CONTRADICTION",
                    "accompaniment",
                    "v1 voiced-nasalized profile is distinct from aspirated and prenasalized profiles",
                ));
            }
        }
    }

    if let Some(voicing) = gesture.laryngeal_voicing {
        if !voicing.overlaps(gesture.ingressive_burst) {
            violations.push(violation(
                "VOICING_NOT_ALIGNED_TO_CLICK",
                "laryngeal_voicing",
                "voicing accompaniment must overlap the click transient in this profile",
            ));
        }
    }
    if let Some(nasal) = gesture.velic_nasal_coupling {
        if !nasal.overlaps(gesture.ingressive_burst) {
            violations.push(violation(
                "NASAL_COUPLING_NOT_ALIGNED_TO_CLICK",
                "velic_nasal_coupling",
                "velic coupling must overlap the click transient in this profile",
            ));
        }
    }
    if let Some(aspiration) = gesture.post_release_aspiration {
        if aspiration.start < click_end || aspiration.end > vowel_start {
            violations.push(violation(
                "ASPIRATION_OUTSIDE_RELEASE_VOWEL_GAP",
                "post_release_aspiration",
                "post-release aspiration must occur after the click burst and before the vowel nucleus",
            ));
        }
    }
    if let Some(nasal_lead) = gesture.prenasal_lead {
        if nasal_lead.start >= gesture.front_closure.start
            || nasal_lead.end < gesture.posterior_closure.start
            || nasal_lead.end > gesture.front_release.start
        {
            violations.push(violation(
                "PRENASAL_LEAD_TIMING",
                "prenasal_lead",
                "nasal lead must begin before click closures and continue into the closure phase without extending beyond release onset",
            ));
        }
    }
    if click_end > vowel_start {
        violations.push(violation(
            "CLICK_OVERLAPS_VOWEL_ONSET",
            "vowel.interval",
            "the v1 click transient must finish before the vowel nucleus begins",
        ));
    }

    if gesture.vowel.event_id.trim().is_empty() {
        violations.push(violation(
            "MISSING_VOWEL_ID",
            "vowel.event_id",
            "vowel nucleus must have a stable event identity",
        ));
    }
    if gesture.vowel.tone_contour.len() < 2 {
        violations.push(violation(
            "INSUFFICIENT_TONE_POINTS",
            "vowel.tone_contour",
            "tone/melody contour requires at least two control points",
        ));
    }
    let mut last_time = f64::NEG_INFINITY;
    for point in &gesture.vowel.tone_contour {
        if !point.time_ms.is_finite() || !point.f0_hz.is_finite() || point.f0_hz <= 0.0 {
            violations.push(violation(
                "INVALID_TONE_POINT",
                "vowel.tone_contour",
                "tone points require finite timestamps and positive F0 values",
            ));
        }
        if point.time_ms < last_time {
            violations.push(violation(
                "NON_MONOTONIC_TONE_CONTOUR",
                "vowel.tone_contour",
                "tone timestamps must be monotonic",
            ));
        }
        if point.time_ms < gesture.vowel.interval.start
            || point.time_ms > gesture.vowel.interval.end
        {
            violations.push(violation(
                "TONE_POINT_OUTSIDE_VOWEL",
                "vowel.tone_contour",
                "tone points must lie inside the vowel nucleus interval",
            ));
        }
        if point.time_ms >= gesture.front_release.start
            && point.time_ms <= gesture.ingressive_burst.end
        {
            violations.push(violation(
                "TONE_ATTACHED_TO_CLICK_TRANSIENT",
                "vowel.tone_contour",
                "melodic/tonal F0 must not be attached to the click-release transient",
            ));
        }
        last_time = point.time_ms;
    }
    if let Some(note_hz) = gesture.vowel.melody_note_hz {
        if !note_hz.is_finite() || note_hz <= 0.0 {
            violations.push(violation(
                "INVALID_MELODY_NOTE",
                "vowel.melody_note_hz",
                "melody target must be finite and positive",
            ));
        }
    }

    GestureValidation {
        accepted: violations.is_empty(),
        violations,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticGestureManifest {
    pub schema_version: String,
    pub content_class: String,
    pub human_recordings_ingested: bool,
    pub external_corpus_assets_used: bool,
    pub voice_identity_targeting: bool,
    pub source_revision: String,
    pub claim_boundary: String,
    pub orthographic_alias_policy: String,
    pub gestures: Vec<ClickGesture>,
    pub validation: Vec<GestureValidation>,
    pub manifest_sha256: String,
}

fn interval(start: f64, end: f64) -> TimeIntervalMs {
    TimeIntervalMs { start, end }
}

fn base_gesture(
    source: &str,
    event: &str,
    place: ClickPlace,
    accompaniment: AccompanimentProfile,
) -> ClickGesture {
    let (front_end, posterior_start, posterior_end, release_end, burst_end, vowel_start) =
        match place {
            ClickPlace::Dental => (90.0, 25.0, 88.0, 94.0, 100.0, 112.0),
            ClickPlace::Postalveolar => (96.0, 22.0, 94.0, 100.0, 106.0, 118.0),
            ClickPlace::Lateral => (102.0, 27.0, 100.0, 106.0, 112.0, 124.0),
        };
    let front_start = 20.0;
    let release_start = front_end;
    let burst_start = release_start;
    let vowel_end = vowel_start + 260.0;
    let is_voiced = matches!(
        accompaniment,
        AccompanimentProfile::Voiced | AccompanimentProfile::VoicedNasalized
    );
    let is_nasal = matches!(
        accompaniment,
        AccompanimentProfile::Nasalized
            | AccompanimentProfile::Prenasalized
            | AccompanimentProfile::VoicedNasalized
    );
    let is_aspirated = accompaniment == AccompanimentProfile::Aspirated;
    let is_prenasal = accompaniment == AccompanimentProfile::Prenasalized;
    let aspiration = is_aspirated.then(|| interval(burst_end, vowel_start - 1.0));
    let nasal_lead = is_prenasal.then(|| interval(0.0, posterior_start + 5.0));
    let laryngeal_voicing = is_voiced.then(|| interval(release_start - 2.0, vowel_start + 20.0));
    let velic_nasal_coupling = is_nasal.then(|| interval(posterior_start + 5.0, burst_end + 4.0));

    ClickGesture {
        event_id: event.into(),
        source_case_id: source.into(),
        place,
        orthographic_hint: place.orthographic_hint().into(),
        accompaniment,
        airstream: AirstreamMechanism::VelaricIngressive,
        front_closure: interval(front_start, front_end),
        posterior_closure: interval(posterior_start, posterior_end),
        rarefaction: interval(posterior_start + 15.0, posterior_end - 2.0),
        front_release: interval(release_start, release_end),
        ingressive_burst: interval(burst_start, burst_end),
        laryngeal_voicing,
        velic_nasal_coupling,
        post_release_aspiration: aspiration,
        prenasal_lead: nasal_lead,
        vowel: VowelNucleus {
            event_id: format!("{event}-vowel"),
            interval: interval(vowel_start, vowel_end),
            tone_contour: vec![
                TonePoint {
                    time_ms: vowel_start,
                    f0_hz: 220.0,
                },
                TonePoint {
                    time_ms: vowel_start + 130.0,
                    f0_hz: 246.94,
                },
                TonePoint {
                    time_ms: vowel_end,
                    f0_hz: 220.0,
                },
            ],
            melody_note_hz: Some(220.0),
        },
        gesture_profile_id: "za-click-synthetic-temporal-profile-v1".into(),
    }
}

pub fn known_answer_gestures() -> Vec<ClickGesture> {
    let mut gestures = Vec::new();
    let places = [
        ClickPlace::Dental,
        ClickPlace::Postalveolar,
        ClickPlace::Lateral,
    ];
    let profiles = [
        AccompanimentProfile::Plain,
        AccompanimentProfile::Aspirated,
        AccompanimentProfile::Voiced,
        AccompanimentProfile::Nasalized,
        AccompanimentProfile::Prenasalized,
        AccompanimentProfile::VoicedNasalized,
    ];
    for place in places {
        for profile in profiles {
            let case_id = format!("{}_{}", place.label().replace('-', "_"), profile.label().to_owned());
            gestures.push(base_gesture(
                "ZA_CLICK_CONFIRMATORY_SYNTH_V1",
                &case_id,
                place,
                profile,
            ));
        }
    }
    gestures
}

/// Render a compact text timeline for human inspection; it does not synthesize audio.
pub fn render_timeline(gesture: &ClickGesture) -> String {
    let mut events = vec![
        ("front_closure", gesture.front_closure),
        ("posterior_closure", gesture.posterior_closure),
        ("rarefaction", gesture.rarefaction),
        ("front_release", gesture.front_release),
        ("ingressive_burst", gesture.ingressive_burst),
        ("vowel", gesture.vowel.interval),
    ];
    for (name, value) in [
        ("laryngeal_voicing", gesture.laryngeal_voicing),
        ("velic_nasal_coupling", gesture.velic_nasal_coupling),
        ("post_release_aspiration", gesture.post_release_aspiration),
        ("prenasal_lead", gesture.prenasal_lead),
    ] {
        if let Some(interval) = value {
            events.push((name, interval));
        }
    }
    events.sort_by(|left, right| {
        left.1
            .start
            .total_cmp(&right.1.start)
            .then_with(|| left.0.cmp(right.0))
    });

    let mut output = format!(
        "{} / {} / {:?}\n",
        gesture.event_id, gesture.place.label(), gesture.accompaniment
    );
    for (name, event) in events {
        let _ = writeln!(
            output,
            "{name:24} {:>7.1} ms -> {:>7.1} ms",
            event.start, event.end
        );
    }
    output
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn build_manifest() -> Result<SyntheticGestureManifest, String> {
    let gestures = known_answer_gestures();
    let validation: Vec<GestureValidation> =
        gestures.iter().map(validate_gesture).collect();
    let raw = serde_json::to_vec(&(&gestures, &validation))
        .map_err(|error| format!("failed to serialize synthetic gesture fixture: {error}"))?;
    let manifest_sha256 = sha256_bytes(&raw);
    Ok(SyntheticGestureManifest {
        schema_version: SCHEMA_VERSION.into(),
        content_class: CONTENT_CLASS.into(),
        human_recordings_ingested: false,
        external_corpus_assets_used: false,
        voice_identity_targeting: false,
        source_revision: std::env::var("ZA_CLICK_SOURCE_REVISION")
            .unwrap_or_else(|_| "unknown-at-build".into()),
        claim_boundary: PROFILE_CLAIM.into(),
        orthographic_alias_policy:
            "c/q/x are conservative place hints for the generated fixture only; no general G2P coverage is claimed"
                .into(),
        gestures,
        validation,
        manifest_sha256,
    })
}

pub fn generate_fixture_bundle(output_dir: &std::path::Path) -> Result<SyntheticGestureManifest, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|error| format!("failed to create fixture output directory: {error}"))?;
    let manifest = build_manifest()?;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to serialize gesture manifest: {error}"))?;
    std::fs::write(output_dir.join("gesture-manifest.json"), &manifest_bytes)
        .map_err(|error| format!("failed to write gesture manifest: {error}"))?;
    std::fs::write(
        output_dir.join("gesture-manifest.sha256"),
        format!("{}  gesture-manifest.json\n", sha256_bytes(&manifest_bytes)),
    )
    .map_err(|error| format!("failed to write gesture manifest digest: {error}"))?;
    let mut timelines = String::new();
    for gesture in &manifest.gestures {
        timelines.push_str(&render_timeline(gesture));
        timelines.push('\n');
    }
    std::fs::write(output_dir.join("gesture-timelines.txt"), timelines)
        .map_err(|error| format!("failed to write gesture timelines: {error}"))?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_suite_covers_three_places_and_six_accompaniments() {
        let gestures = known_answer_gestures();
        assert_eq!(gestures.len(), 18);
        for place in [
            ClickPlace::Dental,
            ClickPlace::Postalveolar,
            ClickPlace::Lateral,
        ] {
            assert_eq!(
                gestures.iter().filter(|gesture| gesture.place == place).count(),
                6
            );
        }
        for profile in [
            AccompanimentProfile::Plain,
            AccompanimentProfile::Aspirated,
            AccompanimentProfile::Voiced,
            AccompanimentProfile::Nasalized,
            AccompanimentProfile::Prenasalized,
            AccompanimentProfile::VoicedNasalized,
        ] {
            assert_eq!(
                gestures
                    .iter()
                    .filter(|gesture| gesture.accompaniment == profile)
                    .count(),
                3
            );
        }
    }

    #[test]
    fn generated_known_answer_gestures_satisfy_the_symbolic_profile() {
        let manifest = build_manifest().expect("manifest");
        assert_eq!(manifest.validation.len(), 18);
        assert!(manifest.validation.iter().all(|result| result.accepted));
        assert_eq!(manifest.content_class, CONTENT_CLASS);
        assert!(!manifest.human_recordings_ingested);
        assert!(!manifest.external_corpus_assets_used);
        assert!(!manifest.voice_identity_targeting);
    }

    #[test]
    fn click_uses_velaric_ingressive_not_pulmonic_only_airstream() {
        let mut gesture = known_answer_gestures().remove(0);
        gesture.airstream = AirstreamMechanism::PulmonicEgressive;
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "WRONG_CLICK_AIRSTREAM"));
    }

    #[test]
    fn missing_posterior_closure_overlap_is_rejected() {
        let mut gesture = known_answer_gestures().remove(0);
        gesture.posterior_closure = interval(95.0, 110.0);
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result.violations.iter().any(|issue| {
            issue.code == "INSUFFICIENT_DOUBLE_CLOSURE_OVERLAP"
                || issue.code == "RAREFACTION_OUTSIDE_DOUBLE_CLOSURE"
        }));
    }

    #[test]
    fn rarefaction_must_remain_inside_the_double_closure() {
        let mut gesture = known_answer_gestures().remove(0);
        gesture.rarefaction = interval(10.0, 20.0);
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "RAREFACTION_OUTSIDE_DOUBLE_CLOSURE"));
    }

    #[test]
    fn pitch_and_tone_must_belong_to_vowel_not_click_transient() {
        let mut gesture = known_answer_gestures().remove(0);
        gesture.vowel.tone_contour[0].time_ms = gesture.front_release.start;
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "TONE_ATTACHED_TO_CLICK_TRANSIENT"));
    }

    #[test]
    fn aspirated_profile_requires_post_release_aspiration() {
        let mut gesture = known_answer_gestures()
            .into_iter()
            .find(|gesture| gesture.accompaniment == AccompanimentProfile::Aspirated)
            .expect("aspirated fixture");
        gesture.post_release_aspiration = None;
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "MISSING_ASPIRATION"));
    }

    #[test]
    fn prenasalized_profile_requires_a_separate_nasal_lead() {
        let mut gesture = known_answer_gestures()
            .into_iter()
            .find(|gesture| gesture.accompaniment == AccompanimentProfile::Prenasalized)
            .expect("prenasalized fixture");
        gesture.prenasal_lead = None;
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "MISSING_PRENASAL_LEAD"));
    }

    #[test]
    fn voicing_and_nasality_are_distinct_mechanisms() {
        let gesture = known_answer_gestures()
            .into_iter()
            .find(|gesture| gesture.accompaniment == AccompanimentProfile::VoicedNasalized)
            .expect("voiced nasalized");
        assert!(gesture.laryngeal_voicing.is_some());
        assert!(gesture.velic_nasal_coupling.is_some());

        let mut missing_nasal = gesture.clone();
        missing_nasal.velic_nasal_coupling = None;
        assert!(!validate_gesture(&missing_nasal).accepted);

        let mut missing_voice = gesture;
        missing_voice.laryngeal_voicing = None;
        assert!(!validate_gesture(&missing_voice).accepted);
    }

    #[test]
    fn tone_points_must_be_ordered_and_inside_the_vowel_interval() {
        let mut gesture = known_answer_gestures().remove(0);
        gesture.vowel.tone_contour.swap(0, 2);
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "NON_MONOTONIC_TONE_CONTOUR"));
    }

    #[test]
    fn invalid_click_id_and_profile_are_rejected() {
        let mut gesture = known_answer_gestures().remove(0);
        gesture.event_id.clear();
        gesture.gesture_profile_id.clear();
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "MISSING_IDENTITY"));
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "MISSING_PROFILE"));
    }

    #[test]
    fn wrong_orthographic_alias_is_a_metadata_error() {
        let mut gesture = known_answer_gestures().remove(0);
        gesture.orthographic_hint = "q".into();
        let result = validate_gesture(&gesture);
        assert!(!result.accepted);
        assert!(result
            .violations
            .iter()
            .any(|issue| issue.code == "ORTHOGRAPHIC_METADATA_MISMATCH"));
    }

    #[test]
    fn timeline_output_is_deterministic_and_contains_independent_sources() {
        let gesture = known_answer_gestures()
            .into_iter()
            .find(|gesture| gesture.accompaniment == AccompanimentProfile::VoicedNasalized)
            .expect("voiced-nasalized gesture");
        let first = render_timeline(&gesture);
        let second = render_timeline(&gesture);
        assert_eq!(first, second);
        assert!(first.contains("laryngeal_voicing"));
        assert!(first.contains("velic_nasal_coupling"));
        assert!(first.contains("posterior_closure"));
    }
}

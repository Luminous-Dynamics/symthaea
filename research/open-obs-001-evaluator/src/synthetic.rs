//! Deterministic synthetic corpus for OPEN-OBS-002.
//!
//! This is a benchmark fixture, not a physically calibrated telescope or radio
//! propagation model. It includes in-family positives, noise controls,
//! target-like interference, and held-out curved signals. Multiple fixed
//! baselines make both useful performance and predictable failure modes visible.

use super::{
    digest_hex, parse_manifest, Capture, DataKind, Decision, FamilyDecision, FamilyTruth,
    Partition, Truth, MANIFEST_HEADER,
};

pub const PROFILE_ID: &str = "open-obs-002-synthetic-track-v2";
pub const DETECTOR_THRESHOLD: f64 = 8.0;
pub const KNOWN_LINEAR_THRESHOLD: f64 = 18.0;
pub const PEAK_THRESHOLD: f64 = 0.20;
const FRAMES: usize = 64;
const BINS: usize = 128;
const NOISE_AMPLITUDE: f64 = 0.12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Baseline {
    MatchedTrack,
    PeakThreshold,
    AlwaysReject,
    AlwaysCandidate,
}

impl Baseline {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "matched-track" => Some(Self::MatchedTrack),
            "peak-threshold" => Some(Self::PeakThreshold),
            "always-reject" => Some(Self::AlwaysReject),
            "always-candidate" => Some(Self::AlwaysCandidate),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::MatchedTrack => "matched-track",
            Self::PeakThreshold => "peak-threshold",
            Self::AlwaysReject => "always-reject",
            Self::AlwaysCandidate => "always-candidate",
        }
    }

    fn profile_text(self) -> String {
        match self {
            Self::MatchedTrack => format!(
                "{PROFILE_ID};baseline=matched-track;candidate_slopes=[-1,0,1];candidate_threshold={DETECTOR_THRESHOLD};known_slopes=[-1,1];known_threshold={KNOWN_LINEAR_THRESHOLD};score=sum-per-frame"
            ),
            Self::PeakThreshold => format!(
                "{PROFILE_ID};baseline=peak-threshold;score=max-absolute-bin;candidate_threshold={PEAK_THRESHOLD};family_output=unknown"
            ),
            Self::AlwaysReject => format!("{PROFILE_ID};baseline=always-reject"),
            Self::AlwaysCandidate => format!("{PROFILE_ID};baseline=always-candidate;family_output=unknown"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    LinearDrift,
    CurvedTrack,
    NoiseOnly,
    TerrestrialMimic,
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn signed_unit(&mut self) -> f64 {
        let fraction = (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64);
        fraction * 2.0 - 1.0
    }
}

const INSTRUMENT_PROFILE: &str =
    "open-obs-002-flat-grid-v2;frames=64;bins=128;prng=SplitMix64-v1;additive_uniform_noise=[-0.12,0.12];serialization=f64-le-bits";

fn shape_name(shape: Shape) -> &'static str {
    match shape {
        Shape::LinearDrift => "linear_drift",
        Shape::CurvedTrack => "curved_track",
        Shape::NoiseOnly => "noise_only",
        Shape::TerrestrialMimic => "terrestrial_mimic",
    }
}

fn observation_with_background(
    seed: u64,
    shape: Shape,
    amplitude: f64,
) -> (Vec<f64>, String) {
    let mut rng = SplitMix64::new(seed);
    let mut grid = Vec::with_capacity(FRAMES * BINS);
    for _ in 0..(FRAMES * BINS) {
        grid.push(rng.signed_unit() * NOISE_AMPLITUDE);
    }

    // Hash the pristine background independently of any injected feature.
    let background_sha = digest_hex(&encode_observation(&grid));

    for frame in 0..FRAMES {
        let bin = match shape {
            Shape::LinearDrift => {
                if seed % 2 == 0 {
                    (12 + (seed % 12) as isize) + frame as isize
                } else {
                    (80 + (seed % 12) as isize) - frame as isize
                }
            }
            Shape::CurvedTrack => {
                let x = frame as f64 - 31.5;
                30 + (seed % 5) as isize + (0.06 * x * x).round() as isize
            }
            Shape::TerrestrialMimic => 50 + (seed % 10) as isize,
            Shape::NoiseOnly => continue,
        };

        if !(0..BINS as isize).contains(&bin) {
            continue;
        }
        grid[frame * BINS + bin as usize] += amplitude;
    }
    (grid, background_sha)
}

#[cfg(test)]
fn observation(seed: u64, shape: Shape, amplitude: f64) -> Vec<f64> {
    observation_with_background(seed, shape, amplitude).0
}

fn max_track_score(grid: &[f64], slopes: &[isize]) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for start in 0..BINS {
        for slope in slopes {
            let mut score = 0.0;
            let mut valid = true;
            for frame in 0..FRAMES {
                let bin = start as isize + *slope * frame as isize;
                if !(0..BINS as isize).contains(&bin) {
                    valid = false;
                    break;
                }
                score += grid[frame * BINS + bin as usize];
            }
            if valid && score > best {
                best = score;
            }
        }
    }
    best
}

fn detect(grid: &[f64], baseline: Baseline) -> DetectorOutput {
    match baseline {
        Baseline::MatchedTrack => {
            // Candidate screening permits stationary lines and one-bin/frame
            // drift. The known-family classifier recognizes strong non-stationary
            // tracks only. Structured interference can still be a false alarm.
            let any_track = max_track_score(grid, &[-1, 0, 1]);
            if any_track < DETECTOR_THRESHOLD {
                return DetectorOutput::rejected();
            }

            let known_linear_track = max_track_score(grid, &[-1, 1]);
            DetectorOutput {
                decision: Decision::Candidate,
                family_decision: if known_linear_track >= KNOWN_LINEAR_THRESHOLD {
                    FamilyDecision::Known
                } else {
                    FamilyDecision::Unknown
                },
            }
        }
        Baseline::PeakThreshold => {
            let peak = grid.iter().map(|value| value.abs()).fold(0.0, f64::max);
            if peak >= PEAK_THRESHOLD {
                DetectorOutput {
                    decision: Decision::Candidate,
                    family_decision: FamilyDecision::Unknown,
                }
            } else {
                DetectorOutput::rejected()
            }
        }
        Baseline::AlwaysReject => DetectorOutput::rejected(),
        Baseline::AlwaysCandidate => DetectorOutput {
            decision: Decision::Candidate,
            family_decision: FamilyDecision::Unknown,
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct DetectorOutput {
    decision: Decision,
    family_decision: FamilyDecision,
}

impl DetectorOutput {
    fn rejected() -> Self {
        Self {
            decision: Decision::Reject,
            family_decision: FamilyDecision::NotAssigned,
        }
    }
}

fn encode_observation(grid: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(grid.len() * 8);
    for value in grid {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn append_case(
    out: &mut String,
    counter: &mut u64,
    seed: u64,
    partition: Partition,
    stratum: &str,
    truth: Truth,
    family_truth: FamilyTruth,
    shape: Shape,
    amplitude: f64,
    baseline: Baseline,
) {
    let (grid, background_sha) = observation_with_background(seed, shape, amplitude);
    let encoded = encode_observation(&grid);
    let observation_sha = digest_hex(&encoded);
    let instrument_profile_sha = digest_hex(INSTRUMENT_PROFILE.as_bytes());
    let detector_profile_sha = digest_hex(baseline.profile_text().as_bytes());
    let scenario_profile_sha = digest_hex(
        format!(
            "{PROFILE_ID};shape={};amplitude={amplitude:.6};seed_schedule=sequential-v1",
            shape_name(shape)
        )
        .as_bytes(),
    );
    let case_id = format!("obs-{counter:05}-{observation_sha}");
    let background_group = format!("synthetic-acquisition-{counter:05}");
    let detected = detect(&grid, baseline);
    let family_decision = if detected.decision == Decision::Candidate {
        detected.family_decision
    } else {
        FamilyDecision::NotAssigned
    };

    out.push_str(&format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        case_id,
        background_group,
        partition.as_str(),
        stratum,
        truth.as_str(),
        detected.decision.as_str(),
        family_truth.as_str(),
        family_decision.as_str(),
        Capture::Complete.as_str(),
        DataKind::Simulated.as_str(),
        observation_sha,
        background_sha,
        instrument_profile_sha,
        scenario_profile_sha,
        detector_profile_sha
    ));
    *counter += 1;
}

pub fn generate_manifest() -> Result<Vec<u8>, String> {
    generate_manifest_with_baseline(Baseline::MatchedTrack)
}

pub fn generate_manifest_with_baseline(baseline: Baseline) -> Result<Vec<u8>, String> {
    let mut out = format!("{MANIFEST_HEADER}\n");
    let mut counter = 0_u64;
    let mut seed = 1_u64;

    // Small development/calibration cohorts are not qualification data.
    for partition in [Partition::Train, Partition::Calibration] {
        for _ in 0..20 {
            append_case(
                &mut out,
                &mut counter,
                seed,
                partition,
                "known_linear_track",
                Truth::Signal,
                FamilyTruth::Known,
                Shape::LinearDrift,
                0.40,
                baseline,
            );
            seed += 1;
            append_case(
                &mut out,
                &mut counter,
                seed,
                partition,
                "noise_control",
                Truth::NoSignal,
                FamilyTruth::NotApplicable,
                Shape::NoiseOnly,
                0.0,
                baseline,
            );
            seed += 1;
        }
    }

    // Frozen challenge panel: every row gets a separate synthetic background.
    for _ in 0..300 {
        append_case(
            &mut out,
            &mut counter,
            seed,
            Partition::Holdout,
            "known_linear_track",
            Truth::Signal,
            FamilyTruth::Known,
            Shape::LinearDrift,
            0.40,
            baseline,
        );
        seed += 1;
    }
    for _ in 0..300 {
        append_case(
            &mut out,
            &mut counter,
            seed,
            Partition::Holdout,
            "noise_control",
            Truth::NoSignal,
            FamilyTruth::NotApplicable,
            Shape::NoiseOnly,
            0.0,
            baseline,
        );
        seed += 1;
    }
    for _ in 0..80 {
        append_case(
            &mut out,
            &mut counter,
            seed,
            Partition::Holdout,
            "terrestrial_mimic_control",
            Truth::NoSignal,
            FamilyTruth::NotApplicable,
            Shape::TerrestrialMimic,
            0.25,
            baseline,
        );
        seed += 1;
    }
    for _ in 0..60 {
        append_case(
            &mut out,
            &mut counter,
            seed,
            Partition::Holdout,
            "held_out_curved_low_amplitude",
            Truth::Signal,
            FamilyTruth::HeldOut,
            Shape::CurvedTrack,
            0.35,
            baseline,
        );
        seed += 1;
    }
    for _ in 0..60 {
        append_case(
            &mut out,
            &mut counter,
            seed,
            Partition::Holdout,
            "held_out_curved_high_amplitude",
            Truth::Signal,
            FamilyTruth::HeldOut,
            Shape::CurvedTrack,
            2.0,
            baseline,
        );
        seed += 1;
    }

    // Validate the completed fixture before emitting it.
    parse_manifest(out.as_bytes())?;
    Ok(out.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paired::{
        assess_paired_bundle, PairedObservationBundle, PairedPanel, PairedStatus, PanelRole,
    };
    use crate::{evaluate, Gate};

    fn candidate_counts(manifest_bytes: &[u8], stratum: &str) -> (usize, usize) {
        let parsed = parse_manifest(manifest_bytes).expect("valid generated manifest");
        let rows: Vec<_> = parsed
            .cases
            .iter()
            .filter(|case| case.partition == Partition::Holdout && case.stratum == stratum)
            .collect();
        (
            rows.iter()
                .filter(|case| case.decision == Decision::Candidate)
                .count(),
            rows.len(),
        )
    }

    fn paired_synthetic_bundle(
        background_group: &str,
        target_seed: u64,
        target_shape: Shape,
        target_amplitude: f64,
        target_capture: Capture,
        control_seed: u64,
        control_shape: Shape,
        control_amplitude: f64,
        control_capture: Capture,
    ) -> PairedObservationBundle {
        let (target_grid, _) =
            observation_with_background(target_seed, target_shape, target_amplitude);
        let (control_grid, _) =
            observation_with_background(control_seed, control_shape, control_amplitude);
        let target_digest = digest_hex(&encode_observation(&target_grid));
        let control_digest = digest_hex(&encode_observation(&control_grid));
        PairedObservationBundle {
            background_group: background_group.to_owned(),
            instrument_profile_sha256: digest_hex(INSTRUMENT_PROFILE.as_bytes()),
            detector_profile_sha256: digest_hex(
                Baseline::MatchedTrack.profile_text().as_bytes(),
            ),
            panels: vec![
                PairedPanel {
                    role: PanelRole::Target,
                    capture: target_capture,
                    panel_sha256: target_digest,
                    candidate_detected: detect(&target_grid, Baseline::MatchedTrack).decision
                        == Decision::Candidate,
                },
                PairedPanel {
                    role: PanelRole::Control,
                    capture: control_capture,
                    panel_sha256: control_digest,
                    candidate_detected: detect(&control_grid, Baseline::MatchedTrack).decision
                        == Decision::Candidate,
                },
            ],
        }
    }

    #[test]
    fn generated_manifest_is_byte_deterministic_and_grouped() {
        let first = generate_manifest().expect("valid generated corpus");
        let second = generate_manifest().expect("valid generated corpus");
        assert_eq!(first, second);

        let parsed = parse_manifest(&first).expect("parser accepts generated corpus");
        assert_eq!(parsed.sha256, digest_hex(&first));
        assert_eq!(parsed.cases.len(), 880);
        assert!(parsed
            .cases
            .iter()
            .all(|case| case.data_kind == DataKind::Simulated));
        assert!(parsed.cases.iter().all(|case| {
            case.case_id.rsplit('-').next().is_some_and(|suffix| {
                suffix.len() == 64 && suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        }));
    }

    #[test]
    fn matched_track_baseline_exposes_false_alarms_and_shift_misses() {
        let bytes = generate_manifest().expect("generated corpus");
        assert_eq!(candidate_counts(&bytes, "known_linear_track"), (300, 300));
        assert_eq!(candidate_counts(&bytes, "noise_control"), (0, 300));
        assert_eq!(
            candidate_counts(&bytes, "terrestrial_mimic_control"),
            (80, 80)
        );
        assert_eq!(
            candidate_counts(&bytes, "held_out_curved_low_amplitude"),
            (0, 60)
        );
        assert_eq!(
            candidate_counts(&bytes, "held_out_curved_high_amplitude"),
            (60, 60)
        );

        let parsed = parse_manifest(&bytes).expect("valid manifest");
        assert!(parsed.cases.iter().filter(|case| {
            case.partition == Partition::Holdout
                && case.stratum == "held_out_curved_high_amplitude"
                && case.decision == Decision::Candidate
        }).all(|case| case.family_decision == FamilyDecision::Unknown));
    }

    #[test]
    fn peak_threshold_finds_injections_but_falsely_flags_terrestrial_mimics() {
        let bytes = generate_manifest_with_baseline(Baseline::PeakThreshold)
            .expect("generated peak-threshold corpus");
        assert_eq!(candidate_counts(&bytes, "known_linear_track"), (300, 300));
        assert_eq!(candidate_counts(&bytes, "noise_control"), (0, 300));
        assert_eq!(
            candidate_counts(&bytes, "terrestrial_mimic_control"),
            (80, 80)
        );
        assert_eq!(
            candidate_counts(&bytes, "held_out_curved_low_amplitude"),
            (60, 60)
        );
        assert_eq!(
            candidate_counts(&bytes, "held_out_curved_high_amplitude"),
            (60, 60)
        );
    }

    #[test]
    fn always_reject_and_always_candidate_controls_fail_different_gates() {
        let reject_bytes = generate_manifest_with_baseline(Baseline::AlwaysReject)
            .expect("generated reject-control corpus");
        let reject_manifest = parse_manifest(&reject_bytes).expect("valid reject corpus");
        let reject_report = evaluate(&reject_manifest, &"a".repeat(40));
        assert_eq!(reject_report.detection_gate, Gate::Fail);

        let accept_bytes = generate_manifest_with_baseline(Baseline::AlwaysCandidate)
            .expect("generated candidate-control corpus");
        let accept_manifest = parse_manifest(&accept_bytes).expect("valid candidate corpus");
        let accept_report = evaluate(&accept_manifest, &"b".repeat(40));
        assert_eq!(accept_report.false_positive_gate, Gate::Fail);
    }

    #[test]
    fn detector_baseline_changes_are_bound_to_a_different_profile_digest() {
        let matched = parse_manifest(
            &generate_manifest_with_baseline(Baseline::MatchedTrack).unwrap()
        ).unwrap();
        let peak = parse_manifest(
            &generate_manifest_with_baseline(Baseline::PeakThreshold).unwrap()
        ).unwrap();
        assert_ne!(
            matched.cases[0].detector_profile_sha256,
            peak.cases[0].detector_profile_sha256
        );
    }

    #[test]
    fn target_only_synthetic_signal_is_target_associated_not_an_origin_claim() {
        let input = paired_synthetic_bundle(
            "paired-acq-target-only",
            42,
            Shape::LinearDrift,
            0.40,
            Capture::Complete,
            43,
            Shape::NoiseOnly,
            0.0,
            Capture::Complete,
        );
        let result = assess_paired_bundle(&input).expect("valid paired synthetic bundle");
        assert_eq!(result.status, PairedStatus::TargetAssociatedCandidate);
    }

    #[test]
    fn same_stationary_interference_in_both_panels_is_not_target_specific() {
        // Both seeds are divisible by ten, so the stationary mimic uses the same
        // feature bin; independent noise keeps the panel byte identities distinct.
        let input = paired_synthetic_bundle(
            "paired-acq-shared-interference",
            2040,
            Shape::TerrestrialMimic,
            0.25,
            Capture::Complete,
            2050,
            Shape::TerrestrialMimic,
            0.25,
            Capture::Complete,
        );
        let result = assess_paired_bundle(&input).expect("valid paired synthetic bundle");
        assert_eq!(result.status, PairedStatus::CandidateAlsoInControl);
    }

    #[test]
    fn control_only_synthetic_transient_is_not_target_associated() {
        let input = paired_synthetic_bundle(
            "paired-acq-control-only",
            43,
            Shape::NoiseOnly,
            0.0,
            Capture::Complete,
            2040,
            Shape::TerrestrialMimic,
            0.25,
            Capture::Complete,
        );
        let result = assess_paired_bundle(&input).expect("valid paired synthetic bundle");
        assert_eq!(result.status, PairedStatus::ControlOnlyCandidate);
    }

    #[test]
    fn missing_control_capture_blocks_paired_attribution() {
        let input = paired_synthetic_bundle(
            "paired-acq-missing-control",
            42,
            Shape::LinearDrift,
            0.40,
            Capture::Complete,
            43,
            Shape::NoiseOnly,
            0.0,
            Capture::Incomplete,
        );
        let result = assess_paired_bundle(&input).expect("valid paired synthetic bundle");
        assert_eq!(result.status, PairedStatus::UnknownIncompleteCapture);
    }

    #[test]
    fn unit_seeded_examples_match_frozen_behavior() {
        let positive = observation(42, Shape::LinearDrift, 0.40);
        let quiet = observation(43, Shape::NoiseOnly, 0.0);
        let mimic = observation(44, Shape::TerrestrialMimic, 0.25);
        let curve = observation(46, Shape::CurvedTrack, 2.0);

        assert_eq!(detect(&positive, Baseline::MatchedTrack).decision, Decision::Candidate);
        assert_eq!(detect(&positive, Baseline::MatchedTrack).family_decision, FamilyDecision::Known);
        assert_eq!(detect(&quiet, Baseline::MatchedTrack).decision, Decision::Reject);
        assert_eq!(detect(&mimic, Baseline::MatchedTrack).family_decision, FamilyDecision::Unknown);
        assert_eq!(detect(&curve, Baseline::MatchedTrack).decision, Decision::Candidate);
        assert_eq!(detect(&curve, Baseline::MatchedTrack).family_decision, FamilyDecision::Unknown);
    }
}

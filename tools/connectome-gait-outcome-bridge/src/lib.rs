//! Source-bound endpoint/metric compatibility and coverage oracle for fly
//! connectome locomotion results.
//!
//! This crate intentionally does not extract gait events, filter kinematics,
//! calculate Hilbert phases, compute TCS/PLV or analyse neural signals. It
//! consumes separately versioned outputs from Symtropy gait metrics and
//! Symthaea phase-response transfer, then prevents endpoint substitution.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const LEG_COUNT: usize = 6;
const PAIR_COUNT: usize = 15;
const TRIPOD_GROUP_COUNT: usize = 2;
const PHASE_BINS: u8 = 8;
const MAGNITUDE_BINS: u8 = 16;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LegId {
    LeftFore,
    LeftMiddle,
    LeftHind,
    RightFore,
    RightMiddle,
    RightHind,
}

impl LegId {
    pub const ALL: [Self; LEG_COUNT] = [
        Self::LeftFore,
        Self::LeftMiddle,
        Self::LeftHind,
        Self::RightFore,
        Self::RightMiddle,
        Self::RightHind,
    ];
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LegPair {
    pub first: LegId,
    pub second: LegId,
}

impl LegPair {
    pub fn new(left: LegId, right: LegId) -> Result<Self, BridgeError> {
        if left == right {
            return Err(BridgeError::new(ReasonCode::SelfPair));
        }
        let (first, second) = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        Ok(Self { first, second })
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TripodGroup {
    LeftForeRightMiddleLeftHind,
    RightForeLeftMiddleRightHind,
}

impl TripodGroup {
    pub const ALL: [Self; TRIPOD_GROUP_COUNT] = [
        Self::LeftForeRightMiddleLeftHind,
        Self::RightForeLeftMiddleRightHind,
    ];
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Condition {
    BallContact,
    SlipperySurface,
    AirStepping,
    AirFeCoSilenced,
    OtherControl,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Stimulus {
    DNg100,
    DNg97,
    DNb08,
    NoStimulation,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SourceAccess {
    Synthetic,
    PermissionVerified,
    PermissionUnverified,
    Restricted,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EndpointKind {
    LocalJointRhythmicity,
    PerLegStepping,
    PairwiseInterlegCoupling,
    TripodCoordination,
    PhaseResetTransfer,
    EmbodiedBehaviour,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MetricId {
    PuglieseAutocorrelationOscillationScore,
    PuglieseAutocorrelationFrequency,
    PuglieseSingleLegStepFrequency,
    GenericPhaseVectorStrength,
    SapkalButterworthHilbertPlvV2,
    TcsNatvar2024,
    TcsSapkal2026V2,
    PhaseResetFractionalMicroturnsV1,
    PhaseResetCycleSlipsV1,
    FicTracForwardVelocity,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Unit {
    Score0To1,
    CyclesPerSample,
    Hertz,
    PhaseVectorStrength,
    Ratio,
    Microturns,
    CycleSlips,
    MillimetresPerSecond,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum AggregationLevel {
    Frame,
    Event,
    Trial,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Availability {
    Measured,
    NotAvailable,
    NotAssessed,
    FailedAssay,
    MissingTrack,
    OutOfSupport,
    NotEstimable,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ProbeClass {
    LocalPhaseKick,
    LoadOrContact,
    ProprioceptiveChannel,
    DescendingCommand,
    MechanicalBodyOrSubstrate,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TransferProbeKey {
    /// None denotes a declared global input; Some denotes a leg-local source.
    pub sender: Option<LegId>,
    pub probe_class: ProbeClass,
    pub recipient: LegId,
    pub phase_bin: u8,
    pub magnitude_bin: u8,
    pub delay_index: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementSourceBinding {
    pub dataset_id: String,
    pub dataset_revision_sha256: String,
    pub raw_trajectory_sha256: String,
    pub animal_id: String,
    pub session_id: String,
    pub trial_id: String,
    pub condition: Condition,
    pub stimulus: Stimulus,
}

#[derive(Clone, Debug)]
pub struct Measurement {
    pub measurement_id: String,
    pub source_binding: MeasurementSourceBinding,
    pub endpoint: EndpointKind,
    pub metric_id: MetricId,
    pub availability: Availability,
    pub unit: Unit,
    pub value: Option<f64>,
    pub aggregation: AggregationLevel,
    pub method_sha256: String,
    pub preprocessing_sha256: Option<String>,
    pub phase_estimator_sha256: Option<String>,
    pub channel_id: Option<String>,
    pub leg: Option<LegId>,
    pub pair: Option<LegPair>,
    pub tripod_group: Option<TripodGroup>,
    pub transfer_probe: Option<TransferProbeKey>,
    /// Exact Symtropy TCS output receipts include numerator and denominator
    /// ticks; they are absent for non-TCS measurements.
    pub numerator_ticks: Option<u64>,
    pub denominator_ticks: Option<u64>,
    pub sample_count: Option<u64>,
    pub support_count: Option<u64>,
    pub window_start_frame: Option<u64>,
    pub window_end_frame: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct GaitOutcomePacket {
    pub packet_id: String,
    pub dataset_id: String,
    pub dataset_revision_sha256: String,
    pub raw_trajectory_sha256: String,
    pub source_code_sha256: String,
    pub analysis_pipeline_sha256: String,
    pub source_access: SourceAccess,
    pub animal_id: String,
    pub session_id: String,
    pub trial_id: String,
    pub condition: Condition,
    pub stimulus: Stimulus,
    pub sampling_hz_numerator: u32,
    pub sampling_hz_denominator: u32,
    pub frame_count: u64,
    /// Whether the selected source materialization actually carries FicTrac
    /// columns. False is not zero forward velocity.
    pub fictrac_columns_available: bool,
    /// Declared number of probe cells in the intended phase-reset transfer
    /// grid for this packet. Zero means no transfer design was declared.
    pub expected_transfer_probe_count: u32,
    pub expected_transfer_grid_sha256: Option<String>,
    pub measurements: Vec<Measurement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoverageState {
    NotReported,
    NotAvailable,
    NoneMeasured,
    Partial,
    Complete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SummaryState {
    NoRhythmOrInterlegData,
    RhythmOnly,
    InterlegCoveragePartial,
    InterlegCoverageComplete,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetricCoverage {
    pub metric_id: MetricId,
    pub method_sha256: String,
    pub preprocessing_sha256: Option<String>,
    pub phase_estimator_sha256: Option<String>,
    pub measured: usize,
    pub required: usize,
    pub state: CoverageState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BridgeReport {
    pub packet_id: String,
    pub dataset_id: String,
    pub dataset_revision_sha256: String,
    pub raw_trajectory_sha256: String,
    pub source_code_sha256: String,
    pub analysis_pipeline_sha256: String,
    pub source_access: SourceAccess,
    pub animal_id: String,
    pub session_id: String,
    pub trial_id: String,
    pub condition: Condition,
    pub stimulus: Stimulus,
    pub local_rhythm_measured_legs: usize,
    pub local_rhythm_state: CoverageState,
    pub per_leg_stepping_measured_legs: usize,
    pub per_leg_stepping_state: CoverageState,
    pub pairwise_coverage: Vec<MetricCoverage>,
    pub tripod_coverage: Vec<MetricCoverage>,
    pub phase_reset_measured_probe_cells: usize,
    pub phase_reset_expected_probe_cells: usize,
    pub phase_reset_state: CoverageState,
    pub fictrac_velocity_state: CoverageState,
    pub summary: SummaryState,
    /// Always false in this adapter: endpoint coverage is not a gait,
    /// mechanism, causality or biological-validity certificate.
    pub emits_global_gait_certification: bool,
    pub upstream_metric_validity: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ReasonCode {
    MissingIdentity,
    InvalidDigest,
    MixedSourceIdentity,
    ZeroSamplingRate,
    ZeroFrameCount,
    InvalidTransferGridCommitment,
    DuplicateMeasurementId,
    MixedAggregationLevelFrame,
    InvalidWindow,
    WindowOutsideTrial,
    InvalidSampleCount,
    InvalidSupportCount,
    MeasuredValueMissing,
    NonMeasuredValuePresent,
    NonFiniteValue,
    NegativeValue,
    InvalidMetricEndpoint,
    InvalidMetricUnit,
    MissingLegIdentity,
    UnexpectedLegIdentity,
    MissingPairIdentity,
    UnexpectedPairIdentity,
    SelfPair,
    NonCanonicalPair,
    MissingTripodIdentity,
    UnexpectedTripodIdentity,
    MissingTransferProbe,
    UnexpectedTransferProbe,
    InvalidPhaseBin,
    InvalidMagnitudeBin,
    MissingMethodDigest,
    MissingPreprocessingDigest,
    MissingPhaseEstimatorDigest,
    MissingChannelIdentity,
    MissingTcsNumeratorDenominator,
    ZeroTcsDenominator,
    TcsValueMismatch,
    NatvarTcsOutOfRange,
    PhaseVectorStrengthOutOfRange,
    WrappedPhaseOutOfRange,
    CycleSlipNotInteger,
    FicTracUnavailableButMeasured,
    FicTracMissingWhenDeclaredAvailable,
    DuplicateEndpointKey,
    MixedTcsMetricDefinitions,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BridgeError {
    pub reasons: Vec<ReasonCode>,
}

impl BridgeError {
    fn new(reason: ReasonCode) -> Self {
        Self {
            reasons: vec![reason],
        }
    }

    fn contains(&self, reason: ReasonCode) -> bool {
        self.reasons.contains(&reason)
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "gait outcome bridge rejected packet: {:?}",
            self.reasons
        )
    }
}

impl Error for BridgeError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn measured(measurement: &Measurement) -> bool {
    measurement.availability == Availability::Measured
}

fn coverage_state(measured: usize, required: usize) -> CoverageState {
    if measured == 0 {
        CoverageState::NoneMeasured
    } else if measured >= required {
        CoverageState::Complete
    } else {
        CoverageState::Partial
    }
}

fn expected_metric(endpoint: EndpointKind, metric: MetricId) -> bool {
    matches!(
        (endpoint, metric),
        (
            EndpointKind::LocalJointRhythmicity,
            MetricId::PuglieseAutocorrelationOscillationScore
                | MetricId::PuglieseAutocorrelationFrequency
        ) | (
            EndpointKind::PerLegStepping,
            MetricId::PuglieseSingleLegStepFrequency
        ) | (
            EndpointKind::PairwiseInterlegCoupling,
            MetricId::GenericPhaseVectorStrength | MetricId::SapkalButterworthHilbertPlvV2
        ) | (
            EndpointKind::TripodCoordination,
            MetricId::TcsNatvar2024 | MetricId::TcsSapkal2026V2
        ) | (
            EndpointKind::PhaseResetTransfer,
            MetricId::PhaseResetFractionalMicroturnsV1 | MetricId::PhaseResetCycleSlipsV1
        ) | (
            EndpointKind::EmbodiedBehaviour,
            MetricId::FicTracForwardVelocity
        )
    )
}

fn expected_unit(metric: MetricId, unit: Unit) -> bool {
    matches!(
        (metric, unit),
        (
            MetricId::PuglieseAutocorrelationOscillationScore,
            Unit::Score0To1
        ) | (
            MetricId::PuglieseAutocorrelationFrequency,
            Unit::CyclesPerSample | Unit::Hertz
        ) | (MetricId::PuglieseSingleLegStepFrequency, Unit::Hertz)
            | (
                MetricId::GenericPhaseVectorStrength | MetricId::SapkalButterworthHilbertPlvV2,
                Unit::PhaseVectorStrength
            )
            | (
                MetricId::TcsNatvar2024 | MetricId::TcsSapkal2026V2,
                Unit::Ratio
            )
            | (MetricId::PhaseResetFractionalMicroturnsV1, Unit::Microturns)
            | (MetricId::PhaseResetCycleSlipsV1, Unit::CycleSlips)
            | (MetricId::FicTracForwardVelocity, Unit::MillimetresPerSecond)
    )
}

fn validate_measurement(
    packet: &GaitOutcomePacket,
    measurement: &Measurement,
) -> Result<(), BridgeError> {
    let mut reasons = Vec::new();
    let mut add = |condition: bool, code| {
        if condition {
            reasons.push(code);
        }
    };

    add(
        !valid_sha256(&measurement.method_sha256),
        ReasonCode::MissingMethodDigest,
    );
    if let Some(value) = &measurement.preprocessing_sha256 {
        add(!valid_sha256(value), ReasonCode::InvalidDigest);
    }
    if let Some(value) = &measurement.phase_estimator_sha256 {
        add(!valid_sha256(value), ReasonCode::InvalidDigest);
    }
    add(
        measurement.aggregation == AggregationLevel::Frame,
        ReasonCode::MixedAggregationLevelFrame,
    );
    add(
        !expected_metric(measurement.endpoint, measurement.metric_id),
        ReasonCode::InvalidMetricEndpoint,
    );
    add(
        !expected_unit(measurement.metric_id, measurement.unit),
        ReasonCode::InvalidMetricUnit,
    );
    add(
        !is_nonempty(&measurement.measurement_id),
        ReasonCode::MissingIdentity,
    );
    add(
        measurement.window_start_frame.is_none() || measurement.window_end_frame.is_none(),
        ReasonCode::InvalidWindow,
    );

    if let (Some(start), Some(end)) = (measurement.window_start_frame, measurement.window_end_frame)
    {
        add(start >= end, ReasonCode::InvalidWindow);
        add(end > packet.frame_count, ReasonCode::WindowOutsideTrial);
    }

    if measured(measurement) {
        add(
            measurement.value.is_none(),
            ReasonCode::MeasuredValueMissing,
        );
        if let Some(value) = measurement.value {
            add(!value.is_finite(), ReasonCode::NonFiniteValue);
            match measurement.metric_id {
                MetricId::TcsNatvar2024
                | MetricId::TcsSapkal2026V2
                | MetricId::PuglieseAutocorrelationOscillationScore
                | MetricId::PuglieseAutocorrelationFrequency
                | MetricId::PuglieseSingleLegStepFrequency
                | MetricId::GenericPhaseVectorStrength
                | MetricId::SapkalButterworthHilbertPlvV2
                | MetricId::FicTracForwardVelocity => {
                    add(value < 0.0, ReasonCode::NegativeValue);
                }
                MetricId::PhaseResetFractionalMicroturnsV1 | MetricId::PhaseResetCycleSlipsV1 => {}
            }
            match measurement.metric_id {
                MetricId::PuglieseAutocorrelationOscillationScore
                | MetricId::GenericPhaseVectorStrength
                | MetricId::SapkalButterworthHilbertPlvV2 => {
                    add(
                        !(0.0..=1.0).contains(&value),
                        ReasonCode::PhaseVectorStrengthOutOfRange,
                    );
                }
                MetricId::TcsNatvar2024 => {
                    add(
                        !(0.0..=1.0).contains(&value),
                        ReasonCode::NatvarTcsOutOfRange,
                    );
                }
                MetricId::PhaseResetFractionalMicroturnsV1 => {
                    add(
                        !(-500_000.0..500_000.0).contains(&value),
                        ReasonCode::WrappedPhaseOutOfRange,
                    );
                }
                MetricId::PhaseResetCycleSlipsV1 => {
                    add(value.fract() != 0.0, ReasonCode::CycleSlipNotInteger);
                }
                _ => {}
            }
        }
    } else {
        add(
            measurement.value.is_some(),
            ReasonCode::NonMeasuredValuePresent,
        );
    }

    match measurement.endpoint {
        EndpointKind::LocalJointRhythmicity => {
            add(measurement.leg.is_none(), ReasonCode::MissingLegIdentity);
            add(
                measurement.pair.is_some(),
                ReasonCode::UnexpectedPairIdentity,
            );
            add(
                measurement.tripod_group.is_some(),
                ReasonCode::UnexpectedTripodIdentity,
            );
            add(
                measurement.transfer_probe.is_some(),
                ReasonCode::UnexpectedTransferProbe,
            );
            add(
                measurement
                    .channel_id
                    .as_deref()
                    .is_none_or(|v| !is_nonempty(v)),
                ReasonCode::MissingChannelIdentity,
            );
            add(
                measurement.preprocessing_sha256.is_none(),
                ReasonCode::MissingPreprocessingDigest,
            );
            add(
                measurement.sample_count.is_none(),
                ReasonCode::InvalidSampleCount,
            );
        }
        EndpointKind::PerLegStepping => {
            add(measurement.leg.is_none(), ReasonCode::MissingLegIdentity);
            add(
                measurement.preprocessing_sha256.is_none(),
                ReasonCode::MissingPreprocessingDigest,
            );
            add(
                measurement.pair.is_some(),
                ReasonCode::UnexpectedPairIdentity,
            );
            add(
                measurement.tripod_group.is_some(),
                ReasonCode::UnexpectedTripodIdentity,
            );
            add(
                measurement.transfer_probe.is_some(),
                ReasonCode::UnexpectedTransferProbe,
            );
            add(
                measurement
                    .channel_id
                    .as_deref()
                    .is_none_or(|v| !is_nonempty(v)),
                ReasonCode::MissingChannelIdentity,
            );
            add(
                measurement.sample_count.is_none(),
                ReasonCode::InvalidSampleCount,
            );
        }
        EndpointKind::PairwiseInterlegCoupling => {
            add(measurement.pair.is_none(), ReasonCode::MissingPairIdentity);
            add(measurement.leg.is_some(), ReasonCode::UnexpectedLegIdentity);
            add(measurement.tripod_group.is_some(), ReasonCode::UnexpectedTripodIdentity);
            add(measurement.transfer_probe.is_some(), ReasonCode::UnexpectedTransferProbe);
            add(measurement.sample_count.is_none(), ReasonCode::InvalidSampleCount);
            add(
                measurement.preprocessing_sha256.is_none(),
                ReasonCode::MissingPreprocessingDigest,
            );
            add(
                measurement.phase_estimator_sha256.is_none(),
                ReasonCode::MissingPhaseEstimatorDigest,
            );
            if let Some(pair) = measurement.pair {
                add(pair.first >= pair.second, ReasonCode::NonCanonicalPair);
            }
        }
        EndpointKind::TripodCoordination => {
            add(
                measurement.tripod_group.is_none(),
                ReasonCode::MissingTripodIdentity,
            );
            add(
                measurement.preprocessing_sha256.is_none(),
                ReasonCode::MissingPreprocessingDigest,
            );
            add(
                measurement.aggregation != AggregationLevel::Trial,
                ReasonCode::MixedAggregationLevelFrame,
            );
            add(
                measurement.support_count.is_none(),
                ReasonCode::InvalidSupportCount,
            );
            add(measurement.leg.is_some(), ReasonCode::UnexpectedLegIdentity);
            add(
                measurement.pair.is_some(),
                ReasonCode::UnexpectedPairIdentity,
            );
            add(
                measurement.transfer_probe.is_some(),
                ReasonCode::UnexpectedTransferProbe,
            );
            if measured(measurement) {
                add(
                    measurement.numerator_ticks.is_none()
                        || measurement.denominator_ticks.is_none(),
                    ReasonCode::MissingTcsNumeratorDenominator,
                );
                if let (Some(numerator), Some(denominator), Some(value)) = (
                    measurement.numerator_ticks,
                    measurement.denominator_ticks,
                    measurement.value,
                ) {
                    add(denominator == 0, ReasonCode::ZeroTcsDenominator);
                    if denominator > 0 {
                        let expected = numerator as f64 / denominator as f64;
                        add(
                            (expected - value).abs() > 1e-12,
                            ReasonCode::TcsValueMismatch,
                        );
                    }
                }
            } else if measurement.availability == Availability::NotEstimable {
                add(
                    measurement.denominator_ticks != Some(0),
                    ReasonCode::ZeroTcsDenominator,
                );
            }
        }
        EndpointKind::PhaseResetTransfer => {
            add(
                measurement.transfer_probe.is_none(),
                ReasonCode::MissingTransferProbe,
            );
            add(
                measurement.preprocessing_sha256.is_none(),
                ReasonCode::MissingPreprocessingDigest,
            );
            add(
                measurement.phase_estimator_sha256.is_none(),
                ReasonCode::MissingPhaseEstimatorDigest,
            );
            add(measurement.leg.is_some(), ReasonCode::UnexpectedLegIdentity);
            add(
                measurement.pair.is_some(),
                ReasonCode::UnexpectedPairIdentity,
            );
            add(
                measurement.tripod_group.is_some(),
                ReasonCode::UnexpectedTripodIdentity,
            );
            if let Some(key) = measurement.transfer_probe {
                add(key.phase_bin >= PHASE_BINS, ReasonCode::InvalidPhaseBin);
                add(
                    key.magnitude_bin >= MAGNITUDE_BINS,
                    ReasonCode::InvalidMagnitudeBin,
                );
                match key.probe_class {
                    ProbeClass::LocalPhaseKick => {
                        add(key.sender.is_none(), ReasonCode::MissingLegIdentity);
                    }
                    ProbeClass::DescendingCommand => {
                        add(key.sender.is_some(), ReasonCode::UnexpectedLegIdentity);
                        add(
                            !matches!(
                                packet.stimulus,
                                Stimulus::DNg100 | Stimulus::DNg97 | Stimulus::DNb08
                            ),
                            ReasonCode::InvalidMetricEndpoint,
                        );
                    }
                    ProbeClass::LoadOrContact
                    | ProbeClass::ProprioceptiveChannel
                    | ProbeClass::MechanicalBodyOrSubstrate => {}
                }
            }
        }
        EndpointKind::EmbodiedBehaviour => {
            add(measurement.leg.is_some(), ReasonCode::UnexpectedLegIdentity);
            add(
                measurement.preprocessing_sha256.is_none(),
                ReasonCode::MissingPreprocessingDigest,
            );
            add(
                measurement.pair.is_some(),
                ReasonCode::UnexpectedPairIdentity,
            );
            add(
                measurement.tripod_group.is_some(),
                ReasonCode::UnexpectedTripodIdentity,
            );
            add(
                measurement.transfer_probe.is_some(),
                ReasonCode::UnexpectedTransferProbe,
            );
            if !packet.fictrac_columns_available {
                add(
                    measured(measurement),
                    ReasonCode::FicTracUnavailableButMeasured,
                );
                add(
                    measurement.availability != Availability::NotAvailable,
                    ReasonCode::FicTracUnavailableButMeasured,
                );
            }
        }
    }

    if let Some(sample_count) = measurement.sample_count {
        add(sample_count == 0, ReasonCode::InvalidSampleCount);
    }
    if let Some(support_count) = measurement.support_count {
        add(support_count == 0, ReasonCode::InvalidSupportCount);
    }

    if reasons.is_empty() {
        Ok(())
    } else {
        reasons.sort();
        reasons.dedup();
        Err(BridgeError { reasons })
    }
}

fn validate_packet(packet: &GaitOutcomePacket) -> Result<(), BridgeError> {
    let mut reasons = Vec::new();
    for value in [
        packet.packet_id.as_str(),
        packet.dataset_id.as_str(),
        packet.animal_id.as_str(),
        packet.session_id.as_str(),
        packet.trial_id.as_str(),
    ] {
        if !is_nonempty(value) {
            reasons.push(ReasonCode::MissingIdentity);
        }
    }
    for digest in [
        packet.dataset_revision_sha256.as_str(),
        packet.raw_trajectory_sha256.as_str(),
        packet.source_code_sha256.as_str(),
        packet.analysis_pipeline_sha256.as_str(),
    ] {
        if !valid_sha256(digest) {
            reasons.push(ReasonCode::InvalidDigest);
        }
    }
    if packet.sampling_hz_numerator == 0 || packet.sampling_hz_denominator == 0 {
        reasons.push(ReasonCode::ZeroSamplingRate);
    }
    if packet.frame_count == 0 {
        reasons.push(ReasonCode::ZeroFrameCount);
    }
    if packet.expected_transfer_probe_count == 0 && packet.expected_transfer_grid_sha256.is_some() {
        reasons.push(ReasonCode::InvalidTransferGridCommitment);
    }
    if packet.expected_transfer_probe_count > 0
        && packet
            .expected_transfer_grid_sha256
            .as_deref()
            .is_none_or(|digest| !valid_sha256(digest))
    {
        reasons.push(ReasonCode::InvalidTransferGridCommitment);
    }

    let mut ids = BTreeSet::new();
    for measurement in &packet.measurements {
        let row_source = &measurement.source_binding;
        if row_source.dataset_id != packet.dataset_id
            || row_source.dataset_revision_sha256 != packet.dataset_revision_sha256
            || row_source.raw_trajectory_sha256 != packet.raw_trajectory_sha256
            || row_source.animal_id != packet.animal_id
            || row_source.session_id != packet.session_id
            || row_source.trial_id != packet.trial_id
            || row_source.condition != packet.condition
            || row_source.stimulus != packet.stimulus
        {
            reasons.push(ReasonCode::MixedSourceIdentity);
        }
        for value in [
            row_source.dataset_id.as_str(),
            row_source.animal_id.as_str(),
            row_source.session_id.as_str(),
            row_source.trial_id.as_str(),
        ] {
            if !is_nonempty(value) {
                reasons.push(ReasonCode::MissingIdentity);
            }
        }
        for digest in [
            row_source.dataset_revision_sha256.as_str(),
            row_source.raw_trajectory_sha256.as_str(),
        ] {
            if !valid_sha256(digest) {
                reasons.push(ReasonCode::InvalidDigest);
            }
        }
        if !ids.insert(measurement.measurement_id.as_str()) {
            reasons.push(ReasonCode::DuplicateMeasurementId);
        }
        if let Err(error) = validate_measurement(packet, measurement) {
            reasons.extend(error.reasons);
        }
    }

    // Reject duplicate semantic keys. A duplicate row may not be used to
    // manufacture the coverage denominator.
    let mut semantic_keys = BTreeSet::new();
    for measurement in &packet.measurements {
        let key = format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}",
            measurement.endpoint,
            measurement.metric_id,
            measurement.leg,
            measurement.pair,
            measurement.tripod_group,
            measurement.transfer_probe,
            measurement.channel_id.as_deref().unwrap_or(""),
            measurement.method_sha256
        );
        if !semantic_keys.insert(key) {
            reasons.push(ReasonCode::DuplicateEndpointKey);
        }
    }

    if reasons.is_empty() {
        Ok(())
    } else {
        reasons.sort();
        reasons.dedup();
        Err(BridgeError { reasons })
    }
}

fn measured_legs(packet: &GaitOutcomePacket, endpoint: EndpointKind) -> usize {
    packet
        .measurements
        .iter()
        .filter(|m| m.endpoint == endpoint && measured(m))
        .filter_map(|m| m.leg)
        .collect::<BTreeSet<_>>()
        .len()
}

fn grouped_metric_coverage(
    packet: &GaitOutcomePacket,
    endpoint: EndpointKind,
    required: usize,
) -> Vec<MetricCoverage> {
    let mut groups: BTreeMap<
        (MetricId, String, Option<String>, Option<String>),
        BTreeSet<String>,
    > = BTreeMap::new();
    for measurement in &packet.measurements {
        if measurement.endpoint != endpoint || !measured(measurement) {
            continue;
        }
        let method = measurement.method_sha256.clone();
        let preprocessing = measurement.preprocessing_sha256.clone();
        let phase_estimator = measurement.phase_estimator_sha256.clone();
        let identity = match endpoint {
            EndpointKind::PairwiseInterlegCoupling => measurement
                .pair
                .map(|pair| format!("{:?}:{:?}", pair.first, pair.second)),
            EndpointKind::TripodCoordination => {
                measurement.tripod_group.map(|group| format!("{group:?}"))
            }
            _ => None,
        };
        if let Some(identity) = identity {
            groups
                .entry((
                    measurement.metric_id,
                    method,
                    preprocessing,
                    phase_estimator,
                ))
                .or_default()
                .insert(identity);
        }
    }

    groups
        .into_iter()
        .map(
            |((metric_id, method_sha256, preprocessing_sha256, phase_estimator_sha256), identities)| {
                let measured = identities.len();
                MetricCoverage {
                    metric_id,
                    method_sha256,
                    preprocessing_sha256,
                    phase_estimator_sha256,
                    measured,
                    required,
                    state: coverage_state(measured, required),
                }
            },
        )
        .collect()
}

fn phase_reset_probe_coverage(packet: &GaitOutcomePacket) -> usize {
    let mut fractional = BTreeSet::new();
    let mut slips = BTreeSet::new();
    for measurement in &packet.measurements {
        if !measured(measurement) {
            continue;
        }
        if let Some(key) = measurement.transfer_probe {
            match measurement.metric_id {
                MetricId::PhaseResetFractionalMicroturnsV1 => {
                    fractional.insert((
                        key,
                        measurement.method_sha256.clone(),
                        measurement.preprocessing_sha256.clone(),
                        measurement.phase_estimator_sha256.clone(),
                    ));
                }
                MetricId::PhaseResetCycleSlipsV1 => {
                    slips.insert((
                        key,
                        measurement.method_sha256.clone(),
                        measurement.preprocessing_sha256.clone(),
                        measurement.phase_estimator_sha256.clone(),
                    ));
                }
                _ => {}
            }
        }
    }
    fractional.intersection(&slips).count()
}

/// Validate one immutable animal/session/trial packet and classify its
/// endpoint coverage. This function does not aggregate biological replicates.
pub fn audit_packet(packet: &GaitOutcomePacket) -> Result<BridgeReport, BridgeError> {
    validate_packet(packet)?;

    let local_rhythm_legs = measured_legs(packet, EndpointKind::LocalJointRhythmicity);
    let stepping_legs = measured_legs(packet, EndpointKind::PerLegStepping);
    let pairwise_coverage =
        grouped_metric_coverage(packet, EndpointKind::PairwiseInterlegCoupling, PAIR_COUNT);
    let tripod_coverage =
        grouped_metric_coverage(packet, EndpointKind::TripodCoordination, TRIPOD_GROUP_COUNT);

    let transfer_measured = phase_reset_probe_coverage(packet);
    let transfer_expected = packet.expected_transfer_probe_count;
    let transfer_state = if transfer_expected == 0 {
        CoverageState::NotReported
    } else {
        coverage_state(transfer_measured, transfer_expected as usize)
    };

    let velocity_rows: Vec<&Measurement> = packet
        .measurements
        .iter()
        .filter(|m| m.metric_id == MetricId::FicTracForwardVelocity)
        .collect();
    let fictrac_velocity_state = velocity_rows
        .first()
        .map(|measurement| match measurement.availability {
            Availability::Measured => CoverageState::Complete,
            Availability::NotAvailable => CoverageState::NotAvailable,
            _ => CoverageState::NoneMeasured,
        })
        .unwrap_or(CoverageState::NotReported);

    let rhythm_count = local_rhythm_legs.max(stepping_legs);
    let interleg_count = pairwise_coverage
        .iter()
        .map(|coverage| coverage.measured)
        .sum::<usize>()
        + tripod_coverage
            .iter()
            .map(|coverage| coverage.measured)
            .sum::<usize>();
    let any_pairwise_complete = pairwise_coverage
        .iter()
        .any(|coverage| coverage.state == CoverageState::Complete);
    let any_tripod_complete = tripod_coverage
        .iter()
        .any(|coverage| coverage.state == CoverageState::Complete);

    let summary = if any_pairwise_complete || any_tripod_complete {
        SummaryState::InterlegCoverageComplete
    } else if interleg_count > 0 {
        SummaryState::InterlegCoveragePartial
    } else if rhythm_count > 0 {
        SummaryState::RhythmOnly
    } else {
        SummaryState::NoRhythmOrInterlegData
    };

    Ok(BridgeReport {
        packet_id: packet.packet_id.clone(),
        dataset_id: packet.dataset_id.clone(),
        dataset_revision_sha256: packet.dataset_revision_sha256.clone(),
        raw_trajectory_sha256: packet.raw_trajectory_sha256.clone(),
        source_code_sha256: packet.source_code_sha256.clone(),
        analysis_pipeline_sha256: packet.analysis_pipeline_sha256.clone(),
        source_access: packet.source_access,
        animal_id: packet.animal_id.clone(),
        session_id: packet.session_id.clone(),
        trial_id: packet.trial_id.clone(),
        condition: packet.condition,
        stimulus: packet.stimulus,
        local_rhythm_measured_legs: local_rhythm_legs,
        local_rhythm_state: coverage_state(local_rhythm_legs, LEG_COUNT),
        per_leg_stepping_measured_legs: stepping_legs,
        per_leg_stepping_state: coverage_state(stepping_legs, LEG_COUNT),
        pairwise_coverage,
        tripod_coverage,
        phase_reset_measured_probe_cells: transfer_measured,
        phase_reset_expected_probe_cells: transfer_expected as usize,
        phase_reset_state: transfer_state,
        fictrac_velocity_state,
        summary,
        emits_global_gait_certification: false,
        upstream_metric_validity: "METRIC_VALUES_AND_EXTRACTION_VALIDITY_ASSUMED_FROM_BOUND_SOURCE_RECEIPTS",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn base_packet() -> GaitOutcomePacket {
        GaitOutcomePacket {
            packet_id: "synthetic-dng100-trial-0001".into(),
            dataset_id: "synthetic-pugliese-anipose".into(),
            dataset_revision_sha256: digest('a'),
            raw_trajectory_sha256: digest('b'),
            source_code_sha256: digest('c'),
            analysis_pipeline_sha256: digest('d'),
            source_access: SourceAccess::Synthetic,
            animal_id: "synthetic-animal-1".into(),
            session_id: "synthetic-session-1".into(),
            trial_id: "synthetic-trial-1".into(),
            condition: Condition::BallContact,
            stimulus: Stimulus::DNg100,
            sampling_hz_numerator: 300,
            sampling_hz_denominator: 1,
            frame_count: 4500,
            fictrac_columns_available: true,
            expected_transfer_probe_count: 8,
            expected_transfer_grid_sha256: Some(digest('e')),
            measurements: Vec::new(),
        }
    }

    fn measurement(
        id: &str,
        endpoint: EndpointKind,
        metric_id: MetricId,
        unit: Unit,
        value: f64,
    ) -> Measurement {
        Measurement {
            measurement_id: id.into(),
            source_binding: MeasurementSourceBinding {
                dataset_id: "synthetic-pugliese-anipose".into(),
                dataset_revision_sha256: digest('a'),
                raw_trajectory_sha256: digest('b'),
                animal_id: "synthetic-animal-1".into(),
                session_id: "synthetic-session-1".into(),
                trial_id: "synthetic-trial-1".into(),
                condition: Condition::BallContact,
                stimulus: Stimulus::DNg100,
            },
            endpoint,
            metric_id,
            availability: Availability::Measured,
            unit,
            value: Some(value),
            aggregation: AggregationLevel::Trial,
            method_sha256: digest('f'),
            preprocessing_sha256: Some(digest('1')),
            phase_estimator_sha256: Some(digest('2')),
            channel_id: Some("C_flex".into()),
            leg: None,
            pair: None,
            tripod_group: None,
            transfer_probe: None,
            numerator_ticks: None,
            denominator_ticks: None,
            sample_count: Some(600),
            support_count: Some(20),
            window_start_frame: Some(1650),
            window_end_frame: Some(2250),
        }
    }

    fn full_rhythm_fixture(stimulus: Stimulus, fictrac: bool) -> GaitOutcomePacket {
        let mut packet = base_packet();
        packet.stimulus = stimulus;
        packet.condition = if stimulus == Stimulus::DNb08 {
            Condition::AirStepping
        } else {
            Condition::BallContact
        };
        packet.fictrac_columns_available = fictrac;

        for leg in LegId::ALL {
            let mut score = measurement(
                &format!("rhythm-{leg:?}"),
                EndpointKind::LocalJointRhythmicity,
                MetricId::PuglieseAutocorrelationOscillationScore,
                Unit::Score0To1,
                0.98,
            );
            score.leg = Some(leg);
            packet.measurements.push(score);

            let mut frequency = measurement(
                &format!("step-{leg:?}"),
                EndpointKind::PerLegStepping,
                MetricId::PuglieseSingleLegStepFrequency,
                Unit::Hertz,
                6.0,
            );
            frequency.leg = Some(leg);
            frequency.channel_id = Some("E_y".into());
            packet.measurements.push(frequency);
        }

        let mut fractional = measurement(
            "reset-phase-probe-1",
            EndpointKind::PhaseResetTransfer,
            MetricId::PhaseResetFractionalMicroturnsV1,
            Unit::Microturns,
            12_500.0,
        );
        fractional.transfer_probe = Some(TransferProbeKey {
            sender: Some(LegId::LeftFore),
            probe_class: ProbeClass::LocalPhaseKick,
            recipient: LegId::RightFore,
            phase_bin: 2,
            magnitude_bin: 1,
            delay_index: 3,
        });
        packet.measurements.push(fractional);

        let mut slips = measurement(
            "reset-cycle-probe-1",
            EndpointKind::PhaseResetTransfer,
            MetricId::PhaseResetCycleSlipsV1,
            Unit::CycleSlips,
            0.0,
        );
        slips.transfer_probe = packet.measurements.last().unwrap().transfer_probe;
        packet.measurements.push(slips);

        let mut velocity = measurement(
            "forward-velocity",
            EndpointKind::EmbodiedBehaviour,
            MetricId::FicTracForwardVelocity,
            Unit::MillimetresPerSecond,
            1.5,
        );
        velocity.channel_id = None;
        velocity.preprocessing_sha256 = Some(digest('7'));
        velocity.phase_estimator_sha256 = None;
        velocity.sample_count = Some(4500);
        velocity.support_count = Some(4500);
        velocity.window_start_frame = Some(0);
        velocity.window_end_frame = Some(4500);
        if !fictrac {
            velocity.availability = Availability::NotAvailable;
            velocity.value = None;
        }
        packet.measurements.push(velocity);
        for row in &mut packet.measurements {
            row.source_binding.stimulus = packet.stimulus;
            row.source_binding.condition = packet.condition;
        }
        packet
    }

    #[test]
    fn six_strong_local_rhythm_scores_do_not_certify_interleg_coordination() {
        let report = audit_packet(&full_rhythm_fixture(Stimulus::DNg100, true)).unwrap();
        assert_eq!(report.local_rhythm_measured_legs, 6);
        assert_eq!(report.local_rhythm_state, CoverageState::Complete);
        assert_eq!(report.per_leg_stepping_measured_legs, 6);
        assert!(report.pairwise_coverage.is_empty());
        assert!(report.tripod_coverage.is_empty());
        assert_eq!(report.phase_reset_measured_probe_cells, 1);
        assert_eq!(report.phase_reset_expected_probe_cells, 8);
        assert_eq!(report.phase_reset_state, CoverageState::Partial);
        assert_eq!(report.fictrac_velocity_state, CoverageState::Complete);
        assert_eq!(report.summary, SummaryState::RhythmOnly);
        assert!(!report.emits_global_gait_certification);
    }

    #[test]
    fn dnb08_missing_fictrac_velocity_is_not_zero() {
        let mut packet = full_rhythm_fixture(Stimulus::DNb08, false);
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.fictrac_velocity_state, CoverageState::NotAvailable);
        let velocity = packet
            .measurements
            .iter_mut()
            .find(|m| m.metric_id == MetricId::FicTracForwardVelocity)
            .unwrap();
        velocity.value = Some(0.0);
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::NonMeasuredValuePresent)
        );
    }

    #[test]
    fn rejects_frame_level_rows_as_independent_outcomes() {
        let mut packet = full_rhythm_fixture(Stimulus::DNg100, true);
        packet.measurements[0].aggregation = AggregationLevel::Frame;
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::MixedAggregationLevelFrame)
        );
    }

    #[test]
    fn rejects_empty_measurement_identity() {
        let mut packet = full_rhythm_fixture(Stimulus::DNg100, true);
        packet.measurements[0].measurement_id.clear();
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::MissingIdentity)
        );
    }

    #[test]
    fn rejects_measurement_from_another_animal_session_or_trial() {
        let mut packet = full_rhythm_fixture(Stimulus::DNg100, true);
        packet.measurements[0].source_binding.animal_id = "other-animal".into();
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::MixedSourceIdentity)
        );
    }

    #[test]
    fn negative_phase_resets_and_negative_cycle_slips_are_valid_outcomes() {
        let mut packet = full_rhythm_fixture(Stimulus::DNg100, true);
        packet
            .measurements
            .iter_mut()
            .find(|m| m.metric_id == MetricId::PhaseResetFractionalMicroturnsV1)
            .unwrap()
            .value = Some(-12_500.0);
        packet
            .measurements
            .iter_mut()
            .find(|m| m.metric_id == MetricId::PhaseResetCycleSlipsV1)
            .unwrap()
            .value = Some(-1.0);
        assert!(audit_packet(&packet).is_ok());
    }

    #[test]
    fn rejects_duplicate_measurement_rows() {
        let mut packet = full_rhythm_fixture(Stimulus::DNg100, true);
        let duplicate = packet.measurements[0].clone();
        packet.measurements.push(duplicate);
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::DuplicateMeasurementId)
        );
    }

    #[test]
    fn rejects_unverified_sapkal_plv_pipeline() {
        let mut packet = base_packet();
        let mut row = measurement(
            "pair-l1-r1",
            EndpointKind::PairwiseInterlegCoupling,
            MetricId::SapkalButterworthHilbertPlvV2,
            Unit::PhaseVectorStrength,
            0.8,
        );
        row.pair = Some(LegPair::new(LegId::LeftFore, LegId::RightFore).unwrap());
        row.phase_estimator_sha256 = None;
        packet.measurements.push(row);
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::MissingPhaseEstimatorDigest)
        );
    }

    #[test]
    fn generic_phase_vector_strength_never_implies_sapkal_pipeline_identity() {
        let mut packet = base_packet();
        let mut row = measurement(
            "pair-l1-r1",
            EndpointKind::PairwiseInterlegCoupling,
            MetricId::GenericPhaseVectorStrength,
            Unit::PhaseVectorStrength,
            0.8,
        );
        row.pair = Some(LegPair::new(LegId::LeftFore, LegId::RightFore).unwrap());
        packet.measurements.push(row);
        let report = audit_packet(&packet).unwrap();
        assert_eq!(
            report.pairwise_coverage[0].metric_id,
            MetricId::GenericPhaseVectorStrength
        );
        assert_eq!(report.summary, SummaryState::InterlegCoveragePartial);
    }

    #[test]
    fn tcs_metric_definitions_keep_separate_coverage_and_values() {
        let mut packet = base_packet();
        for (metric, method, group, value, numerator, denominator) in [
            (
                MetricId::TcsNatvar2024,
                digest('4'),
                TripodGroup::LeftForeRightMiddleLeftHind,
                0.5,
                5,
                10,
            ),
            (
                MetricId::TcsSapkal2026V2,
                digest('5'),
                TripodGroup::LeftForeRightMiddleLeftHind,
                1.5,
                6,
                4,
            ),
        ] {
            let mut row = measurement(
                &format!("tcs-{metric:?}"),
                EndpointKind::TripodCoordination,
                metric,
                Unit::Ratio,
                value,
            );
            row.method_sha256 = method;
            row.tripod_group = Some(group);
            row.numerator_ticks = Some(numerator);
            row.denominator_ticks = Some(denominator);
            row.preprocessing_sha256 = Some(digest('1'));
            row.phase_estimator_sha256 = Some(digest('2'));
            packet.measurements.push(row);
        }
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.tripod_coverage.len(), 2);
        assert!(
            report
                .tripod_coverage
                .iter()
                .all(|coverage| coverage.measured == 1 && coverage.state == CoverageState::Partial)
        );
        assert_eq!(report.summary, SummaryState::InterlegCoveragePartial);
    }

    #[test]
    fn rejects_tcs_value_that_does_not_match_source_metric_receipt() {
        let mut packet = base_packet();
        let mut row = measurement(
            "tcs-natvar",
            EndpointKind::TripodCoordination,
            MetricId::TcsNatvar2024,
            Unit::Ratio,
            0.9,
        );
        row.tripod_group = Some(TripodGroup::LeftForeRightMiddleLeftHind);
        row.numerator_ticks = Some(5);
        row.denominator_ticks = Some(10);
        packet.measurements.push(row);
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::TcsValueMismatch)
        );
    }

    #[test]
    fn pairwise_complete_requires_all_fifteen_pairs_with_one_method() {
        let mut packet = base_packet();
        let mut pairs = Vec::new();
        for (i, first) in LegId::ALL.iter().enumerate() {
            for second in LegId::ALL.iter().skip(i + 1) {
                pairs.push((*first, *second));
            }
        }
        assert_eq!(pairs.len(), PAIR_COUNT);
        for (index, (first, second)) in pairs.into_iter().enumerate() {
            let mut row = measurement(
                &format!("pair-{index}"),
                EndpointKind::PairwiseInterlegCoupling,
                MetricId::GenericPhaseVectorStrength,
                Unit::PhaseVectorStrength,
                0.7,
            );
            row.pair = Some(LegPair::new(first, second).unwrap());
            row.preprocessing_sha256 = Some(digest('1'));
            row.phase_estimator_sha256 = Some(digest('2'));
            packet.measurements.push(row);
        }
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.pairwise_coverage.len(), 1);
        assert_eq!(report.pairwise_coverage[0].measured, 15);
        assert_eq!(report.pairwise_coverage[0].state, CoverageState::Complete);
        assert_eq!(report.summary, SummaryState::InterlegCoverageComplete);
    }

    #[test]
    fn mixed_pairwise_method_digests_do_not_make_full_coverage() {
        let mut packet = base_packet();
        let pairs = [
            (LegId::LeftFore, LegId::LeftMiddle),
            (LegId::LeftFore, LegId::LeftHind),
        ];
        for (index, pair) in pairs.into_iter().enumerate() {
            let mut row = measurement(
                &format!("pair-{index}"),
                EndpointKind::PairwiseInterlegCoupling,
                MetricId::GenericPhaseVectorStrength,
                Unit::PhaseVectorStrength,
                0.7,
            );
            row.pair = Some(LegPair::new(pair.0, pair.1).unwrap());
            row.method_sha256 = digest(if index == 0 { 'a' } else { 'b' });
            row.preprocessing_sha256 = Some(digest('1'));
            row.phase_estimator_sha256 = Some(digest('2'));
            packet.measurements.push(row);
        }
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.pairwise_coverage.len(), 2);
        assert!(
            report
                .pairwise_coverage
                .iter()
                .all(|coverage| coverage.state == CoverageState::Partial)
        );
        assert_eq!(report.summary, SummaryState::InterlegCoveragePartial);
    }

    #[test]
    fn matching_pairs_with_different_phase_estimators_do_not_form_complete_coverage() {
        let mut packet = base_packet();
        let pairs = [
            (LegId::LeftFore, LegId::LeftMiddle),
            (LegId::LeftFore, LegId::LeftHind),
        ];
        for (index, pair) in pairs.into_iter().enumerate() {
            let mut row = measurement(
                &format!("pair-estimator-{index}"),
                EndpointKind::PairwiseInterlegCoupling,
                MetricId::GenericPhaseVectorStrength,
                Unit::PhaseVectorStrength,
                0.7,
            );
            row.pair = Some(LegPair::new(pair.0, pair.1).unwrap());
            row.preprocessing_sha256 = Some(digest('1'));
            row.phase_estimator_sha256 = Some(digest(if index == 0 { '2' } else { '3' }));
            packet.measurements.push(row);
        }
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.pairwise_coverage.len(), 2);
        assert!(report.pairwise_coverage.iter().all(|row| row.measured == 1));
        assert!(report
            .pairwise_coverage
            .iter()
            .all(|row| row.state == CoverageState::Partial));
    }

    #[test]
    fn phase_and_cycle_slip_outputs_from_different_estimators_do_not_complete_probe() {
        let mut packet = base_packet();
        let key = TransferProbeKey {
            sender: Some(LegId::LeftFore),
            probe_class: ProbeClass::LocalPhaseKick,
            recipient: LegId::RightFore,
            phase_bin: 1,
            magnitude_bin: 0,
            delay_index: 1,
        };
        let mut phase = measurement(
            "phase-estimator-a",
            EndpointKind::PhaseResetTransfer,
            MetricId::PhaseResetFractionalMicroturnsV1,
            Unit::Microturns,
            10.0,
        );
        phase.transfer_probe = Some(key);
        phase.preprocessing_sha256 = Some(digest('1'));
        phase.phase_estimator_sha256 = Some(digest('2'));
        let mut slips = measurement(
            "cycle-estimator-b",
            EndpointKind::PhaseResetTransfer,
            MetricId::PhaseResetCycleSlipsV1,
            Unit::CycleSlips,
            0.0,
        );
        slips.transfer_probe = Some(key);
        slips.preprocessing_sha256 = Some(digest('1'));
        slips.phase_estimator_sha256 = Some(digest('3'));
        packet.measurements.extend([phase, slips]);
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.phase_reset_measured_probe_cells, 0);
        assert_eq!(report.phase_reset_state, CoverageState::NoneMeasured);
    }

    #[test]
    fn not_estimable_zero_denominator_tcs_is_preserved_without_zero_imputation() {
        let mut packet = base_packet();
        let mut row = measurement(
            "tcs-not-estimable",
            EndpointKind::TripodCoordination,
            MetricId::TcsSapkal2026V2,
            Unit::Ratio,
            0.0,
        );
        row.availability = Availability::NotEstimable;
        row.value = None;
        row.tripod_group = Some(TripodGroup::LeftForeRightMiddleLeftHind);
        row.numerator_ticks = Some(0);
        row.denominator_ticks = Some(0);
        row.preprocessing_sha256 = Some(digest('1'));
        row.phase_estimator_sha256 = None;
        packet.measurements.push(row);
        let report = audit_packet(&packet).unwrap();
        assert!(report.tripod_coverage.is_empty());
        assert_eq!(report.summary, SummaryState::NoRhythmOrInterlegData);
    }

    #[test]
    fn two_tripod_groups_are_required_for_complete_metric_coverage() {
        let mut packet = base_packet();
        for group in TripodGroup::ALL {
            let mut row = measurement(
                &format!("tcs-{group:?}"),
                EndpointKind::TripodCoordination,
                MetricId::TcsNatvar2024,
                Unit::Ratio,
                0.5,
            );
            row.tripod_group = Some(group);
            row.numerator_ticks = Some(5);
            row.denominator_ticks = Some(10);
            row.preprocessing_sha256 = Some(digest('1'));
            row.phase_estimator_sha256 = Some(digest('2'));
            packet.measurements.push(row);
        }
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.tripod_coverage.len(), 1);
        assert_eq!(report.tripod_coverage[0].measured, 2);
        assert_eq!(report.tripod_coverage[0].state, CoverageState::Complete);
        assert_eq!(report.summary, SummaryState::InterlegCoverageComplete);
    }

    #[test]
    fn tcs_2024_and_sapkal_2026_are_not_collapsed_to_one_median() {
        let mut packet = base_packet();
        for metric in [MetricId::TcsNatvar2024, MetricId::TcsSapkal2026V2] {
            let mut row = measurement(
                &format!("tcs-{metric:?}"),
                EndpointKind::TripodCoordination,
                metric,
                Unit::Ratio,
                0.5,
            );
            row.tripod_group = Some(TripodGroup::LeftForeRightMiddleLeftHind);
            row.numerator_ticks = Some(1);
            row.denominator_ticks = Some(2);
            row.preprocessing_sha256 = Some(digest('1'));
            row.phase_estimator_sha256 = None;
            packet.measurements.push(row);
        }
        assert!(audit_packet(&packet).is_ok());
        assert_eq!(packet.measurements[0].metric_id, MetricId::TcsNatvar2024);
        assert_eq!(packet.measurements[1].metric_id, MetricId::TcsSapkal2026V2);
    }

    #[test]
    fn dng100_and_dnb08_are_distinct_trial_packets() {
        let dng100 = full_rhythm_fixture(Stimulus::DNg100, true);
        let mut dnb08 = full_rhythm_fixture(Stimulus::DNb08, false);
        dnb08.packet_id = "synthetic-dnb08-trial-0001".into();
        assert_eq!(
            audit_packet(&dng100).unwrap().fictrac_velocity_state,
            CoverageState::Complete
        );
        assert_eq!(
            audit_packet(&dnb08).unwrap().fictrac_velocity_state,
            CoverageState::NotAvailable
        );
        // A single packet has one immutable stimulus identity. The adapter
        // intentionally offers no cross-stimulus pooling function.
        assert_ne!(dng100.stimulus, dnb08.stimulus);
    }

    #[test]
    fn rejects_same_trial_packet_when_tcs_group_is_missing() {
        let mut packet = base_packet();
        let mut row = measurement(
            "tcs-one-tripod",
            EndpointKind::TripodCoordination,
            MetricId::TcsNatvar2024,
            Unit::Ratio,
            0.5,
        );
        row.tripod_group = Some(TripodGroup::LeftForeRightMiddleLeftHind);
        row.numerator_ticks = Some(5);
        row.denominator_ticks = Some(10);
        row.preprocessing_sha256 = Some(digest('1'));
        row.phase_estimator_sha256 = None;
        packet.measurements.push(row);
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.tripod_coverage[0].state, CoverageState::Partial);
        assert_eq!(report.summary, SummaryState::InterlegCoveragePartial);
    }

    #[test]
    fn a_frame_count_is_metadata_not_an_iid_sample_size() {
        let packet = full_rhythm_fixture(Stimulus::DNg100, true);
        let report = audit_packet(&packet).unwrap();
        // The raw trial has 4,500 frames, but the independent-unit identity
        // remains exactly one animal/session/trial packet.
        assert_eq!(packet.frame_count, 4500);
        assert_eq!(report.packet_id, packet.packet_id);
    }

    #[test]
    fn rejects_malformed_dataset_or_method_digest() {
        let mut packet = full_rhythm_fixture(Stimulus::DNg100, true);
        packet.raw_trajectory_sha256 = "not-a-digest".into();
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::InvalidDigest)
        );
    }

    #[test]
    fn rejects_out_of_window_measurement() {
        let mut packet = full_rhythm_fixture(Stimulus::DNg100, true);
        packet.measurements[0].window_end_frame = Some(4501);
        assert!(
            audit_packet(&packet)
                .unwrap_err()
                .contains(ReasonCode::WindowOutsideTrial)
        );
    }

    #[test]
    fn transfer_probe_coverage_requires_both_phase_and_winding_outputs() {
        let mut packet = base_packet();
        let mut phase = measurement(
            "phase-only",
            EndpointKind::PhaseResetTransfer,
            MetricId::PhaseResetFractionalMicroturnsV1,
            Unit::Microturns,
            10.0,
        );
        phase.transfer_probe = Some(TransferProbeKey {
            sender: Some(LegId::LeftFore),
            probe_class: ProbeClass::LocalPhaseKick,
            recipient: LegId::RightFore,
            phase_bin: 1,
            magnitude_bin: 0,
            delay_index: 1,
        });
        packet.measurements.push(phase);
        let report = audit_packet(&packet).unwrap();
        assert_eq!(report.phase_reset_measured_probe_cells, 0);
    }
}

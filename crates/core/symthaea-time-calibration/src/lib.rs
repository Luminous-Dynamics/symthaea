// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-first clock calibration from four timestamp exchanges.
//!
//! This crate does not synchronize clocks and does not authenticate timing
//! claims. It derives the set of source->target clock offsets consistent with
//! one four-timestamp exchange under explicit assumptions:
//!
//! - source and target clocks remain continuous within their declared epochs;
//! - timestamp uncertainty is finitely bounded;
//! - one-way transport delays are non-negative;
//! - source->target offset is approximately constant during the exchange.
//!
//! No symmetric-delay assumption is made. Multiple exchanges can be combined
//! only by interval intersection; contradictory evidence fails closed.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};
use symthaea_time_integrity::{
    ClockDomainId, ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
};

/// One timestamp plus the temporal evidence attached to that timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimestampEvidence {
    pub timestamp_us: u64,
    pub receipt: TimeIntegrityReceipt,
}

impl TimestampEvidence {
    pub fn new(timestamp_us: u64, receipt: TimeIntegrityReceipt) -> Self {
        Self {
            timestamp_us,
            receipt,
        }
    }
}

/// Position of one timestamp in the four-timestamp exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeStamp {
    SourceSend,
    TargetReceive,
    TargetSend,
    SourceReceive,
}

/// Fail-closed validation or calibration error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationError {
    MissingEpoch { stamp: ExchangeStamp },
    ContinuityNotEstablished {
        stamp: ExchangeStamp,
        status: ContinuityStatus,
    },
    UnboundedUncertainty { stamp: ExchangeStamp },
    SourceClockDomainMismatch {
        send: ClockDomainId,
        receive: ClockDomainId,
    },
    SourceClockEpochMismatch {
        send: ClockEpochId,
        receive: ClockEpochId,
    },
    TargetClockDomainMismatch {
        receive: ClockDomainId,
        send: ClockDomainId,
    },
    TargetClockEpochMismatch {
        receive: ClockEpochId,
        send: ClockEpochId,
    },
    SourceLocalOrderImpossible,
    TargetLocalOrderImpossible,
    InconsistentOffsetInterval { lower_us: i128, upper_us: i128 },
    DeclaredSameTimebaseContradiction { lower_us: i128, upper_us: i128 },
    EmptyCalibrationSet,
    CalibrationIdentityMismatch,
    DisjointCalibrationIntervals,
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEpoch { stamp } => {
                write!(f, "{stamp:?} timestamp has no continuity epoch")
            }
            Self::ContinuityNotEstablished { stamp, status } => write!(
                f,
                "{stamp:?} timestamp continuity is not established: {status:?}"
            ),
            Self::UnboundedUncertainty { stamp } => {
                write!(f, "{stamp:?} timestamp has no finite uncertainty bound")
            }
            Self::SourceClockDomainMismatch { send, receive } => write!(
                f,
                "source send/receive clock domains differ: {send} != {receive}"
            ),
            Self::SourceClockEpochMismatch { send, receive } => write!(
                f,
                "source send/receive clock epochs differ: {send} != {receive}"
            ),
            Self::TargetClockDomainMismatch { receive, send } => write!(
                f,
                "target receive/send clock domains differ: {receive} != {send}"
            ),
            Self::TargetClockEpochMismatch { receive, send } => write!(
                f,
                "target receive/send clock epochs differ: {receive} != {send}"
            ),
            Self::SourceLocalOrderImpossible => write!(
                f,
                "source receive occurs definitely before source send after uncertainty"
            ),
            Self::TargetLocalOrderImpossible => write!(
                f,
                "target send occurs definitely before target receive after uncertainty"
            ),
            Self::InconsistentOffsetInterval { lower_us, upper_us } => write!(
                f,
                "four-timestamp evidence admits no offset: lower {lower_us} > upper {upper_us}"
            ),
            Self::DeclaredSameTimebaseContradiction { lower_us, upper_us } => write!(
                f,
                "receipts declare one clock domain/epoch but calibration interval [{lower_us}, {upper_us}] excludes zero offset"
            ),
            Self::EmptyCalibrationSet => write!(f, "calibration set must not be empty"),
            Self::CalibrationIdentityMismatch => write!(
                f,
                "calibration evidence does not share one source/target domain and epoch pair"
            ),
            Self::DisjointCalibrationIntervals => write!(
                f,
                "calibration intervals are disjoint; evidence is mutually inconsistent"
            ),
        }
    }
}

impl std::error::Error for CalibrationError {}

fn bounded_error(
    stamp: ExchangeStamp,
    evidence: &TimestampEvidence,
) -> Result<u64, CalibrationError> {
    if evidence.receipt.clock_epoch.is_none() {
        return Err(CalibrationError::MissingEpoch { stamp });
    }
    if evidence.receipt.continuity != ContinuityStatus::Continuous {
        return Err(CalibrationError::ContinuityNotEstablished {
            stamp,
            status: evidence.receipt.continuity,
        });
    }
    match evidence.receipt.uncertainty {
        TimeUncertainty::Bounded { max_error_us } => Ok(max_error_us),
        TimeUncertainty::Unbounded => Err(CalibrationError::UnboundedUncertainty { stamp }),
    }
}

/// A validated four-timestamp exchange.
///
/// Naming follows the common request/response convention:
///
/// - `t1`: source sends request
/// - `t2`: target receives request
/// - `t3`: target sends response
/// - `t4`: source receives response
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FourTimestampExchange {
    source_send: TimestampEvidence,
    target_receive: TimestampEvidence,
    target_send: TimestampEvidence,
    source_receive: TimestampEvidence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FourTimestampExchangeWire {
    source_send: TimestampEvidence,
    target_receive: TimestampEvidence,
    target_send: TimestampEvidence,
    source_receive: TimestampEvidence,
}

impl<'de> Deserialize<'de> for FourTimestampExchange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FourTimestampExchangeWire::deserialize(deserializer)?;
        Self::new(
            wire.source_send,
            wire.target_receive,
            wire.target_send,
            wire.source_receive,
        )
        .map_err(de::Error::custom)
    }
}

impl FourTimestampExchange {
    pub fn new(
        source_send: TimestampEvidence,
        target_receive: TimestampEvidence,
        target_send: TimestampEvidence,
        source_receive: TimestampEvidence,
    ) -> Result<Self, CalibrationError> {
        let e1 = bounded_error(ExchangeStamp::SourceSend, &source_send)?;
        let e2 = bounded_error(ExchangeStamp::TargetReceive, &target_receive)?;
        let e3 = bounded_error(ExchangeStamp::TargetSend, &target_send)?;
        let e4 = bounded_error(ExchangeStamp::SourceReceive, &source_receive)?;

        if source_send.receipt.clock_domain != source_receive.receipt.clock_domain {
            return Err(CalibrationError::SourceClockDomainMismatch {
                send: source_send.receipt.clock_domain.clone(),
                receive: source_receive.receipt.clock_domain.clone(),
            });
        }
        let source_send_epoch = source_send.receipt.clock_epoch.clone().unwrap();
        let source_receive_epoch = source_receive.receipt.clock_epoch.clone().unwrap();
        if source_send_epoch != source_receive_epoch {
            return Err(CalibrationError::SourceClockEpochMismatch {
                send: source_send_epoch,
                receive: source_receive_epoch,
            });
        }

        if target_receive.receipt.clock_domain != target_send.receipt.clock_domain {
            return Err(CalibrationError::TargetClockDomainMismatch {
                receive: target_receive.receipt.clock_domain.clone(),
                send: target_send.receipt.clock_domain.clone(),
            });
        }
        let target_receive_epoch = target_receive.receipt.clock_epoch.clone().unwrap();
        let target_send_epoch = target_send.receipt.clock_epoch.clone().unwrap();
        if target_receive_epoch != target_send_epoch {
            return Err(CalibrationError::TargetClockEpochMismatch {
                receive: target_receive_epoch,
                send: target_send_epoch,
            });
        }

        // Physical order is source-send <= source-receive and target-receive <=
        // target-send. We reject only when the bounded timestamp intervals make
        // that order impossible; overlapping uncertainty remains admissible.
        let source_send_earliest = i128::from(source_send.timestamp_us) - i128::from(e1);
        let source_receive_latest = i128::from(source_receive.timestamp_us) + i128::from(e4);
        if source_receive_latest < source_send_earliest {
            return Err(CalibrationError::SourceLocalOrderImpossible);
        }

        let target_receive_earliest =
            i128::from(target_receive.timestamp_us) - i128::from(e2);
        let target_send_latest = i128::from(target_send.timestamp_us) + i128::from(e3);
        if target_send_latest < target_receive_earliest {
            return Err(CalibrationError::TargetLocalOrderImpossible);
        }

        let exchange = Self {
            source_send,
            target_receive,
            target_send,
            source_receive,
        };

        // Validate that at least one source->target offset satisfies the
        // exchange assumptions before allowing the evidence object to exist.
        let interval = exchange.offset_interval()?;
        if exchange.source_domain() == exchange.target_domain()
            && exchange.source_epoch() == exchange.target_epoch()
            && !interval.contains(0)
        {
            return Err(CalibrationError::DeclaredSameTimebaseContradiction {
                lower_us: interval.lower_us,
                upper_us: interval.upper_us,
            });
        }

        Ok(exchange)
    }

    pub fn source_send(&self) -> &TimestampEvidence {
        &self.source_send
    }

    pub fn target_receive(&self) -> &TimestampEvidence {
        &self.target_receive
    }

    pub fn target_send(&self) -> &TimestampEvidence {
        &self.target_send
    }

    pub fn source_receive(&self) -> &TimestampEvidence {
        &self.source_receive
    }

    pub fn source_domain(&self) -> &ClockDomainId {
        &self.source_send.receipt.clock_domain
    }

    pub fn target_domain(&self) -> &ClockDomainId {
        &self.target_receive.receipt.clock_domain
    }

    pub fn source_epoch(&self) -> &ClockEpochId {
        self.source_send.receipt.clock_epoch.as_ref().unwrap()
    }

    pub fn target_epoch(&self) -> &ClockEpochId {
        self.target_receive.receipt.clock_epoch.as_ref().unwrap()
    }

    /// Derive the admissible source->target offset interval.
    ///
    /// Let `theta = target_time - source_time`. Non-negative one-way delay gives:
    ///
    /// `theta <= t2 - t1`
    ///
    /// `theta >= t3 - t4`
    ///
    /// Endpoint timestamp uncertainty widens those bounds conservatively:
    ///
    /// `lower = t3 - t4 - e3 - e4`
    ///
    /// `upper = t2 - t1 + e2 + e1`.
    pub fn offset_interval(&self) -> Result<ClockOffsetIntervalUs, CalibrationError> {
        let e1 = self.source_send.receipt.uncertainty.max_error_us().unwrap();
        let e2 = self.target_receive.receipt.uncertainty.max_error_us().unwrap();
        let e3 = self.target_send.receipt.uncertainty.max_error_us().unwrap();
        let e4 = self.source_receive.receipt.uncertainty.max_error_us().unwrap();

        let lower_us = i128::from(self.target_send.timestamp_us)
            - i128::from(self.source_receive.timestamp_us)
            - i128::from(e3)
            - i128::from(e4);
        let upper_us = i128::from(self.target_receive.timestamp_us)
            - i128::from(self.source_send.timestamp_us)
            + i128::from(e2)
            + i128::from(e1);

        ClockOffsetIntervalUs::new(lower_us, upper_us)
    }
}

/// Closed interval containing every source->target offset admitted by the
/// calibration evidence and its stated assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ClockOffsetIntervalUs {
    lower_us: i128,
    upper_us: i128,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClockOffsetIntervalWire {
    lower_us: i128,
    upper_us: i128,
}

impl<'de> Deserialize<'de> for ClockOffsetIntervalUs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ClockOffsetIntervalWire::deserialize(deserializer)?;
        Self::new(wire.lower_us, wire.upper_us).map_err(de::Error::custom)
    }
}

impl ClockOffsetIntervalUs {
    pub fn new(lower_us: i128, upper_us: i128) -> Result<Self, CalibrationError> {
        if lower_us > upper_us {
            return Err(CalibrationError::InconsistentOffsetInterval {
                lower_us,
                upper_us,
            });
        }
        Ok(Self { lower_us, upper_us })
    }

    pub fn lower_us(self) -> i128 {
        self.lower_us
    }

    pub fn upper_us(self) -> i128 {
        self.upper_us
    }

    pub fn contains(self, offset_us: i128) -> bool {
        self.lower_us <= offset_us && offset_us <= self.upper_us
    }

    pub fn width_us(self) -> u128 {
        (self.upper_us - self.lower_us) as u128
    }

    /// Deterministic midpoint, rounding toward the lower bound for odd widths.
    pub fn midpoint_us(self) -> i128 {
        self.lower_us + (self.upper_us - self.lower_us) / 2
    }

    /// Smallest symmetric integer error radius around [`Self::midpoint_us`]
    /// that covers the full interval.
    pub fn symmetric_radius_us(self) -> u128 {
        let width = self.width_us();
        width / 2 + width % 2
    }

    pub fn intersect(self, other: Self) -> Result<Self, CalibrationError> {
        let lower_us = self.lower_us.max(other.lower_us);
        let upper_us = self.upper_us.min(other.upper_us);
        if lower_us > upper_us {
            return Err(CalibrationError::DisjointCalibrationIntervals);
        }
        Ok(Self { lower_us, upper_us })
    }
}

/// Validated evidence from one four-timestamp calibration exchange.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClockCalibrationEvidence {
    pub exchange: FourTimestampExchange,
    pub offset_interval: ClockOffsetIntervalUs,
}

impl ClockCalibrationEvidence {
    pub fn derive(exchange: FourTimestampExchange) -> Result<Self, CalibrationError> {
        let offset_interval = exchange.offset_interval()?;
        Ok(Self {
            exchange,
            offset_interval,
        })
    }
}

/// Non-statistical consensus from intersecting several compatible calibration
/// intervals. No averaging or distributional assumption is performed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CalibrationConsensus {
    pub source_domain: ClockDomainId,
    pub source_epoch: ClockEpochId,
    pub target_domain: ClockDomainId,
    pub target_epoch: ClockEpochId,
    pub offset_interval: ClockOffsetIntervalUs,
    pub exchange_count: usize,
}

impl CalibrationConsensus {
    pub fn from_evidence(
        evidence: &[ClockCalibrationEvidence],
    ) -> Result<Self, CalibrationError> {
        let first = evidence.first().ok_or(CalibrationError::EmptyCalibrationSet)?;
        let source_domain = first.exchange.source_domain().clone();
        let source_epoch = first.exchange.source_epoch().clone();
        let target_domain = first.exchange.target_domain().clone();
        let target_epoch = first.exchange.target_epoch().clone();
        let mut interval = first.offset_interval;

        for item in &evidence[1..] {
            if item.exchange.source_domain() != &source_domain
                || item.exchange.source_epoch() != &source_epoch
                || item.exchange.target_domain() != &target_domain
                || item.exchange.target_epoch() != &target_epoch
            {
                return Err(CalibrationError::CalibrationIdentityMismatch);
            }
            interval = interval.intersect(item.offset_interval)?;
        }

        Ok(Self {
            source_domain,
            source_epoch,
            target_domain,
            target_epoch,
            offset_interval: interval,
            exchange_count: evidence.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_domain() -> ClockDomainId {
        ClockDomainId::new("sensor-a/monotonic").unwrap()
    }

    fn target_domain() -> ClockDomainId {
        ClockDomainId::new("capture-host/monotonic").unwrap()
    }

    fn source_epoch() -> ClockEpochId {
        ClockEpochId::new("sensor-a-boot-7").unwrap()
    }

    fn target_epoch() -> ClockEpochId {
        ClockEpochId::new("capture-host-boot-3").unwrap()
    }

    fn receipt(domain: ClockDomainId, epoch: ClockEpochId, error_us: u64) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(domain)
            .with_epoch(epoch)
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(error_us))
    }

    fn stamp(timestamp_us: u64, domain: ClockDomainId, epoch: ClockEpochId, error_us: u64) -> TimestampEvidence {
        TimestampEvidence::new(timestamp_us, receipt(domain, epoch, error_us))
    }

    fn exchange(t1: u64, t2: u64, t3: u64, t4: u64, error_us: u64) -> FourTimestampExchange {
        FourTimestampExchange::new(
            stamp(t1, source_domain(), source_epoch(), error_us),
            stamp(t2, target_domain(), target_epoch(), error_us),
            stamp(t3, target_domain(), target_epoch(), error_us),
            stamp(t4, source_domain(), source_epoch(), error_us),
        )
        .unwrap()
    }

    #[test]
    fn asymmetric_delay_produces_interval_not_fake_point_estimate() {
        // True source->target offset is +500 us.
        // Forward delay 30 us, target processing 20 us, reverse delay 70 us.
        let evidence = ClockCalibrationEvidence::derive(exchange(1_000, 1_530, 1_550, 1_120, 0)).unwrap();
        assert_eq!(evidence.offset_interval.lower_us(), 430);
        assert_eq!(evidence.offset_interval.upper_us(), 530);
        assert!(evidence.offset_interval.contains(500));
        assert_eq!(evidence.offset_interval.midpoint_us(), 480);
    }

    #[test]
    fn endpoint_uncertainty_widens_offset_interval() {
        let evidence = ClockCalibrationEvidence::derive(exchange(1_000, 1_530, 1_550, 1_120, 5)).unwrap();
        assert_eq!(evidence.offset_interval.lower_us(), 420);
        assert_eq!(evidence.offset_interval.upper_us(), 540);
        assert_eq!(evidence.offset_interval.symmetric_radius_us(), 60);
    }

    #[test]
    fn symmetric_delay_places_true_offset_at_midpoint() {
        let evidence = ClockCalibrationEvidence::derive(exchange(1_000, 1_550, 1_570, 1_120, 0)).unwrap();
        assert_eq!(evidence.offset_interval, ClockOffsetIntervalUs::new(450, 550).unwrap());
        assert_eq!(evidence.offset_interval.midpoint_us(), 500);
    }

    #[test]
    fn interval_intersection_tightens_without_averaging() {
        let first = ClockCalibrationEvidence::derive(exchange(1_000, 1_530, 1_550, 1_120, 0)).unwrap(); // [430, 530]
        let second = ClockCalibrationEvidence::derive(exchange(2_000, 2_560, 2_580, 2_100, 0)).unwrap(); // [480, 560]
        let consensus = CalibrationConsensus::from_evidence(&[first, second]).unwrap();
        assert_eq!(consensus.offset_interval, ClockOffsetIntervalUs::new(480, 530).unwrap());
        assert_eq!(consensus.exchange_count, 2);
    }

    #[test]
    fn disjoint_intervals_fail_instead_of_being_averaged() {
        let first = ClockCalibrationEvidence::derive(exchange(1_000, 1_530, 1_550, 1_120, 0)).unwrap();
        let second = ClockCalibrationEvidence::derive(exchange(2_000, 2_800, 2_820, 2_100, 0)).unwrap();
        let error = CalibrationConsensus::from_evidence(&[first, second]).unwrap_err();
        assert_eq!(error, CalibrationError::DisjointCalibrationIntervals);
    }

    #[test]
    fn missing_epoch_fails_closed() {
        let unepoched = TimeIntegrityReceipt::declared(source_domain())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(5));
        let error = FourTimestampExchange::new(
            TimestampEvidence::new(1_000, unepoched),
            stamp(1_530, target_domain(), target_epoch(), 5),
            stamp(1_550, target_domain(), target_epoch(), 5),
            stamp(1_120, source_domain(), source_epoch(), 5),
        )
        .unwrap_err();
        assert_eq!(
            error,
            CalibrationError::MissingEpoch {
                stamp: ExchangeStamp::SourceSend,
            }
        );
    }

    #[test]
    fn broken_continuity_fails_closed() {
        let broken = TimeIntegrityReceipt::declared(source_domain())
            .with_epoch(source_epoch())
            .with_continuity(ContinuityStatus::Broken)
            .with_uncertainty(TimeUncertainty::bounded(5));
        let error = FourTimestampExchange::new(
            TimestampEvidence::new(1_000, broken),
            stamp(1_530, target_domain(), target_epoch(), 5),
            stamp(1_550, target_domain(), target_epoch(), 5),
            stamp(1_120, source_domain(), source_epoch(), 5),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CalibrationError::ContinuityNotEstablished {
                stamp: ExchangeStamp::SourceSend,
                status: ContinuityStatus::Broken,
            }
        ));
    }

    #[test]
    fn definitely_inverted_local_order_fails_closed() {
        let error = FourTimestampExchange::new(
            stamp(2_000, source_domain(), source_epoch(), 0),
            stamp(1_530, target_domain(), target_epoch(), 0),
            stamp(1_550, target_domain(), target_epoch(), 0),
            stamp(1_000, source_domain(), source_epoch(), 0),
        )
        .unwrap_err();
        assert_eq!(error, CalibrationError::SourceLocalOrderImpossible);
    }

    #[test]
    fn declared_same_timebase_must_admit_zero_offset() {
        let domain = ClockDomainId::new("shared-clock").unwrap();
        let epoch = ClockEpochId::new("epoch-1").unwrap();
        let result = FourTimestampExchange::new(
            stamp(1_000, domain.clone(), epoch.clone(), 0),
            stamp(1_530, domain.clone(), epoch.clone(), 0),
            stamp(1_550, domain.clone(), epoch.clone(), 0),
            stamp(1_120, domain, epoch, 0),
        );
        assert!(matches!(
            result,
            Err(CalibrationError::DeclaredSameTimebaseContradiction { .. })
        ));
    }

    #[test]
    fn wire_roundtrip_revalidates_exchange() {
        let value = exchange(1_000, 1_530, 1_550, 1_120, 5);
        let json = serde_json::to_string(&value).unwrap();
        let decoded: FourTimestampExchange = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn unknown_wire_fields_fail_closed() {
        let value = exchange(1_000, 1_530, 1_550, 1_120, 5);
        let mut json = serde_json::to_value(&value).unwrap();
        json.as_object_mut().unwrap().insert("unsupported".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<FourTimestampExchange>(json).is_err());
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated sequences of protocol-bound chemical observations.
//!
//! A trace is intentionally structural rather than prescriptive: it guarantees
//! that observations belong to one protocol execution and are ordered, while
//! leaving protocol-specific phase requirements to experiment definitions.

use crate::{ChemicalModality, ChemicalObservation, SamplingPhase};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalTraceError {
    MissingSamplingContext,
    ProtocolMismatch { expected: String, actual: String },
    RunMismatch { expected: String, actual: String },
    ModalityMismatch {
        expected: ChemicalModality,
        actual: ChemicalModality,
    },
    ReplicateMismatch { expected: u32, actual: u32 },
    NonMonotonicTimestamp { previous_us: u64, next_us: u64 },
    StepRegression { previous: u32, next: u32 },
}

/// One internally consistent acquisition trace.
#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalTrace {
    protocol_id: String,
    run_id: String,
    modality: ChemicalModality,
    replicate: u32,
    observations: Vec<ChemicalObservation>,
}

impl ChemicalTrace {
    /// Start a trace from one protocol-bound observation.
    pub fn new(first: ChemicalObservation) -> Result<Self, ChemicalTraceError> {
        let sampling = first
            .sampling
            .as_ref()
            .ok_or(ChemicalTraceError::MissingSamplingContext)?;
        Ok(Self {
            protocol_id: sampling.protocol_id.clone(),
            run_id: sampling.run_id.clone(),
            modality: first.modality,
            replicate: sampling.replicate,
            observations: vec![first],
        })
    }

    pub fn protocol_id(&self) -> &str {
        &self.protocol_id
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn modality(&self) -> ChemicalModality {
        self.modality
    }

    pub fn replicate(&self) -> u32 {
        self.replicate
    }

    pub fn observations(&self) -> &[ChemicalObservation] {
        &self.observations
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    pub fn duration_us(&self) -> u64 {
        match (self.observations.first(), self.observations.last()) {
            (Some(first), Some(last)) => last.timestamp_us.saturating_sub(first.timestamp_us),
            _ => 0,
        }
    }

    /// Append after validating all trace invariants. Validation occurs before
    /// mutation, so a rejected observation leaves the trace unchanged.
    pub fn append(&mut self, observation: ChemicalObservation) -> Result<(), ChemicalTraceError> {
        self.validate_next(&observation)?;
        self.observations.push(observation);
        Ok(())
    }

    fn validate_next(&self, observation: &ChemicalObservation) -> Result<(), ChemicalTraceError> {
        let sampling = observation
            .sampling
            .as_ref()
            .ok_or(ChemicalTraceError::MissingSamplingContext)?;

        if sampling.protocol_id != self.protocol_id {
            return Err(ChemicalTraceError::ProtocolMismatch {
                expected: self.protocol_id.clone(),
                actual: sampling.protocol_id.clone(),
            });
        }
        if sampling.run_id != self.run_id {
            return Err(ChemicalTraceError::RunMismatch {
                expected: self.run_id.clone(),
                actual: sampling.run_id.clone(),
            });
        }
        if observation.modality != self.modality {
            return Err(ChemicalTraceError::ModalityMismatch {
                expected: self.modality,
                actual: observation.modality,
            });
        }
        if sampling.replicate != self.replicate {
            return Err(ChemicalTraceError::ReplicateMismatch {
                expected: self.replicate,
                actual: sampling.replicate,
            });
        }

        if let Some(previous) = self.observations.last() {
            if observation.timestamp_us <= previous.timestamp_us {
                return Err(ChemicalTraceError::NonMonotonicTimestamp {
                    previous_us: previous.timestamp_us,
                    next_us: observation.timestamp_us,
                });
            }
            let previous_step = previous
                .sampling
                .as_ref()
                .expect("trace invariant: stored observations have sampling context")
                .step_index;
            if sampling.step_index < previous_step {
                return Err(ChemicalTraceError::StepRegression {
                    previous: previous_step,
                    next: sampling.step_index,
                });
            }
        }
        Ok(())
    }

    /// Iterate observations collected during one acquisition phase.
    pub fn phase(&self, phase: SamplingPhase) -> impl Iterator<Item = &ChemicalObservation> {
        self.observations.iter().filter(move |observation| {
            observation
                .sampling
                .as_ref()
                .is_some_and(|sampling| sampling.phase == phase)
        })
    }

    /// Consecutive phase sequence, useful for experiment receipts without
    /// duplicating every frame-level phase label.
    pub fn phase_sequence(&self) -> Vec<SamplingPhase> {
        let mut sequence = Vec::new();
        for observation in &self.observations {
            let phase = observation
                .sampling
                .as_ref()
                .expect("trace invariant: stored observations have sampling context")
                .phase;
            if sequence.last().copied() != Some(phase) {
                sequence.push(phase);
            }
        }
        sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalModality, SamplingContext};

    fn observation(
        timestamp_us: u64,
        modality: ChemicalModality,
        protocol: &str,
        run: &str,
        phase: SamplingPhase,
        step: u32,
        replicate: u32,
    ) -> ChemicalObservation {
        let sampling = SamplingContext::new(protocol, run, phase, step)
            .unwrap()
            .with_replicate(replicate);
        ChemicalObservation::new(timestamp_us, modality, "fixture", vec![]).with_sampling(sampling)
    }

    #[test]
    fn trace_preserves_protocol_order_without_prescribing_phase_recipe() {
        let mut trace = ChemicalTrace::new(observation(
            1_000,
            ChemicalModality::Olfactory,
            "od001-v1",
            "run-a",
            SamplingPhase::Baseline,
            0,
            1,
        ))
        .unwrap();
        trace
            .append(observation(
                2_000,
                ChemicalModality::Olfactory,
                "od001-v1",
                "run-a",
                SamplingPhase::Exposure,
                1,
                1,
            ))
            .unwrap();
        trace
            .append(observation(
                3_000,
                ChemicalModality::Olfactory,
                "od001-v1",
                "run-a",
                SamplingPhase::Exposure,
                1,
                1,
            ))
            .unwrap();
        trace
            .append(observation(
                4_000,
                ChemicalModality::Olfactory,
                "od001-v1",
                "run-a",
                SamplingPhase::Purge,
                2,
                1,
            ))
            .unwrap();

        assert_eq!(trace.len(), 4);
        assert_eq!(trace.duration_us(), 3_000);
        assert_eq!(trace.phase(SamplingPhase::Exposure).count(), 2);
        assert_eq!(
            trace.phase_sequence(),
            vec![
                SamplingPhase::Baseline,
                SamplingPhase::Exposure,
                SamplingPhase::Purge,
            ]
        );
    }

    #[test]
    fn rejected_append_is_transactional() {
        let first = observation(
            2_000,
            ChemicalModality::Gustatory,
            "gt004-v1",
            "run-a",
            SamplingPhase::Exposure,
            2,
            0,
        );
        let mut trace = ChemicalTrace::new(first.clone()).unwrap();
        let bad = observation(
            1_000,
            ChemicalModality::Gustatory,
            "gt004-v1",
            "run-a",
            SamplingPhase::Rinse,
            3,
            0,
        );

        assert!(matches!(
            trace.append(bad),
            Err(ChemicalTraceError::NonMonotonicTimestamp { .. })
        ));
        assert_eq!(trace.observations(), &[first]);
    }

    #[test]
    fn trace_rejects_cross_run_and_cross_modality_contamination() {
        let first = observation(
            1_000,
            ChemicalModality::Olfactory,
            "od001-v1",
            "run-a",
            SamplingPhase::Baseline,
            0,
            0,
        );
        let mut trace = ChemicalTrace::new(first).unwrap();

        assert!(matches!(
            trace.append(observation(
                2_000,
                ChemicalModality::Olfactory,
                "od001-v1",
                "run-b",
                SamplingPhase::Exposure,
                1,
                0,
            )),
            Err(ChemicalTraceError::RunMismatch { .. })
        ));
        assert!(matches!(
            trace.append(observation(
                2_000,
                ChemicalModality::Gustatory,
                "od001-v1",
                "run-a",
                SamplingPhase::Exposure,
                1,
                0,
            )),
            Err(ChemicalTraceError::ModalityMismatch { .. })
        ));
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn trace_requires_sampling_context() {
        let plain = ChemicalObservation::new(
            1_000,
            ChemicalModality::Olfactory,
            "fixture",
            vec![],
        );
        assert!(matches!(
            ChemicalTrace::new(plain),
            Err(ChemicalTraceError::MissingSamplingContext)
        ));
    }
}

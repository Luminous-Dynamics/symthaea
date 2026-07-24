//! End-to-end orchestration through the common communication stages.

use crate::{
    CapabilityLevel, CommunicationError, CommunicationModel, CommunicationResult,
    CommunicationUnit, InteractionContext, MeaningHypothesis, SignalObservation,
};

#[derive(Clone, Debug)]
pub struct PipelineOutput<E, S> {
    pub observation: CommunicationResult<SignalObservation>,
    pub units: CommunicationResult<Vec<CommunicationUnit>>,
    pub encoding: CommunicationResult<E>,
    pub structure: CommunicationResult<S>,
    pub meanings: CommunicationResult<Vec<MeaningHypothesis>>,
}

/// Execute every stage while enforcing the maximum honest capability of each.
pub fn analyze<M: CommunicationModel>(
    model: &mut M,
    observation: SignalObservation,
    context: &InteractionContext,
) -> Result<PipelineOutput<M::Encoding, M::Structure>, CommunicationError> {
    observation.validate_identity()?;
    let observed = model.observe(observation)?;
    observed.validate(CapabilityLevel::Signal)?;
    let units = model.segment(&observed.value)?;
    units.validate(CapabilityLevel::Unit)?;
    let encoding = model.encode(&units.value)?;
    encoding.validate(CapabilityLevel::Unit)?;
    let structure = model.infer_structure(&encoding.value)?;
    structure.validate(CapabilityLevel::Structure)?;
    let meanings = model.propose_meanings(&structure.value, context)?;
    meanings.validate(CapabilityLevel::Intent)?;
    Ok(PipelineOutput {
        observation: observed,
        units,
        encoding,
        structure,
        meanings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CommunicationEvidence, DatasetSplit, EventFrame, EvidenceDomain, MeaningHypothesis,
        Provenance, ReplicationStatus,
    };

    fn stub_evidence() -> CommunicationEvidence {
        let mut e = CommunicationEvidence {
            id: String::new(),
            dataset_uri: "test:stub".into(),
            dataset_hash: "dhash".into(),
            model_hash: "mhash".into(),
            lineage: vec![],
            split: DatasetSplit::default(),
            evidence_records: vec![],
            preregistration_uri: None,
            replication: ReplicationStatus::Unreplicated,
            calibration: vec![],
            experimental: true, // explicit: stub results are always experimental
            domain: EvidenceDomain::HumanLanguage,
        };
        e.id = e.computed_id().unwrap();
        e
    }

    fn stub_provenance() -> Provenance {
        Provenance {
            provider: "stub".into(),
            provider_version: "0".into(),
            model_hash: "mhash".into(),
            feature_flags: vec![],
            transformations: vec![],
        }
    }

    // A minimal stub that produces valid (but experimental) results at every stage.
    struct StubModel;

    impl CommunicationModel for StubModel {
        type Encoding = Vec<f32>;
        type Structure = String;

        fn observe(
            &mut self,
            observation: SignalObservation,
        ) -> Result<CommunicationResult<SignalObservation>, CommunicationError> {
            Ok(CommunicationResult {
                value: observation,
                capability: CapabilityLevel::Signal,
                confidence: 0.5,
                alternatives: vec![],
                provenance: stub_provenance(),
                evidence: vec![stub_evidence()],
            })
        }

        fn segment(
            &mut self,
            _obs: &SignalObservation,
        ) -> Result<CommunicationResult<Vec<CommunicationUnit>>, CommunicationError> {
            Ok(CommunicationResult {
                value: vec![],
                capability: CapabilityLevel::Unit,
                confidence: 0.5,
                alternatives: vec![],
                provenance: stub_provenance(),
                evidence: vec![stub_evidence()],
            })
        }

        fn encode(
            &self,
            _units: &[CommunicationUnit],
        ) -> Result<CommunicationResult<Vec<f32>>, CommunicationError> {
            Ok(CommunicationResult {
                value: vec![0.1_f32],
                capability: CapabilityLevel::Unit,
                confidence: 0.5,
                alternatives: vec![],
                provenance: stub_provenance(),
                evidence: vec![stub_evidence()],
            })
        }

        fn infer_structure(
            &self,
            _enc: &Vec<f32>,
        ) -> Result<CommunicationResult<String>, CommunicationError> {
            Ok(CommunicationResult {
                value: "stub-structure".into(),
                capability: CapabilityLevel::Structure,
                confidence: 0.5,
                alternatives: vec![],
                provenance: stub_provenance(),
                evidence: vec![stub_evidence()],
            })
        }

        fn propose_meanings(
            &self,
            _s: &String,
            _ctx: &InteractionContext,
        ) -> Result<CommunicationResult<Vec<MeaningHypothesis>>, CommunicationError> {
            Ok(CommunicationResult {
                value: vec![],
                capability: CapabilityLevel::Intent,
                confidence: 0.5,
                alternatives: vec![],
                provenance: stub_provenance(),
                evidence: vec![stub_evidence()],
            })
        }

        fn update_from_outcome(
            &mut self,
            _h: &[MeaningHypothesis],
            _outcome: &EventFrame,
        ) -> Result<(), CommunicationError> {
            Ok(())
        }
    }

    // A model that claims Dialogue at the observation stage — must be rejected.
    struct OverclaimingModel;

    impl CommunicationModel for OverclaimingModel {
        type Encoding = ();
        type Structure = ();

        fn observe(
            &mut self,
            obs: SignalObservation,
        ) -> Result<CommunicationResult<SignalObservation>, CommunicationError> {
            Ok(CommunicationResult {
                value: obs,
                capability: CapabilityLevel::Dialogue, // overclaim
                confidence: 0.5,
                alternatives: vec![],
                provenance: stub_provenance(),
                evidence: vec![stub_evidence()],
            })
        }
        fn segment(
            &mut self,
            _: &SignalObservation,
        ) -> Result<CommunicationResult<Vec<CommunicationUnit>>, CommunicationError> {
            Err(CommunicationError::Provider("stub".into()))
        }
        fn encode(
            &self,
            _: &[CommunicationUnit],
        ) -> Result<CommunicationResult<()>, CommunicationError> {
            Err(CommunicationError::Provider("stub".into()))
        }
        fn infer_structure(&self, _: &()) -> Result<CommunicationResult<()>, CommunicationError> {
            Err(CommunicationError::Provider("stub".into()))
        }
        fn propose_meanings(
            &self,
            _: &(),
            _: &InteractionContext,
        ) -> Result<CommunicationResult<Vec<MeaningHypothesis>>, CommunicationError> {
            Err(CommunicationError::Provider("stub".into()))
        }
        fn update_from_outcome(
            &mut self,
            _: &[MeaningHypothesis],
            _: &EventFrame,
        ) -> Result<(), CommunicationError> {
            Ok(())
        }
    }

    fn make_observation() -> SignalObservation {
        let mut obs = SignalObservation {
            id: String::new(),
            modality: crate::Modality::Audio {
                sample_rate_hz: 16000,
                channels: 1,
            },
            samples: vec![0.0_f32; 16],
            features: Default::default(),
            original_text: None,
            normalized_text: None,
            uncertain_spans: vec![],
            timing: crate::TimeSpan {
                start_s: 0.0,
                end_s: 0.001,
            },
            location: None,
            calibration: crate::SensorCalibration::default(),
            source_identity: None,
            environment: Default::default(),
        };
        obs.refresh_id().unwrap();
        obs
    }

    #[test]
    fn stub_model_completes_without_error() {
        let result = analyze(
            &mut StubModel,
            make_observation(),
            &InteractionContext::default(),
        );
        assert!(result.is_ok(), "{result:?}");
        let out = result.unwrap();
        assert_eq!(out.observation.capability, CapabilityLevel::Signal);
        assert_eq!(out.structure.capability, CapabilityLevel::Structure);
    }

    #[test]
    fn overclaiming_model_is_rejected_at_observation_stage() {
        let result = analyze(
            &mut OverclaimingModel,
            make_observation(),
            &InteractionContext::default(),
        );
        assert!(
            matches!(result, Err(CommunicationError::InvalidResult(_))),
            "expected InvalidResult, got {result:?}"
        );
    }

    #[test]
    fn invalid_timing_is_caught_before_model_is_called() {
        let mut obs = make_observation();
        // Break timing after ID is set — validate_identity must detect it.
        obs.timing.end_s = -1.0;
        obs.id = String::new(); // no longer content-addressed
        let result = analyze(&mut StubModel, obs, &InteractionContext::default());
        assert!(matches!(
            result,
            Err(CommunicationError::InvalidObservation(_))
        ));
    }
}

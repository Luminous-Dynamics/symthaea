// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-only sampling-design theorem for Economic Science.
//!
//! A declared target population, a large sample, or the presence of weights does
//! not establish representativeness. Population inference needs an explicit
//! sampling/coverage contract whose evidence class survives admission.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SamplingClass {
    CompleteFrameEnumeration,
    ProbabilitySample,
    AdministrativeCoverage,
    NonProbabilitySample,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SamplingAdmission {
    CompleteFrameEnumeration,
    ProbabilitySample {
        inclusion_design_id: String,
        weighting_policy_id: Option<String>,
        nonresponse_policy_id: String,
    },
    AdministrativeCoverage {
        limitations: String,
    },
    NonProbabilitySample {
        weighting_policy_id: Option<String>,
        limitations: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SamplingError {
    EmptyText(&'static str),
    InvalidDesignContract,
    EmptySample,
    TargetPopulationMismatch,
    SamplingFrameMismatch,
    SamplingClassMismatch,
    InclusionDesignMismatch,
    WeightingPolicyMismatch,
    NonresponsePolicyMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SamplingSpecification {
    specification_id: String,
    target_population_id: String,
    sampling_frame_id: String,
    class: SamplingClass,
    inclusion_design_id: Option<String>,
    weighting_policy_id: Option<String>,
    nonresponse_policy_id: Option<String>,
    limitations: Option<String>,
}

impl SamplingSpecification {
    #[allow(clippy::too_many_arguments)]
    fn new(
        specification_id: impl Into<String>,
        target_population_id: impl Into<String>,
        sampling_frame_id: impl Into<String>,
        class: SamplingClass,
        inclusion_design_id: Option<String>,
        weighting_policy_id: Option<String>,
        nonresponse_policy_id: Option<String>,
        limitations: Option<String>,
    ) -> Result<Self, SamplingError> {
        let specification_id = specification_id.into();
        let target_population_id = target_population_id.into();
        let sampling_frame_id = sampling_frame_id.into();
        for (field, value) in [
            ("sampling specification id", specification_id.as_str()),
            ("target population id", target_population_id.as_str()),
            ("sampling frame id", sampling_frame_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(SamplingError::EmptyText(field));
            }
        }

        let nonempty = |value: &Option<String>| {
            value
                .as_ref()
                .is_some_and(|item| !item.trim().is_empty())
        };
        if inclusion_design_id.as_ref().is_some_and(|value| value.trim().is_empty())
            || weighting_policy_id.as_ref().is_some_and(|value| value.trim().is_empty())
            || nonresponse_policy_id
                .as_ref()
                .is_some_and(|value| value.trim().is_empty())
            || limitations.as_ref().is_some_and(|value| value.trim().is_empty())
        {
            return Err(SamplingError::InvalidDesignContract);
        }

        match class {
            SamplingClass::CompleteFrameEnumeration => {
                if inclusion_design_id.is_some()
                    || weighting_policy_id.is_some()
                    || nonresponse_policy_id.is_some()
                    || limitations.is_some()
                {
                    return Err(SamplingError::InvalidDesignContract);
                }
            }
            SamplingClass::ProbabilitySample => {
                if !nonempty(&inclusion_design_id)
                    || !nonempty(&nonresponse_policy_id)
                    || limitations.is_some()
                {
                    return Err(SamplingError::InvalidDesignContract);
                }
            }
            SamplingClass::AdministrativeCoverage => {
                if inclusion_design_id.is_some()
                    || weighting_policy_id.is_some()
                    || nonresponse_policy_id.is_some()
                    || !nonempty(&limitations)
                {
                    return Err(SamplingError::InvalidDesignContract);
                }
            }
            SamplingClass::NonProbabilitySample => {
                if inclusion_design_id.is_some()
                    || nonresponse_policy_id.is_some()
                    || !nonempty(&limitations)
                {
                    return Err(SamplingError::InvalidDesignContract);
                }
            }
        }

        Ok(Self {
            specification_id,
            target_population_id,
            sampling_frame_id,
            class,
            inclusion_design_id,
            weighting_policy_id,
            nonresponse_policy_id,
            limitations,
        })
    }

    fn specification_id(&self) -> &str {
        &self.specification_id
    }

    fn qualify(
        &self,
        observed: &ObservedSampleDescriptor,
    ) -> Result<SamplingAdmission, SamplingError> {
        if observed.sample_size == 0 {
            return Err(SamplingError::EmptySample);
        }
        if observed.target_population_id != self.target_population_id {
            return Err(SamplingError::TargetPopulationMismatch);
        }
        if observed.sampling_frame_id != self.sampling_frame_id {
            return Err(SamplingError::SamplingFrameMismatch);
        }
        if observed.class != self.class {
            return Err(SamplingError::SamplingClassMismatch);
        }
        if observed.inclusion_design_id != self.inclusion_design_id {
            return Err(SamplingError::InclusionDesignMismatch);
        }
        if observed.weighting_policy_id != self.weighting_policy_id {
            return Err(SamplingError::WeightingPolicyMismatch);
        }
        if observed.nonresponse_policy_id != self.nonresponse_policy_id {
            return Err(SamplingError::NonresponsePolicyMismatch);
        }

        match self.class {
            SamplingClass::CompleteFrameEnumeration => {
                Ok(SamplingAdmission::CompleteFrameEnumeration)
            }
            SamplingClass::ProbabilitySample => Ok(SamplingAdmission::ProbabilitySample {
                inclusion_design_id: self
                    .inclusion_design_id
                    .clone()
                    .expect("validated probability inclusion design"),
                weighting_policy_id: self.weighting_policy_id.clone(),
                nonresponse_policy_id: self
                    .nonresponse_policy_id
                    .clone()
                    .expect("validated probability nonresponse policy"),
            }),
            SamplingClass::AdministrativeCoverage => {
                Ok(SamplingAdmission::AdministrativeCoverage {
                    limitations: self
                        .limitations
                        .clone()
                        .expect("validated administrative limitations"),
                })
            }
            SamplingClass::NonProbabilitySample => {
                Ok(SamplingAdmission::NonProbabilitySample {
                    weighting_policy_id: self.weighting_policy_id.clone(),
                    limitations: self
                        .limitations
                        .clone()
                        .expect("validated nonprobability limitations"),
                })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservedSampleDescriptor {
    target_population_id: String,
    sampling_frame_id: String,
    class: SamplingClass,
    sample_size: u64,
    inclusion_design_id: Option<String>,
    weighting_policy_id: Option<String>,
    nonresponse_policy_id: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn observed(
    target_population_id: &str,
    sampling_frame_id: &str,
    class: SamplingClass,
    sample_size: u64,
    inclusion_design_id: Option<&str>,
    weighting_policy_id: Option<&str>,
    nonresponse_policy_id: Option<&str>,
) -> ObservedSampleDescriptor {
    ObservedSampleDescriptor {
        target_population_id: target_population_id.into(),
        sampling_frame_id: sampling_frame_id.into(),
        class,
        sample_size,
        inclusion_design_id: inclusion_design_id.map(str::to_string),
        weighting_policy_id: weighting_policy_id.map(str::to_string),
        nonresponse_policy_id: nonresponse_policy_id.map(str::to_string),
    }
}

fn probability_spec() -> SamplingSpecification {
    SamplingSpecification::new(
        "sampling:household-probability-v1",
        "population:civilian-noninstitutional-16plus",
        "frame:household-address-frame-v3",
        SamplingClass::ProbabilitySample,
        Some("inclusion:stratified-multistage-v2".into()),
        Some("weights:survey-design-v4".into()),
        Some("nonresponse:adjustment-v3".into()),
        None,
    )
    .unwrap()
}

#[test]
fn probability_sample_preserves_design_weighting_and_nonresponse_identity() {
    let spec = probability_spec();
    let sample = observed(
        "population:civilian-noninstitutional-16plus",
        "frame:household-address-frame-v3",
        SamplingClass::ProbabilitySample,
        60_000,
        Some("inclusion:stratified-multistage-v2"),
        Some("weights:survey-design-v4"),
        Some("nonresponse:adjustment-v3"),
    );

    assert_eq!(spec.specification_id(), "sampling:household-probability-v1");
    assert_eq!(
        spec.qualify(&sample),
        Ok(SamplingAdmission::ProbabilitySample {
            inclusion_design_id: "inclusion:stratified-multistage-v2".into(),
            weighting_policy_id: Some("weights:survey-design-v4".into()),
            nonresponse_policy_id: "nonresponse:adjustment-v3".into(),
        })
    );
}

#[test]
fn same_target_population_does_not_make_a_different_frame_equivalent() {
    let spec = probability_spec();
    let wrong_frame = observed(
        "population:civilian-noninstitutional-16plus",
        "frame:online-volunteer-panel",
        SamplingClass::ProbabilitySample,
        60_000,
        Some("inclusion:stratified-multistage-v2"),
        Some("weights:survey-design-v4"),
        Some("nonresponse:adjustment-v3"),
    );

    assert_eq!(
        spec.qualify(&wrong_frame),
        Err(SamplingError::SamplingFrameMismatch)
    );
}

#[test]
fn very_large_nonprobability_sample_does_not_become_probability_evidence() {
    let spec = SamplingSpecification::new(
        "sampling:online-convenience-v1",
        "population:working-age-adults",
        "frame:platform-volunteers",
        SamplingClass::NonProbabilitySample,
        None,
        None,
        None,
        Some("self-selection and platform coverage are not population-random".into()),
    )
    .unwrap();
    let million = observed(
        "population:working-age-adults",
        "frame:platform-volunteers",
        SamplingClass::NonProbabilitySample,
        1_000_000,
        None,
        None,
        None,
    );

    let admission = spec.qualify(&million).unwrap();
    assert_eq!(
        admission,
        SamplingAdmission::NonProbabilitySample {
            weighting_policy_id: None,
            limitations: "self-selection and platform coverage are not population-random".into(),
        }
    );
    assert_ne!(
        admission,
        SamplingAdmission::ProbabilitySample {
            inclusion_design_id: "not-applicable".into(),
            weighting_policy_id: None,
            nonresponse_policy_id: "not-applicable".into(),
        }
    );
}

#[test]
fn weighting_a_nonprobability_sample_does_not_upgrade_its_design_class() {
    let spec = SamplingSpecification::new(
        "sampling:weighted-convenience-v1",
        "population:working-age-adults",
        "frame:platform-volunteers",
        SamplingClass::NonProbabilitySample,
        None,
        Some("weights:poststratification-v1".into()),
        None,
        Some("weights adjust margins but do not create known inclusion probabilities".into()),
    )
    .unwrap();
    let sample = observed(
        "population:working-age-adults",
        "frame:platform-volunteers",
        SamplingClass::NonProbabilitySample,
        500_000,
        None,
        Some("weights:poststratification-v1"),
        None,
    );

    assert_eq!(
        spec.qualify(&sample),
        Ok(SamplingAdmission::NonProbabilitySample {
            weighting_policy_id: Some("weights:poststratification-v1".into()),
            limitations: "weights adjust margins but do not create known inclusion probabilities"
                .into(),
        })
    );
}

#[test]
fn administrative_coverage_remains_distinct_from_complete_enumeration() {
    let spec = SamplingSpecification::new(
        "sampling:tax-records-v1",
        "population:resident-workers",
        "frame:tax-filers",
        SamplingClass::AdministrativeCoverage,
        None,
        None,
        None,
        Some("nonfilers and informal workers are outside the administrative frame".into()),
    )
    .unwrap();
    let records = observed(
        "population:resident-workers",
        "frame:tax-filers",
        SamplingClass::AdministrativeCoverage,
        20_000_000,
        None,
        None,
        None,
    );

    let admission = spec.qualify(&records).unwrap();
    assert_eq!(
        admission,
        SamplingAdmission::AdministrativeCoverage {
            limitations: "nonfilers and informal workers are outside the administrative frame"
                .into(),
        }
    );
    assert_ne!(admission, SamplingAdmission::CompleteFrameEnumeration);
}

#[test]
fn complete_frame_enumeration_cannot_carry_hidden_sampling_adjustments() {
    assert_eq!(
        SamplingSpecification::new(
            "sampling:census-like-v1",
            "population:registered-units",
            "frame:complete-register-v1",
            SamplingClass::CompleteFrameEnumeration,
            None,
            Some("weights:hidden".into()),
            None,
            None,
        ),
        Err(SamplingError::InvalidDesignContract)
    );

    let spec = SamplingSpecification::new(
        "sampling:complete-register-v1",
        "population:registered-units",
        "frame:complete-register-v1",
        SamplingClass::CompleteFrameEnumeration,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let complete = observed(
        "population:registered-units",
        "frame:complete-register-v1",
        SamplingClass::CompleteFrameEnumeration,
        10_000,
        None,
        None,
        None,
    );
    assert_eq!(
        spec.qualify(&complete),
        Ok(SamplingAdmission::CompleteFrameEnumeration)
    );
}

#[test]
fn probability_design_requires_inclusion_and_nonresponse_contracts() {
    assert_eq!(
        SamplingSpecification::new(
            "sampling:broken-probability-v1",
            "population:adults",
            "frame:addresses",
            SamplingClass::ProbabilitySample,
            None,
            Some("weights:v1".into()),
            None,
            None,
        ),
        Err(SamplingError::InvalidDesignContract)
    );
}

#[test]
fn empty_sample_never_receives_sampling_evidence_credit() {
    let spec = probability_spec();
    let empty = observed(
        "population:civilian-noninstitutional-16plus",
        "frame:household-address-frame-v3",
        SamplingClass::ProbabilitySample,
        0,
        Some("inclusion:stratified-multistage-v2"),
        Some("weights:survey-design-v4"),
        Some("nonresponse:adjustment-v3"),
    );
    assert_eq!(spec.qualify(&empty), Err(SamplingError::EmptySample));
}

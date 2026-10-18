//! Type-safe routing for materially different uncertainty semantics.
//!
//! This module does not certify the source or calibrate probabilities. It
//! requires an explicit qualification receipt/reference and routes the declared
//! uncertainty semantics to a compatible analysis family; out-of-domain inputs
//! fail closed to abstention.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UncertaintyEvidenceKind {
    /// A point distribution with its qualification process and validation corpus referenced.
    PointProbabilityDistribution {
        qualification_receipt_id: String,
        validation_corpus_digest: String,
    },
    /// Per-state lower/upper probability bounds, not output/model-discrepancy bounds.
    BoundedStateProbabilities {
        bounds_receipt_id: String,
    },
    /// A conservative set of compatible outcomes/actions without a probability model.
    OutcomeCompatibilitySet {
        compatibility_receipt_id: String,
    },
    /// An envelope for model output discrepancy; not a probability distribution.
    ModelDiscrepancyEnvelope {
        envelope_receipt_id: String,
    },
    /// The decision lies outside the declared support of an assessment.
    OutOfDomain {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UncertaintyAssessmentRef {
    pub scenario_id: String,
    pub producer_version: String,
    pub source_manifest_digest: String,
    pub evidence_kind: UncertaintyEvidenceKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatibleAnalysisRoute {
    ExpectedValueOrValueOfInformation,
    RobustProbabilityRanking,
    ConservativeCompatibilityPlanning,
    DiscrepancyAwareAssessment,
    Abstain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UncertaintyRoutingError {
    MissingScenarioId,
    MissingProducerVersion,
    MissingSourceManifestDigest,
    MalformedSourceManifestDigest,
    MissingQualificationReceipt,
    MissingValidationCorpusDigest,
    MalformedValidationCorpusDigest,
    MissingCompatibilityReceipt,
    MissingDiscrepancyEnvelopeReceipt,
    MissingOutOfDomainReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UncertaintyRoutingReceipt {
    pub scenario_id: String,
    /// Version of the producer that emitted the referenced uncertainty assessment.
    pub producer_version: String,
    /// Full typed input is retained; route/evidence_reference alone are lossy summaries.
    pub evidence_kind: UncertaintyEvidenceKind,
    pub route: CompatibleAnalysisRoute,
    /// The evidence/qualification reference used for routing.
    pub evidence_reference: String,
    pub source_manifest_digest: String,
    /// This is a semantic routing decision, not a certification of the evidence.
    pub limitation: String,
}

/// Checks SHA-256 digest syntax only; it does not verify the referenced content.
///
/// Both plain 64-character hexadecimal and the explicit sha256: prefix are
/// accepted so producers can share a simple stable representation.
fn has_sha256_digest_shape(value: &str) -> bool {
    let hex = value.strip_prefix("sha256:").unwrap_or(value);
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl UncertaintyAssessmentRef {
    pub fn route(&self) -> Result<UncertaintyRoutingReceipt, UncertaintyRoutingError> {
        if self.scenario_id.trim().is_empty() {
            return Err(UncertaintyRoutingError::MissingScenarioId);
        }
        if self.producer_version.trim().is_empty() {
            return Err(UncertaintyRoutingError::MissingProducerVersion);
        }
        if self.source_manifest_digest.trim().is_empty() {
            return Err(UncertaintyRoutingError::MissingSourceManifestDigest);
        }
        if !has_sha256_digest_shape(&self.source_manifest_digest) {
            return Err(UncertaintyRoutingError::MalformedSourceManifestDigest);
        }

        let (route, evidence_reference, limitation) = match &self.evidence_kind {
            UncertaintyEvidenceKind::PointProbabilityDistribution {
                qualification_receipt_id,
                validation_corpus_digest,
            } => {
                if qualification_receipt_id.trim().is_empty() {
                    return Err(UncertaintyRoutingError::MissingQualificationReceipt);
                }
                if validation_corpus_digest.trim().is_empty() {
                    return Err(UncertaintyRoutingError::MissingValidationCorpusDigest);
                }
                if !has_sha256_digest_shape(validation_corpus_digest) {
                    return Err(UncertaintyRoutingError::MalformedValidationCorpusDigest);
                }
                (
                    CompatibleAnalysisRoute::ExpectedValueOrValueOfInformation,
                    qualification_receipt_id.clone(),
                    "Point-probability calculations are conditional on the declared qualification and validation evidence; this router does not validate calibration.".to_owned(),
                )
            }
            UncertaintyEvidenceKind::BoundedStateProbabilities { bounds_receipt_id } => {
                if bounds_receipt_id.trim().is_empty() {
                    return Err(UncertaintyRoutingError::MissingQualificationReceipt);
                }
                (
                    CompatibleAnalysisRoute::RobustProbabilityRanking,
                    bounds_receipt_id.clone(),
                    "Only probability-mass ambiguity is routed here; payoff/model discrepancy and out-of-domain states require separate treatment.".to_owned(),
                )
            }
            UncertaintyEvidenceKind::OutcomeCompatibilitySet {
                compatibility_receipt_id,
            } => {
                if compatibility_receipt_id.trim().is_empty() {
                    return Err(UncertaintyRoutingError::MissingCompatibilityReceipt);
                }
                (
                    CompatibleAnalysisRoute::ConservativeCompatibilityPlanning,
                    compatibility_receipt_id.clone(),
                    "Compatibility sets do not imply probabilities; no expected value or EVSI may be inferred from this route alone.".to_owned(),
                )
            }
            UncertaintyEvidenceKind::ModelDiscrepancyEnvelope {
                envelope_receipt_id,
            } => {
                if envelope_receipt_id.trim().is_empty() {
                    return Err(UncertaintyRoutingError::MissingDiscrepancyEnvelopeReceipt);
                }
                (
                    CompatibleAnalysisRoute::DiscrepancyAwareAssessment,
                    envelope_receipt_id.clone(),
                    "Model-output discrepancy is not a state-probability interval; preserve the envelope and its validation/coverage limits.".to_owned(),
                )
            }
            UncertaintyEvidenceKind::OutOfDomain { reason } => {
                if reason.trim().is_empty() {
                    return Err(UncertaintyRoutingError::MissingOutOfDomainReason);
                }
                (
                    CompatibleAnalysisRoute::Abstain,
                    format!("out-of-domain:{reason}"),
                    "No in-domain decision estimate is emitted; obtain qualified evidence or revise the model domain.".to_owned(),
                )
            }
        };

        Ok(UncertaintyRoutingReceipt {
            scenario_id: self.scenario_id.clone(),
            producer_version: self.producer_version.clone(),
            evidence_kind: self.evidence_kind.clone(),
            route,
            evidence_reference,
            source_manifest_digest: self.source_manifest_digest.clone(),
            limitation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(kind: UncertaintyEvidenceKind) -> UncertaintyAssessmentRef {
        UncertaintyAssessmentRef {
            scenario_id: "ICCB-UNCERTAINTY-ROUTING-001".into(),
            producer_version: "fixture-producer-v1".into(),
            source_manifest_digest:
                "sha256:937a22712ea5c20073d2c8a226f1f4a9c2c30aba40a497f2412a34426f986535"
                    .into(),
            evidence_kind: kind,
        }
    }

    #[test]
    fn point_probabilities_route_to_ev_not_compatibility_planner() {
        let reference = base(UncertaintyEvidenceKind::PointProbabilityDistribution {
            qualification_receipt_id: "qual-01".into(),
            validation_corpus_digest:
                "sha256:21c1af4953138dc2e5e83e78aa4254f39c8c54d2d74ad8a13b1fdfba7193f6d4"
                    .into(),
        });
        let receipt = reference.route().unwrap();
        assert_eq!(
            receipt.route,
            CompatibleAnalysisRoute::ExpectedValueOrValueOfInformation
        );
        assert!(receipt.limitation.contains("does not validate calibration"));
        assert_eq!(receipt.producer_version, "fixture-producer-v1");
        assert_eq!(receipt.evidence_kind, reference.evidence_kind);
    }

    #[test]
    fn probability_bounds_route_to_robust_probability_ranking() {
        let reference = base(UncertaintyEvidenceKind::BoundedStateProbabilities {
            bounds_receipt_id: "bounds-01".into(),
        });
        assert_eq!(
            reference.route().unwrap().route,
            CompatibleAnalysisRoute::RobustProbabilityRanking
        );
    }

    #[test]
    fn compatibility_sets_never_route_to_point_probability_evsi() {
        let reference = base(UncertaintyEvidenceKind::OutcomeCompatibilitySet {
            compatibility_receipt_id: "compat-01".into(),
        });
        let receipt = reference.route().unwrap();
        assert_eq!(
            receipt.route,
            CompatibleAnalysisRoute::ConservativeCompatibilityPlanning
        );
        assert!(receipt.limitation.contains("do not imply probabilities"));
    }

    #[test]
    fn model_discrepancy_envelopes_do_not_route_as_probability_intervals() {
        let reference = base(UncertaintyEvidenceKind::ModelDiscrepancyEnvelope {
            envelope_receipt_id: "discrepancy-01".into(),
        });
        assert_eq!(
            reference.route().unwrap().route,
            CompatibleAnalysisRoute::DiscrepancyAwareAssessment
        );
    }

    #[test]
    fn out_of_domain_inputs_route_to_abstention() {
        let reference = base(UncertaintyEvidenceKind::OutOfDomain {
            reason: "missing-source-class".into(),
        });
        assert_eq!(
            reference.route().unwrap().route,
            CompatibleAnalysisRoute::Abstain
        );
    }

    #[test]
    fn missing_qualification_or_source_metadata_fails_closed() {
        let missing_qualification = base(UncertaintyEvidenceKind::PointProbabilityDistribution {
            qualification_receipt_id: " ".into(),
            validation_corpus_digest:
                "sha256:21c1af4953138dc2e5e83e78aa4254f39c8c54d2d74ad8a13b1fdfba7193f6d4"
                    .into(),
        });
        assert_eq!(
            missing_qualification.route(),
            Err(UncertaintyRoutingError::MissingQualificationReceipt)
        );

        let mut missing_manifest = base(UncertaintyEvidenceKind::BoundedStateProbabilities {
            bounds_receipt_id: "bounds-02".into(),
        });
        missing_manifest.source_manifest_digest.clear();
        assert_eq!(
            missing_manifest.route(),
            Err(UncertaintyRoutingError::MissingSourceManifestDigest)
        );
    }

    #[test]
    fn point_probability_route_requires_validation_corpus_digest() {
        let reference = base(UncertaintyEvidenceKind::PointProbabilityDistribution {
            qualification_receipt_id: "qual-03".into(),
            validation_corpus_digest: " ".into(),
        });
        assert_eq!(
            reference.route(),
            Err(UncertaintyRoutingError::MissingValidationCorpusDigest)
        );
    }

    #[test]
    fn missing_compatibility_envelope_or_out_of_domain_reason_fails_closed() {
        let missing_compatibility = base(UncertaintyEvidenceKind::OutcomeCompatibilitySet {
            compatibility_receipt_id: String::new(),
        });
        assert_eq!(
            missing_compatibility.route(),
            Err(UncertaintyRoutingError::MissingCompatibilityReceipt)
        );

        let missing_discrepancy = base(UncertaintyEvidenceKind::ModelDiscrepancyEnvelope {
            envelope_receipt_id: String::new(),
        });
        assert_eq!(
            missing_discrepancy.route(),
            Err(UncertaintyRoutingError::MissingDiscrepancyEnvelopeReceipt)
        );

        let missing_reason = base(UncertaintyEvidenceKind::OutOfDomain {
            reason: " ".into(),
        });
        assert_eq!(
            missing_reason.route(),
            Err(UncertaintyRoutingError::MissingOutOfDomainReason)
        );
    }

    #[test]
    fn invalid_common_metadata_fails_closed() {
        let mut reference = base(UncertaintyEvidenceKind::BoundedStateProbabilities {
            bounds_receipt_id: "bounds-03".into(),
        });
        reference.scenario_id.clear();
        assert_eq!(
            reference.route(),
            Err(UncertaintyRoutingError::MissingScenarioId)
        );

        let mut reference = base(UncertaintyEvidenceKind::BoundedStateProbabilities {
            bounds_receipt_id: "bounds-04".into(),
        });
        reference.producer_version.clear();
        assert_eq!(
            reference.route(),
            Err(UncertaintyRoutingError::MissingProducerVersion)
        );
    }

    #[test]
    fn malformed_digest_shapes_fail_closed() {
        let mut reference = base(UncertaintyEvidenceKind::BoundedStateProbabilities {
            bounds_receipt_id: "bounds-05".into(),
        });
        reference.source_manifest_digest = "sha256:synthetic-manifest".into();
        assert_eq!(
            reference.route(),
            Err(UncertaintyRoutingError::MalformedSourceManifestDigest)
        );

        let mut reference = base(UncertaintyEvidenceKind::PointProbabilityDistribution {
            qualification_receipt_id: "qual-04".into(),
            validation_corpus_digest: "sha256:heldout-cohort".into(),
        });
        assert_eq!(
            reference.route(),
            Err(UncertaintyRoutingError::MalformedValidationCorpusDigest)
        );
    }

    #[test]
    fn digest_shape_accepts_prefixed_and_unprefixed_sha256_hex() {
        assert!(has_sha256_digest_shape(
            "sha256:937a22712ea5c20073d2c8a226f1f4a9c2c30aba40a497f2412a34426f986535"
        ));
        assert!(has_sha256_digest_shape(
            "937a22712ea5c20073d2c8a226f1f4a9c2c30aba40a497f2412a34426f986535"
        ));
        assert!(!has_sha256_digest_shape("sha256:"));
        assert!(!has_sha256_digest_shape(&"g".repeat(64)));
    }
}

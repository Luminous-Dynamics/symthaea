//! SYM-CIV-003: independent evaluator / oracle / authority smoke control.
//!
//! This executable is deliberately dependency-free. It demonstrates a narrow
//! capability boundary rather than claiming real evaluator independence.
//!
//! Core non-equivalences:
//! candidate/model != evaluator
//! evaluation result != truth
//! evaluation evidence != authorization
//! authorization != execution
//! oracle != authority

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Digest(&'static str);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Verdict {
    Pass,
    Fail,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CandidateRevision {
    candidate: Digest,
    parent: Option<Digest>,
    code_lineage: Digest,
    change_set: Digest,
}

impl CandidateRevision {
    fn digest(self) -> Digest {
        // The smoke fixture uses content-address-like stable identities.
        // A production implementation must bind the canonical artifact bytes.
        match (self.candidate, self.change_set) {
            (Digest("candidate-v1"), Digest("candidate-expected")) => Digest("candidate-v1"),
            (Digest("candidate-v1"), Digest("candidate-tampered")) => Digest("candidate-tampered"),
            (Digest("candidate-degraded"), Digest("candidate-degraded")) => {
                Digest("candidate-degraded")
            }
            _ => Digest("candidate-other"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EvaluationProfile {
    evaluator: Digest,
    profile: Digest,
    holdout: Digest,
    evaluator_code_lineage: Digest,
    evaluator_data_lineage: Digest,
    authority_lineage: Digest,
    valid_from_epoch: u64,
    valid_until_epoch: u64,
}

impl EvaluationProfile {
    fn current(self, epoch: u64) -> bool {
        self.valid_from_epoch <= epoch && epoch <= self.valid_until_epoch
    }

    fn digest(self) -> Digest {
        // Stable fixture identity for this smoke. The profile field is part of
        // the bound evaluation subject rather than free-form metadata.
        match self.profile {
            Digest("evaluation-profile-v1") => Digest("evaluator-profile-v1"),
            _ => Digest("evaluator-profile-other"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Observation {
    id: Digest,
    epoch: u64,
    measured_value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EvaluationReceipt {
    candidate: Digest,
    evaluator: Digest,
    profile: Digest,
    holdout: Digest,
    observation: Digest,
    verdict: Verdict,
    issued_epoch: u64,
}

impl EvaluationReceipt {
    fn digest(self) -> Digest {
        match self.verdict {
            Verdict::Pass => Digest("evaluation-pass-v1"),
            Verdict::Fail => Digest("evaluation-fail-v1"),
            Verdict::Indeterminate => Digest("evaluation-indeterminate-v1"),
        }
    }

    fn matches_subject(self, profile: EvaluationProfile, candidate: CandidateRevision) -> bool {
        self.candidate == candidate.digest()
            && self.evaluator == profile.evaluator
            && self.profile == profile.digest()
            && self.holdout == profile.holdout
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObservationAdmission {
    Accepted,
    Duplicate,
}

struct Evaluator {
    profile: EvaluationProfile,
    // The oracle/reference state never crosses the evaluator boundary.
    oracle_holdout: Digest,
    oracle_expected_value: u64,
    consumed_observations: BTreeSet<Digest>,
}

impl Evaluator {
    fn new(profile: EvaluationProfile, oracle_holdout: Digest, oracle_expected_value: u64) -> Self {
        assert_eq!(profile.holdout, oracle_holdout, "oracle must bind the declared holdout");
        Self {
            profile,
            oracle_holdout,
            oracle_expected_value,
            consumed_observations: BTreeSet::new(),
        }
    }

    fn admit_observation(&mut self, observation: Observation) -> ObservationAdmission {
        if !self.consumed_observations.insert(observation.id) {
            return ObservationAdmission::Duplicate;
        }
        ObservationAdmission::Accepted
    }

    fn evaluate(
        &mut self,
        candidate: CandidateRevision,
        observation: Observation,
    ) -> Result<EvaluationReceipt, &'static str> {
        if !self.profile.current(observation.epoch) {
            return Err("stale evaluator profile");
        }
        if self.admit_observation(observation) == ObservationAdmission::Duplicate {
            return Err("duplicate observation replay");
        }

        // The candidate receives no oracle state. Only the evaluator can inspect
        // the reference value and decide the local experimental verdict.
        let verdict = if candidate.change_set == Digest("candidate-degraded") {
            Verdict::Fail
        } else if observation.measured_value == self.oracle_expected_value {
            Verdict::Pass
        } else {
            Verdict::Indeterminate
        };

        Ok(EvaluationReceipt {
            candidate: candidate.digest(),
            evaluator: self.profile.evaluator,
            profile: self.profile.digest(),
            holdout: self.oracle_holdout,
            observation: observation.id,
            verdict,
            issued_epoch: observation.epoch,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IndependenceFinding {
    SharedCodeLineage,
    SharedDataLineage,
    SharedAuthorityLineage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IndependenceRecord {
    candidate_code_lineage: Digest,
    evaluator_code_lineage: Digest,
    candidate_data_lineage: Digest,
    evaluator_data_lineage: Digest,
    candidate_authority_lineage: Digest,
    evaluator_authority_lineage: Digest,
}

impl IndependenceRecord {
    fn findings(self) -> [Option<IndependenceFinding>; 3] {
        [
            (self.candidate_code_lineage == self.evaluator_code_lineage)
                .then_some(IndependenceFinding::SharedCodeLineage),
            (self.candidate_data_lineage == self.evaluator_data_lineage)
                .then_some(IndependenceFinding::SharedDataLineage),
            (self.candidate_authority_lineage == self.evaluator_authority_lineage)
                .then_some(IndependenceFinding::SharedAuthorityLineage),
        ]
    }
}

// Evaluation evidence is intentionally not a capability to deploy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PromotionAuthorization {
    authority: Digest,
    evaluation: Digest,
    scope: Digest,
    expires_at_epoch: u64,
    allowed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeploymentReceipt {
    candidate: Digest,
    evaluation: Digest,
    authority: Digest,
    scope: Digest,
    expires_at_epoch: u64,
}

fn issue_deployment_receipt(
    evaluation: EvaluationReceipt,
    promotion: PromotionAuthorization,
    now_epoch: u64,
) -> Result<DeploymentReceipt, &'static str> {
    if evaluation.verdict != Verdict::Pass {
        return Err("only an explicit Pass may be considered for promotion");
    }
    if !promotion.allowed {
        return Err("external authority refused promotion");
    }
    if promotion.evaluation != evaluation.digest() {
        return Err("promotion does not bind the exact evaluation receipt");
    }
    if promotion.expires_at_epoch < now_epoch {
        return Err("promotion authorization is expired");
    }
    Ok(DeploymentReceipt {
        candidate: evaluation.candidate,
        evaluation: evaluation.digest(),
        authority: promotion.authority,
        scope: promotion.scope,
        expires_at_epoch: promotion.expires_at_epoch,
    })
}

fn assert_oracle_is_not_authority() {
    // These marker types deliberately have no conversion path from evaluation.
    struct EconomicInstrument(Digest);
    struct PhysicalCapacity(u64);

    let instrument = EconomicInstrument(Digest("external-authority-required"));
    let capacity = PhysicalCapacity(0);

    // The values exist only to make the semantic boundary explicit in the
    // fixture. Neither can be constructed from EvaluationReceipt.
    assert_eq!(instrument.0, Digest("external-authority-required"));
    assert_eq!(capacity.0, 0);
}

fn main() {
    let evaluator_profile = EvaluationProfile {
        evaluator: Digest("evaluator-v1"),
        profile: Digest("evaluation-profile-v1"),
        holdout: Digest("holdout-v1"),
        evaluator_code_lineage: Digest("evaluator-code-v1"),
        evaluator_data_lineage: Digest("evaluator-data-v1"),
        authority_lineage: Digest("authority-evaluator-v1"),
        valid_from_epoch: 100,
        valid_until_epoch: 200,
    };

    let mut evaluator = Evaluator::new(evaluator_profile, Digest("holdout-v1"), 42);

    let candidate = CandidateRevision {
        candidate: Digest("candidate-v1"),
        parent: None,
        code_lineage: Digest("candidate-code-v1"),
        change_set: Digest("candidate-expected"),
    };

    let observation = Observation {
        id: Digest("observation-1"),
        epoch: 150,
        measured_value: 42,
    };

    let pass = evaluator
        .evaluate(candidate, observation)
        .expect("benign candidate should evaluate");
    assert_eq!(pass.verdict, Verdict::Pass);
    assert!(pass.matches_subject(evaluator_profile, candidate));

    // Candidate/evaluator identities remain distinct even when the result is PASS.
    assert_ne!(candidate.digest(), pass.evaluator);

    // Candidate tampering changes the candidate identity. The old receipt cannot
    // be rebound to the modified subject.
    let tampered_candidate = CandidateRevision {
        change_set: Digest("candidate-tampered"),
        ..candidate
    };
    assert!(!pass.matches_subject(evaluator_profile, tampered_candidate));

    // A different holdout is a different evaluation subject. Reusing the old
    // receipt would be stale/non-matching rather than a valid reinterpretation.
    let changed_holdout = EvaluationProfile {
        holdout: Digest("holdout-replaced-after-observation"),
        ..evaluator_profile
    };
    assert!(!pass.matches_subject(changed_holdout, candidate));

    // Failed and indeterminate outcomes remain first-class.
    let degraded = CandidateRevision {
        candidate: Digest("candidate-degraded"),
        parent: Some(candidate.digest()),
        code_lineage: Digest("candidate-code-v2"),
        change_set: Digest("candidate-degraded"),
    };
    let degraded_observation = Observation {
        id: Digest("observation-2"),
        epoch: 151,
        measured_value: 42,
    };
    assert_eq!(degraded.parent, Some(candidate.digest()));
    let fail = evaluator
        .evaluate(degraded, degraded_observation)
        .expect("degraded candidate should evaluate");
    assert_eq!(fail.verdict, Verdict::Fail);
    assert_eq!(fail.candidate, degraded.digest());

    let uncertain_observation = Observation {
        id: Digest("observation-3"),
        epoch: 152,
        measured_value: 7,
    };
    let uncertain = evaluator
        .evaluate(candidate, uncertain_observation)
        .expect("mismatch should be indeterminate");
    assert_eq!(uncertain.verdict, Verdict::Indeterminate);

    let forced_pass = PromotionAuthorization {
        authority: Digest("external-authority"),
        evaluation: uncertain.digest(),
        scope: Digest("bounded-test-scope"),
        expires_at_epoch: 180,
        allowed: true,
    };
    assert_eq!(
        issue_deployment_receipt(uncertain, forced_pass, 160),
        Err("only an explicit Pass may be considered for promotion")
    );

    // A PASS is still not enough: an independent authority can refuse promotion.
    let refused = PromotionAuthorization {
        authority: Digest("external-authority"),
        evaluation: pass.digest(),
        scope: Digest("bounded-test-scope"),
        expires_at_epoch: 180,
        allowed: false,
    };
    assert_eq!(
        issue_deployment_receipt(pass, refused, 160),
        Err("external authority refused promotion")
    );

    // Only an externally supplied, scoped authorization can bridge evaluation
    // evidence into a deployment receipt.
    let admitted = PromotionAuthorization {
        authority: Digest("external-authority"),
        evaluation: pass.digest(),
        scope: Digest("bounded-test-scope"),
        expires_at_epoch: 180,
        allowed: true,
    };
    let deployment = issue_deployment_receipt(pass, admitted, 160)
        .expect("external authority should be able to authorize");
    assert_eq!(deployment.candidate, pass.candidate);
    assert_eq!(deployment.evaluation, pass.digest());
    assert_eq!(deployment.authority, Digest("external-authority"));
    assert_eq!(deployment.scope, Digest("bounded-test-scope"));
    assert!(deployment.expires_at_epoch >= pass.issued_epoch);

    // Duplicate replay cannot become fresh independent evidence.
    let duplicate = evaluator.admit_observation(observation);
    assert_eq!(duplicate, ObservationAdmission::Duplicate);

    // Stale evaluator/evidence is rejected at evaluation time.
    let stale_observation = Observation {
        id: Digest("observation-stale"),
        epoch: 201,
        measured_value: 42,
    };
    assert_eq!(
        evaluator.evaluate(candidate, stale_observation),
        Err("stale evaluator profile")
    );

    // Independence is multi-dimensional evidence, not a magic trust scalar.
    let independence = IndependenceRecord {
        candidate_code_lineage: candidate.code_lineage,
        evaluator_code_lineage: evaluator_profile.evaluator_code_lineage,
        candidate_data_lineage: Digest("shared-dataset-v1"),
        evaluator_data_lineage: Digest("shared-dataset-v1"),
        candidate_authority_lineage: Digest("candidate-owner-v1"),
        evaluator_authority_lineage: evaluator_profile.authority_lineage,
    };
    let findings = independence.findings();
    assert_eq!(findings[0], None);
    assert_eq!(findings[1], Some(IndependenceFinding::SharedDataLineage));
    assert_eq!(findings[2], None);

    assert_oracle_is_not_authority();

    println!("SYM-CIV-003 PASS: independent-evaluation boundary smoke controls hold.");
    println!("Claim ceiling: local semantic/type-flow control only; not evaluator correctness or deployment safety.");
}

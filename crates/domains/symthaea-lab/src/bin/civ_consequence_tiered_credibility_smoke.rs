//! SYM-CIV-005: consequence-tiered credibility and uncertainty smoke.
//!
//! Dependency-free research control.
//!
//! Core non-equivalences:
//! PASS != sufficient evidence
//! fit != validity
//! validation != verification
//! uncertainty estimate != certainty
//! in-domain evidence != universal truth
//! low-consequence qualification != high-consequence qualification
//! evidence coverage != authorization
//!
//! Claim ceiling: local evidence-admission/type-flow invariants only.

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Digest(&'static str);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ConsequenceTier {
    Low,
    Significant,
    Critical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EvidenceProfile {
    candidate: Digest,
    evaluator: Digest,
    profile: Digest,
    verification_ok: bool,
    validation_ok: bool,
    in_domain: bool,
    uncertainty_basis_points: u32,
    scenario_coverage_percent: u8,
    hard_invariants_hold: bool,
    fit_score: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TierRequirements {
    max_uncertainty_basis_points: u32,
    min_scenario_coverage_percent: u8,
}

impl ConsequenceTier {
    fn requirements(self) -> TierRequirements {
        match self {
            ConsequenceTier::Low => TierRequirements {
                max_uncertainty_basis_points: 5000,
                min_scenario_coverage_percent: 40,
            },
            ConsequenceTier::Significant => TierRequirements {
                max_uncertainty_basis_points: 2000,
                min_scenario_coverage_percent: 80,
            },
            ConsequenceTier::Critical => TierRequirements {
                max_uncertainty_basis_points: 500,
                min_scenario_coverage_percent: 95,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdmissionFailure {
    VerificationFailed,
    ValidationFailed,
    OutOfDomain,
    UncertaintyTooHigh,
    ScenarioCoverageInsufficient,
    HardInvariantFailed,
    TierInsufficient,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QualificationReceipt {
    id: Digest,
    candidate: Digest,
    evaluator: Digest,
    profile: Digest,
    tier: ConsequenceTier,
    evidence: EvidenceProfile,
}

impl QualificationReceipt {
    fn admits(self, requested_tier: ConsequenceTier) -> Result<(), AdmissionFailure> {
        if self.tier < requested_tier {
            return Err(AdmissionFailure::TierInsufficient);
        }
        Ok(())
    }
}

fn evaluate_evidence(
    evidence: EvidenceProfile,
    tier: ConsequenceTier,
) -> Result<QualificationReceipt, AdmissionFailure> {
    let requirements = tier.requirements();

    // Hard invariants dominate all aggregate or fit metrics.
    if !evidence.hard_invariants_hold {
        return Err(AdmissionFailure::HardInvariantFailed);
    }
    if !evidence.verification_ok {
        return Err(AdmissionFailure::VerificationFailed);
    }
    if !evidence.validation_ok {
        return Err(AdmissionFailure::ValidationFailed);
    }
    if !evidence.in_domain {
        return Err(AdmissionFailure::OutOfDomain);
    }
    if evidence.uncertainty_basis_points > requirements.max_uncertainty_basis_points {
        return Err(AdmissionFailure::UncertaintyTooHigh);
    }
    if evidence.scenario_coverage_percent < requirements.min_scenario_coverage_percent {
        return Err(AdmissionFailure::ScenarioCoverageInsufficient);
    }

    Ok(QualificationReceipt {
        id: match tier {
            ConsequenceTier::Low => Digest("qualification-low-v1"),
            ConsequenceTier::Significant => Digest("qualification-significant-v1"),
            ConsequenceTier::Critical => Digest("qualification-critical-v1"),
        },
        candidate: evidence.candidate,
        evaluator: evidence.evaluator,
        profile: evidence.profile,
        tier,
        evidence,
    })
}

fn upgrade_without_requalification(
    prior: QualificationReceipt,
    requested_tier: ConsequenceTier,
) -> Result<(), AdmissionFailure> {
    prior.admits(requested_tier)
}

fn main() {
    // The requirements become strictly stronger as consequence rises.
    let low = ConsequenceTier::Low.requirements();
    let significant = ConsequenceTier::Significant.requirements();
    let critical = ConsequenceTier::Critical.requirements();

    assert!(low.max_uncertainty_basis_points > significant.max_uncertainty_basis_points);
    assert!(
        significant.max_uncertainty_basis_points > critical.max_uncertainty_basis_points
    );
    assert!(low.scenario_coverage_percent < significant.scenario_coverage_percent);
    assert!(significant.scenario_coverage_percent < critical.scenario_coverage_percent);

    let candidate = Digest("candidate-v1");
    let evaluator = Digest("evaluator-independent-v1");
    let profile = Digest("profile-v1");

    let strong_evidence = EvidenceProfile {
        candidate,
        evaluator,
        profile,
        verification_ok: true,
        validation_ok: true,
        in_domain: true,
        uncertainty_basis_points: 250,
        scenario_coverage_percent: 98,
        hard_invariants_hold: true,
        fit_score: 990,
    };

    let critical = evaluate_evidence(strong_evidence, ConsequenceTier::Critical)
        .expect("strong evidence should qualify for critical consequence");
    assert_eq!(critical.tier, ConsequenceTier::Critical);
    assert_eq!(critical.candidate, candidate);
    assert_eq!(critical.evidence.fit_score, 990);

    // Excellent fit does not compensate for unbounded uncertainty.
    let uncertain = EvidenceProfile {
        uncertainty_basis_points: 6000,
        scenario_coverage_percent: 99,
        ..strong_evidence
    };
    assert_eq!(
        evaluate_evidence(uncertain, ConsequenceTier::Critical),
        Err(AdmissionFailure::UncertaintyTooHigh)
    );

    // Excellent fit outside the validated domain cannot be promoted.
    let extrapolated = EvidenceProfile {
        in_domain: false,
        uncertainty_basis_points: 100,
        fit_score: 1000,
        ..strong_evidence
    };
    assert_eq!(
        evaluate_evidence(extrapolated, ConsequenceTier::Critical),
        Err(AdmissionFailure::OutOfDomain)
    );

    // Sparse scenario coverage cannot be laundered by a good fit.
    let sparse = EvidenceProfile {
        scenario_coverage_percent: 60,
        fit_score: 1000,
        ..strong_evidence
    };
    assert_eq!(
        evaluate_evidence(sparse, ConsequenceTier::Critical),
        Err(AdmissionFailure::ScenarioCoverageInsufficient)
    );

    // Aggregate performance cannot erase a hard invariant.
    let rights_failure = EvidenceProfile {
        hard_invariants_hold: false,
        fit_score: 1000,
        uncertainty_basis_points: 100,
        scenario_coverage_percent: 100,
        ..strong_evidence
    };
    assert_eq!(
        evaluate_evidence(rights_failure, ConsequenceTier::Critical),
        Err(AdmissionFailure::HardInvariantFailed)
    );

    // Verification and validation are independently required.
    let verification_failure = EvidenceProfile {
        verification_ok: false,
        ..strong_evidence
    };
    assert_eq!(
        evaluate_evidence(verification_failure, ConsequenceTier::Critical),
        Err(AdmissionFailure::VerificationFailed)
    );

    let validation_failure = EvidenceProfile {
        validation_ok: false,
        ..strong_evidence
    };
    assert_eq!(
        evaluate_evidence(validation_failure, ConsequenceTier::Critical),
        Err(AdmissionFailure::ValidationFailed)
    );

    // A critical-tier evidence result cannot be synthesized from low-tier
    // requirements by simply declaring a more consequential use.
    let low_tier_evidence = EvidenceProfile {
        uncertainty_basis_points: 3000,
        scenario_coverage_percent: 70,
        ..strong_evidence
    };
    let low_qualification = evaluate_evidence(low_tier_evidence, ConsequenceTier::Low)
        .expect("evidence should satisfy low-tier requirements");
    assert_eq!(low_qualification.tier, ConsequenceTier::Low);

    assert_eq!(
        low_qualification.admits(ConsequenceTier::Critical),
        Err(AdmissionFailure::TierInsufficient)
    );
    assert_eq!(
        upgrade_without_requalification(low_qualification, ConsequenceTier::Critical),
        Err(AdmissionFailure::TierInsufficient)
    );

    // Exact evaluator/profile identities remain part of the qualification record.
    assert_eq!(critical.evaluator, evaluator);
    assert_eq!(critical.profile, profile);

    // The qualification receipt is not itself authorization.
    // A consequential admission system must still cross the independent
    // authority boundary established by CIV-CORE-003.
    assert_ne!(critical.id, Digest("authorization-v1"));

    println!("SYM-CIV-005 PASS: consequence-tiered credibility gates hold.");
    println!("Claim ceiling: evidence-admission control only; not proof of model truth or complete UQ.");
}

//! SWA-020: governance-state binding for a Sol Atlas intervention review.
//!
//! Mycelix Commons already contains a Housing Governance DNA with a Resolution
//! entry and a Housing Maintenance DNA. This fixture models the *boundary*
//! between a Sol Atlas review and that existing governance lifecycle.
//!
//! Important: a Resolution record is not treated as physical execution, and a
//! Sol Atlas projection never sets "passed" on behalf of governance. Adoption
//! must be an independently observed governance fact.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GovernanceState {
    ReviewRequested,
    ResolutionObserved,
    Adopted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GovernanceReference {
    resolution_id: &'static str,
    revision: &'static str,
    author_ref: &'static str,
    source_projection_id: &'static str,
    passed: bool,
    quorum_met: bool,
    effective_date_declared: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthorizationBoundary {
    NotAuthorized,
    ExternallyAuthorized,
}

fn state(reference: &GovernanceReference) -> GovernanceState {
    if reference.passed && reference.quorum_met && reference.effective_date_declared {
        GovernanceState::Adopted
    } else {
        GovernanceState::ResolutionObserved
    }
}

fn authorization_boundary(
    governance_state: GovernanceState,
    external_authority_reference: Option<&'static str>,
) -> AuthorizationBoundary {
    match (governance_state, external_authority_reference) {
        (GovernanceState::Adopted, Some(reference))
            if !reference.trim().is_empty() =>
        {
            AuthorizationBoundary::ExternallyAuthorized
        }
        _ => AuthorizationBoundary::NotAuthorized,
    }
}

fn main() {
    let review = GovernanceReference {
        resolution_id: "housing-resolution:sol-atlas-001",
        revision: "1",
        author_ref: "agent:governance-proposer",
        source_projection_id: "sol-atlas:projection:001",
        passed: false,
        quorum_met: false,
        effective_date_declared: false,
    };

    assert_eq!(state(&review), GovernanceState::ResolutionObserved);
    assert_eq!(
        authorization_boundary(state(&review), None),
        AuthorizationBoundary::NotAuthorized
    );

    // A later independently observed governance state can establish adoption,
    // but the adapter still requires an explicit authority reference.
    let adopted = GovernanceReference {
        passed: true,
        quorum_met: true,
        effective_date_declared: true,
        ..review
    };

    assert_eq!(state(&adopted), GovernanceState::Adopted);
    assert_eq!(
        authorization_boundary(
            state(&adopted),
            Some("housing-governance:resolution:action-hash")
        ),
        AuthorizationBoundary::ExternallyAuthorized
    );

    // Adoption still does not imply physical actuation.
    assert_ne!(state(&adopted), GovernanceState::ReviewRequested);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn review() -> GovernanceReference {
        GovernanceReference {
            resolution_id: "housing-resolution:sol-atlas-001",
            revision: "1",
            author_ref: "agent:governance-proposer",
            source_projection_id: "sol-atlas:projection:001",
            passed: false,
            quorum_met: false,
            effective_date_declared: false,
        }
    }

    #[test]
    fn review_request_is_not_adoption() {
        assert_eq!(state(&review()), GovernanceState::ResolutionObserved);
        assert_eq!(
            authorization_boundary(state(&review()), None),
            AuthorizationBoundary::NotAuthorized
        );
    }

    #[test]
    fn adoption_requires_all_declared_governance_conditions() {
        let mut candidate = review();
        candidate.passed = true;
        candidate.quorum_met = true;
        assert_eq!(state(&candidate), GovernanceState::ResolutionObserved);

        candidate.effective_date_declared = true;
        assert_eq!(state(&candidate), GovernanceState::Adopted);
    }

    #[test]
    fn adoption_without_authority_reference_is_not_authorization() {
        let adopted = GovernanceReference {
            passed: true,
            quorum_met: true,
            effective_date_declared: true,
            ..review()
        };
        assert_eq!(
            authorization_boundary(state(&adopted), None),
            AuthorizationBoundary::NotAuthorized
        );
    }

    #[test]
    fn empty_authority_reference_is_not_authorization() {
        let adopted = GovernanceReference {
            passed: true,
            quorum_met: true,
            effective_date_declared: true,
            ..review()
        };
        assert_eq!(
            authorization_boundary(state(&adopted), Some("   ")),
            AuthorizationBoundary::NotAuthorized
        );
    }

    #[test]
    fn external_authority_is_separate_from_source_projection() {
        let adopted = GovernanceReference {
            passed: true,
            quorum_met: true,
            effective_date_declared: true,
            ..review()
        };
        assert_ne!(adopted.source_projection_id, adopted.resolution_id);
        assert_ne!(adopted.author_ref, adopted.source_projection_id);
    }

    #[test]
    fn governance_adoption_does_not_model_actuation() {
        let adopted = GovernanceReference {
            passed: true,
            quorum_met: true,
            effective_date_declared: true,
            ..review()
        };
        assert_eq!(state(&adopted), GovernanceState::Adopted);
        // There is deliberately no actuator, command, device, or execution
        // field in this fixture.
    }

    #[test]
    fn revision_is_explicit() {
        let r = review();
        assert_eq!(r.revision, "1");
    }

    #[test]
    fn proposer_identity_is_preserved() {
        let r = review();
        assert_eq!(r.author_ref, "agent:governance-proposer");
    }
}

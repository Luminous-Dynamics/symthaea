//! Small finite-state model for workflow trust invariants.
//!
//! Deliberately independent from the JSON event parser in first_aid_trajectory.rs.
//! This model exhaustively explores the reachable abstract state graph. Passing it
//! establishes only that these modeled invariants hold in this model, not clinical safety.

#[cfg(test)]
mod tests {
    use std::collections::{HashSet, VecDeque};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Bundle {
        A,
        B,
    }

    impl Bundle {
        fn bit(self) -> u8 {
            match self {
                Self::A => 0b01,
                Self::B => 0b10,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct State {
        session_active: bool,
        loaded: Option<Bundle>,
        verified: Option<Bundle>,
        revoked_mask: u8,
        uncertainty_pending: bool,
        uncertainty_preserved: bool,
    }

    impl State {
        const INITIAL: Self = Self {
            session_active: false,
            loaded: None,
            verified: None,
            revoked_mask: 0,
            uncertainty_pending: false,
            uncertainty_preserved: false,
        };

        fn is_revoked(self, bundle: Bundle) -> bool {
            self.revoked_mask & bundle.bit() != 0
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Violation {
        InvalidBundleVerification,
        RenderWithoutActiveSession,
        RenderAfterRevocation,
        RenderWithoutVerifiedBundle,
        UncertaintyLostAtHandoff,
    }

    #[derive(Debug, Clone, Copy)]
    enum Action {
        StartSession,
        LoadA,
        LoadB,
        VerifyAValid,
        VerifyABadDigest,
        VerifyAExpired,
        VerifyBValid,
        RevokeA,
        RevokeB,
        Interrupt,
        Resume,
        DetectUncertainty,
        PreserveUncertainty,
        Handoff,
        RenderA,
        RenderB,
    }

    const ACTIONS: [Action; 16] = [
        Action::StartSession,
        Action::LoadA,
        Action::LoadB,
        Action::VerifyAValid,
        Action::VerifyABadDigest,
        Action::VerifyAExpired,
        Action::VerifyBValid,
        Action::RevokeA,
        Action::RevokeB,
        Action::Interrupt,
        Action::Resume,
        Action::DetectUncertainty,
        Action::PreserveUncertainty,
        Action::Handoff,
        Action::RenderA,
        Action::RenderB,
    ];

    #[derive(Debug)]
    struct Transition {
        next: State,
        violations: Vec<Violation>,
    }

    fn transition(state: State, action: Action) -> Transition {
        let mut next = state;
        let mut violations = Vec::new();

        match action {
            Action::StartSession => {
                next.session_active = true;
                next.loaded = None;
                next.verified = None;
                next.uncertainty_pending = false;
                next.uncertainty_preserved = false;
            }
            Action::LoadA | Action::LoadB => {
                next.loaded = Some(match action {
                    Action::LoadA => Bundle::A,
                    Action::LoadB => Bundle::B,
                    _ => unreachable!(),
                });
                next.verified = None;
            }
            Action::VerifyAValid
            | Action::VerifyABadDigest
            | Action::VerifyAExpired
            | Action::VerifyBValid => {
                let (bundle, valid_state, digest_verified) = match action {
                    Action::VerifyAValid => (Bundle::A, true, true),
                    Action::VerifyABadDigest => (Bundle::A, true, false),
                    Action::VerifyAExpired => (Bundle::A, false, true),
                    Action::VerifyBValid => (Bundle::B, true, true),
                    _ => unreachable!(),
                };
                if state.loaded == Some(bundle)
                    && valid_state
                    && digest_verified
                    && !state.is_revoked(bundle)
                {
                    next.verified = Some(bundle);
                } else {
                    next.verified = None;
                    violations.push(Violation::InvalidBundleVerification);
                }
            }
            Action::RevokeA | Action::RevokeB => {
                let bundle = if matches!(action, Action::RevokeA) {
                    Bundle::A
                } else {
                    Bundle::B
                };
                next.revoked_mask |= bundle.bit();
                if next.verified == Some(bundle) {
                    next.verified = None;
                }
            }
            Action::Interrupt => {
                next.session_active = false;
                next.verified = None;
            }
            Action::Resume => {
                // Resume must never carry verification across the boundary,
                // including malformed traces that resume without an interrupt.
                next.session_active = true;
                next.verified = None;
            }
            Action::DetectUncertainty => {
                next.uncertainty_pending = true;
                next.uncertainty_preserved = false;
            }
            Action::PreserveUncertainty => {
                if state.uncertainty_pending {
                    next.uncertainty_preserved = true;
                }
            }
            Action::Handoff => {
                if state.uncertainty_pending && !state.uncertainty_preserved {
                    violations.push(Violation::UncertaintyLostAtHandoff);
                }
                next.uncertainty_pending = false;
                next.uncertainty_preserved = false;
            }
            Action::RenderA | Action::RenderB => {
                let bundle = if matches!(action, Action::RenderA) {
                    Bundle::A
                } else {
                    Bundle::B
                };
                if !state.session_active {
                    violations.push(Violation::RenderWithoutActiveSession);
                }
                if state.is_revoked(bundle) {
                    violations.push(Violation::RenderAfterRevocation);
                } else if state.verified != Some(bundle) {
                    violations.push(Violation::RenderWithoutVerifiedBundle);
                }
            }
        }

        Transition { next, violations }
    }

    fn reachable_states() -> HashSet<State> {
        let mut seen = HashSet::from([State::INITIAL]);
        let mut queue = VecDeque::from([State::INITIAL]);
        while let Some(state) = queue.pop_front() {
            for action in ACTIONS {
                let next = transition(state, action).next;
                if seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }
        seen
    }

    #[test]
    fn exhaustive_reachable_state_graph_preserves_trust_invariants() {
        let states = reachable_states();
        let mut transition_count = 0usize;

        for state in &states {
            for action in ACTIONS {
                transition_count += 1;
                let outcome = transition(*state, action);
                assert_eq!(
                    outcome.next.revoked_mask & state.revoked_mask,
                    state.revoked_mask,
                    "revocation must be monotonic for {state:?} via {action:?}"
                );

                match action {
                    Action::StartSession => {
                        assert!(outcome.next.session_active);
                        assert_eq!(outcome.next.loaded, None);
                        assert_eq!(outcome.next.verified, None);
                        assert!(!outcome.next.uncertainty_pending);
                        assert!(!outcome.next.uncertainty_preserved);
                    }
                    Action::Interrupt => {
                        assert!(!outcome.next.session_active);
                        assert_eq!(outcome.next.verified, None);
                    }
                    Action::Resume => {
                        assert!(outcome.next.session_active);
                        assert_eq!(outcome.next.verified, None);
                    }
                    Action::Handoff => {
                        assert_eq!(
                            outcome
                                .violations
                                .contains(&Violation::UncertaintyLostAtHandoff),
                            state.uncertainty_pending && !state.uncertainty_preserved
                        );
                    }
                    Action::RenderA | Action::RenderB if outcome.violations.is_empty() => {
                        let rendered = if matches!(action, Action::RenderA) {
                            Bundle::A
                        } else {
                            Bundle::B
                        };
                        assert!(state.session_active);
                        assert_eq!(state.verified, Some(rendered));
                        assert!(!state.is_revoked(rendered));
                    }
                    Action::VerifyAValid | Action::VerifyBValid
                        if outcome.next.verified.is_some() =>
                    {
                        let verified = outcome.next.verified.expect("checked above");
                        assert_eq!(state.loaded, Some(verified));
                        assert!(!state.is_revoked(verified));
                    }
                    _ => {}
                }
            }
        }

        assert!(states.len() >= 100, "state exploration unexpectedly shallow");
        assert!(transition_count >= 1_600, "transition exploration unexpectedly shallow");
    }

    #[test]
    fn resuming_without_an_interrupt_cannot_retain_a_verified_bundle() {
        let state = State {
            session_active: true,
            loaded: Some(Bundle::A),
            verified: Some(Bundle::A),
            revoked_mask: 0,
            uncertainty_pending: false,
            uncertainty_preserved: false,
        };
        let resumed = transition(state, Action::Resume);
        assert_eq!(resumed.next.verified, None);
        let rendered = transition(resumed.next, Action::RenderA);
        assert!(rendered
            .violations
            .contains(&Violation::RenderWithoutVerifiedBundle));
    }

    #[test]
    fn revocation_and_uncertainty_obligations_survive_unrelated_transitions() {
        let state = State {
            session_active: true,
            loaded: Some(Bundle::B),
            verified: Some(Bundle::B),
            revoked_mask: Bundle::A.bit(),
            uncertainty_pending: true,
            uncertainty_preserved: false,
        };
        let started = transition(state, Action::StartSession);
        assert_eq!(started.next.revoked_mask, Bundle::A.bit());
        assert!(!started.next.uncertainty_pending);
        let revoked_render = transition(started.next, Action::RenderA);
        assert!(revoked_render
            .violations
            .contains(&Violation::RenderAfterRevocation));
        let handoff = transition(state, Action::Handoff);
        assert!(handoff
            .violations
            .contains(&Violation::UncertaintyLostAtHandoff));
    }
}

//! Small finite-state model for workflow trust invariants.
//!
//! Deliberately independent from the JSON event parser in first_aid_trajectory.rs.
//! This model exhaustively explores the reachable abstract state graph. Passing it
//! establishes only that these modeled invariants hold in this model, not clinical safety.

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet, VecDeque, hash_map::Entry};

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
        connectivity: Connectivity,
        loaded: Option<Bundle>,
        verified: Option<Bundle>,
        revoked_mask: u8,
        uncertainty_pending: bool,
        uncertainty_preserved: bool,
    }

    impl State {
        const INITIAL: Self = Self {
            session_active: false,
            connectivity: Connectivity::Online,
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
        UncertaintyLostAtSessionBoundary,
        UnknownNormalized,
        StaleResourceClaimedConfirmed,
        OfflineCoreNetworkDependency,
        IntermittentCoreNetworkDependency,
        SensitiveDataEgressWithoutConsent,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Connectivity {
        Online,
        Offline,
        Intermittent,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ResourceFreshness {
        Fresh,
        Stale,
        Unknown,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ResourcePresentation {
        Confirmed,
        Unconfirmed,
    }

    fn network_policy_violation(
        connectivity: Connectivity,
        core_network_required: bool,
    ) -> Option<Violation> {
        if !core_network_required {
            return None;
        }
        match connectivity {
            Connectivity::Online => None,
            Connectivity::Offline => Some(Violation::OfflineCoreNetworkDependency),
            Connectivity::Intermittent => Some(Violation::IntermittentCoreNetworkDependency),
        }
    }

    fn resource_policy_violation(
        freshness: ResourceFreshness,
        presentation: ResourcePresentation,
    ) -> Option<Violation> {
        match (freshness, presentation) {
            (
                ResourceFreshness::Stale | ResourceFreshness::Unknown,
                ResourcePresentation::Confirmed,
            ) => Some(Violation::StaleResourceClaimedConfirmed),
            _ => None,
        }
    }

    fn egress_policy_violation(sensitive_payload: bool, consent_scoped: bool) -> Option<Violation> {
        if sensitive_payload && !consent_scoped {
            Some(Violation::SensitiveDataEgressWithoutConsent)
        } else {
            None
        }
    }

    fn unknown_normalization_violation(normalized_unknown: bool) -> Option<Violation> {
        normalized_unknown.then_some(Violation::UnknownNormalized)
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
        ConnectivityOnline,
        ConnectivityOffline,
        ConnectivityIntermittent,
        RequireCoreNetwork,
        PresentFreshConfirmed,
        PresentFreshUnconfirmed,
        PresentStaleConfirmed,
        PresentStaleUnconfirmed,
        PresentUnknownConfirmed,
        PresentUnknownUnconfirmed,
        EgressNonSensitiveWithoutConsent,
        EgressNonSensitiveWithConsent,
        EgressSensitiveWithoutConsent,
        EgressSensitiveWithConsent,
        NormalizeUnknown,
    }

    const ACTIONS: [Action; 31] = [
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
        Action::ConnectivityOnline,
        Action::ConnectivityOffline,
        Action::ConnectivityIntermittent,
        Action::RequireCoreNetwork,
        Action::PresentFreshConfirmed,
        Action::PresentFreshUnconfirmed,
        Action::PresentStaleConfirmed,
        Action::PresentStaleUnconfirmed,
        Action::PresentUnknownConfirmed,
        Action::PresentUnknownUnconfirmed,
        Action::EgressNonSensitiveWithoutConsent,
        Action::EgressNonSensitiveWithConsent,
        Action::EgressSensitiveWithoutConsent,
        Action::EgressSensitiveWithConsent,
        Action::NormalizeUnknown,
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
            Action::ConnectivityOnline => next.connectivity = Connectivity::Online,
            Action::ConnectivityOffline => next.connectivity = Connectivity::Offline,
            Action::ConnectivityIntermittent => next.connectivity = Connectivity::Intermittent,
            Action::RequireCoreNetwork => {
                if let Some(violation) = network_policy_violation(state.connectivity, true) {
                    violations.push(violation);
                }
            }
            Action::StartSession => {
                next.session_active = true;
                next.loaded = None;
                next.verified = None;
                if state.uncertainty_pending && !state.uncertainty_preserved {
                    violations.push(Violation::UncertaintyLostAtSessionBoundary);
                }
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
                if state.session_active
                    && state.loaded == Some(bundle)
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
            Action::PresentFreshConfirmed
            | Action::PresentFreshUnconfirmed
            | Action::PresentStaleConfirmed
            | Action::PresentStaleUnconfirmed
            | Action::PresentUnknownConfirmed
            | Action::PresentUnknownUnconfirmed => {
                let (freshness, presentation) = match action {
                    Action::PresentFreshConfirmed => {
                        (ResourceFreshness::Fresh, ResourcePresentation::Confirmed)
                    }
                    Action::PresentFreshUnconfirmed => {
                        (ResourceFreshness::Fresh, ResourcePresentation::Unconfirmed)
                    }
                    Action::PresentStaleConfirmed => {
                        (ResourceFreshness::Stale, ResourcePresentation::Confirmed)
                    }
                    Action::PresentStaleUnconfirmed => {
                        (ResourceFreshness::Stale, ResourcePresentation::Unconfirmed)
                    }
                    Action::PresentUnknownConfirmed => {
                        (ResourceFreshness::Unknown, ResourcePresentation::Confirmed)
                    }
                    Action::PresentUnknownUnconfirmed => {
                        (ResourceFreshness::Unknown, ResourcePresentation::Unconfirmed)
                    }
                    _ => unreachable!(),
                };
                if let Some(violation) = resource_policy_violation(freshness, presentation) {
                    violations.push(violation);
                }
            }
            Action::EgressNonSensitiveWithoutConsent
            | Action::EgressNonSensitiveWithConsent
            | Action::EgressSensitiveWithoutConsent
            | Action::EgressSensitiveWithConsent => {
                let (sensitive_payload, consent_scoped) = match action {
                    Action::EgressNonSensitiveWithoutConsent => (false, false),
                    Action::EgressNonSensitiveWithConsent => (false, true),
                    Action::EgressSensitiveWithoutConsent => (true, false),
                    Action::EgressSensitiveWithConsent => (true, true),
                    _ => unreachable!(),
                };
                if let Some(violation) =
                    egress_policy_violation(sensitive_payload, consent_scoped)
                {
                    violations.push(violation);
                }
            }
            Action::NormalizeUnknown => {
                if let Some(violation) = unknown_normalization_violation(true) {
                    violations.push(violation);
                }
            }
        }

        Transition { next, violations }
    }

    fn assert_state_invariants(state: State) {
        assert_eq!(
            state.revoked_mask & !0b11,
            0,
            "revocation mask may contain only declared bundle identities: {state:?}"
        );
        assert!(
            !state.uncertainty_preserved || state.uncertainty_pending,
            "preserved uncertainty must remain pending until handoff: {state:?}"
        );
        if let Some(verified) = state.verified {
            assert!(state.session_active, "verified bundle outside active session: {state:?}");
            assert_eq!(
                state.loaded,
                Some(verified),
                "verified identity must equal the currently loaded identity: {state:?}"
            );
            assert!(
                !state.is_revoked(verified),
                "revoked bundle cannot remain verified: {state:?}"
            );
        }
    }

    fn shortest_paths_to_reachable_states() -> HashMap<State, Vec<Action>> {
        let mut paths = HashMap::from([(State::INITIAL, Vec::new())]);
        let mut queue = VecDeque::from([State::INITIAL]);

        while let Some(state) = queue.pop_front() {
            let prefix = paths
                .get(&state)
                .expect("queued state has a shortest path")
                .clone();
            for action in ACTIONS {
                let next = transition(state, action).next;
                if let Entry::Vacant(entry) = paths.entry(next) {
                    let mut path = prefix.clone();
                    path.push(action);
                    entry.insert(path);
                    queue.push_back(next);
                }
            }
        }
        paths
    }

    fn action_event_json(action: Action, sequence: u32) -> serde_json::Value {
        use serde_json::json;

        match action {
            Action::StartSession => json!({"sequence": sequence, "kind": "session_started"}),
            Action::LoadA => json!({
                "sequence": sequence, "kind": "bundle_loaded",
                "bundle_id": "bundle-a", "bundle_state": "valid"
            }),
            Action::LoadB => json!({
                "sequence": sequence, "kind": "bundle_loaded",
                "bundle_id": "bundle-b", "bundle_state": "valid"
            }),
            Action::VerifyAValid => json!({
                "sequence": sequence, "kind": "bundle_verified",
                "bundle_id": "bundle-a", "bundle_state": "valid", "digest_verified": true
            }),
            Action::VerifyABadDigest => json!({
                "sequence": sequence, "kind": "bundle_verified",
                "bundle_id": "bundle-a", "bundle_state": "valid", "digest_verified": false
            }),
            Action::VerifyAExpired => json!({
                "sequence": sequence, "kind": "bundle_verified",
                "bundle_id": "bundle-a", "bundle_state": "expired", "digest_verified": true
            }),
            Action::VerifyBValid => json!({
                "sequence": sequence, "kind": "bundle_verified",
                "bundle_id": "bundle-b", "bundle_state": "valid", "digest_verified": true
            }),
            Action::RevokeA => json!({
                "sequence": sequence, "kind": "bundle_revoked", "bundle_id": "bundle-a"
            }),
            Action::RevokeB => json!({
                "sequence": sequence, "kind": "bundle_revoked", "bundle_id": "bundle-b"
            }),
            Action::Interrupt => json!({"sequence": sequence, "kind": "session_interrupted"}),
            Action::Resume => json!({"sequence": sequence, "kind": "session_resumed"}),
            Action::DetectUncertainty => json!({"sequence": sequence, "kind": "uncertainty_detected"}),
            Action::PreserveUncertainty => json!({"sequence": sequence, "kind": "uncertainty_preserved"}),
            Action::Handoff => json!({"sequence": sequence, "kind": "handoff_emitted"}),
            Action::RenderA => json!({
                "sequence": sequence, "kind": "protocol_rendered", "bundle_id": "bundle-a"
            }),
            Action::RenderB => json!({
                "sequence": sequence, "kind": "protocol_rendered", "bundle_id": "bundle-b"
            }),
            Action::ConnectivityOnline => json!({
                "sequence": sequence, "kind": "connectivity_changed", "connectivity": "online"
            }),
            Action::ConnectivityOffline => json!({
                "sequence": sequence, "kind": "connectivity_changed", "connectivity": "offline"
            }),
            Action::ConnectivityIntermittent => json!({
                "sequence": sequence, "kind": "connectivity_changed", "connectivity": "intermittent"
            }),
            Action::RequireCoreNetwork => json!({
                "sequence": sequence, "kind": "network_required_for_core"
            }),
            Action::PresentFreshConfirmed => json!({
                "sequence": sequence, "kind": "resource_presented",
                "resource_state": "fresh", "presented_as": "confirmed"
            }),
            Action::PresentFreshUnconfirmed => json!({
                "sequence": sequence, "kind": "resource_presented",
                "resource_state": "fresh", "presented_as": "unconfirmed"
            }),
            Action::PresentStaleConfirmed => json!({
                "sequence": sequence, "kind": "resource_presented",
                "resource_state": "stale", "presented_as": "confirmed"
            }),
            Action::PresentStaleUnconfirmed => json!({
                "sequence": sequence, "kind": "resource_presented",
                "resource_state": "stale", "presented_as": "unconfirmed"
            }),
            Action::PresentUnknownConfirmed => json!({
                "sequence": sequence, "kind": "resource_presented",
                "resource_state": "unknown", "presented_as": "confirmed"
            }),
            Action::PresentUnknownUnconfirmed => json!({
                "sequence": sequence, "kind": "resource_presented",
                "resource_state": "unknown", "presented_as": "unconfirmed"
            }),
            Action::EgressNonSensitiveWithoutConsent => json!({
                "sequence": sequence, "kind": "sensitive_data_egress",
                "sensitive_payload": false, "consent_scoped": false
            }),
            Action::EgressNonSensitiveWithConsent => json!({
                "sequence": sequence, "kind": "sensitive_data_egress",
                "sensitive_payload": false, "consent_scoped": true
            }),
            Action::EgressSensitiveWithoutConsent => json!({
                "sequence": sequence, "kind": "sensitive_data_egress",
                "sensitive_payload": true, "consent_scoped": false
            }),
            Action::EgressSensitiveWithConsent => json!({
                "sequence": sequence, "kind": "sensitive_data_egress",
                "sensitive_payload": true, "consent_scoped": true
            }),
            Action::NormalizeUnknown => json!({
                "sequence": sequence, "kind": "unknown_normalized"
            }),
        }
    }

    fn action_history_json(actions: &[Action]) -> serde_json::Value {
        use serde_json::json;

        let events: Vec<_> = actions
            .iter()
            .enumerate()
            .map(|(index, action)| action_event_json(*action, (index + 1) as u32))
            .collect();
        json!({
            "trace_id": "generated-finite-state-edge",
            "description": "generated abstract state/action path",
            "criticality": "critical",
            "connectivity": "online",
            "events": events,
            "expected_violation_codes": []
        })
    }

    fn policy_trace_json(
        trace_id: &str,
        connectivity: &str,
        event: Option<serde_json::Value>,
    ) -> serde_json::Value {
        use serde_json::json;

        json!({
            "trace_id": trace_id,
            "description": "generated orthogonal policy truth-table case",
            "criticality": "critical",
            "connectivity": connectivity,
            "events": event.into_iter().collect::<Vec<_>>(),
            "expected_violation_codes": []
        })
    }

    fn connectivity_label(connectivity: Connectivity) -> &'static str {
        match connectivity {
            Connectivity::Online => "online",
            Connectivity::Offline => "offline",
            Connectivity::Intermittent => "intermittent",
        }
    }

    fn violation_code(violation: Violation) -> &'static str {
        match violation {
            Violation::InvalidBundleVerification => "invalid_bundle_verification",
            Violation::RenderWithoutActiveSession => "render_without_active_session",
            Violation::RenderAfterRevocation => "render_after_revocation",
            Violation::RenderWithoutVerifiedBundle => "render_without_verified_bundle",
            Violation::UncertaintyLostAtHandoff => "uncertainty_lost_at_handoff",
            Violation::UncertaintyLostAtSessionBoundary => "uncertainty_lost_at_session_boundary",
            Violation::UnknownNormalized => "unknown_normalized",
            Violation::StaleResourceClaimedConfirmed => "stale_resource_claimed_confirmed",
            Violation::OfflineCoreNetworkDependency => "offline_core_network_dependency",
            Violation::IntermittentCoreNetworkDependency => "intermittent_core_network_dependency",
            Violation::SensitiveDataEgressWithoutConsent => "sensitive_data_egress_without_consent",
        }
    }

    fn model_violation_codes(actions: &[Action]) -> Vec<String> {
        let mut state = State::INITIAL;
        let mut codes = Vec::new();
        for action in actions {
            let outcome = transition(state, *action);
            codes.extend(
                outcome
                    .violations
                    .into_iter()
                    .map(|violation| violation_code(violation).to_owned()),
            );
            state = outcome.next;
        }
        codes
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
            assert_state_invariants(*state);
            for action in ACTIONS {
                transition_count += 1;
                let outcome = transition(*state, action);
                assert_state_invariants(outcome.next);
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
                        assert_eq!(outcome.next.uncertainty_pending, state.uncertainty_pending);
                        assert_eq!(
                            outcome
                                .violations
                                .contains(&Violation::UncertaintyLostAtSessionBoundary),
                            state.uncertainty_pending && !state.uncertainty_preserved
                        );
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
                    Action::RenderA | Action::RenderB => {
                        let rendered = if matches!(action, Action::RenderA) {
                            Bundle::A
                        } else {
                            Bundle::B
                        };
                        let mut expected = Vec::new();
                        if !state.session_active {
                            expected.push(Violation::RenderWithoutActiveSession);
                        }
                        if state.is_revoked(rendered) {
                            expected.push(Violation::RenderAfterRevocation);
                        } else if state.verified != Some(rendered) {
                            expected.push(Violation::RenderWithoutVerifiedBundle);
                        }
                        assert_eq!(
                            outcome.violations, expected,
                            "render authorization mismatch for {state:?} via {action:?}"
                        );
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
                        let allowed = state.session_active
                            && state.loaded == Some(bundle)
                            && valid_state
                            && digest_verified
                            && !state.is_revoked(bundle);
                        assert_eq!(
                            outcome.next.verified,
                            allowed.then_some(bundle),
                            "verification result mismatch for {state:?} via {action:?}"
                        );
                        assert_eq!(
                            outcome.violations.contains(&Violation::InvalidBundleVerification),
                            !allowed,
                            "verification violation mismatch for {state:?} via {action:?}"
                        );
                    }
                    Action::ConnectivityOnline => {
                        assert_eq!(outcome.next.connectivity, Connectivity::Online);
                    }
                    Action::ConnectivityOffline => {
                        assert_eq!(outcome.next.connectivity, Connectivity::Offline);
                    }
                    Action::ConnectivityIntermittent => {
                        assert_eq!(outcome.next.connectivity, Connectivity::Intermittent);
                    }
                    Action::RequireCoreNetwork => {
                        let expected = network_policy_violation(state.connectivity, true);
                        assert_eq!(outcome.violations.as_slice(), expected.as_slice());
                    }
                    Action::LoadA | Action::LoadB => {
                        assert_eq!(outcome.next.verified, None);
                    }
                    Action::RevokeA | Action::RevokeB => {
                        let revoked = if matches!(action, Action::RevokeA) {
                            Bundle::A
                        } else {
                            Bundle::B
                        };
                        assert!(outcome.next.is_revoked(revoked));
                        assert_ne!(
                            outcome.next.verified,
                            Some(revoked),
                            "revocation must invalidate matching verification"
                        );
                    }
                    _ => {}
                }
            }
        }

        assert_eq!(
            states.len(),
            252,
            "reachable abstract-state count changed; review and update the model baseline"
        );
        assert_eq!(
            transition_count, 7_812,
            "state/action transition count changed; review and update the model baseline"
        );
        println!(
            "abstract_states_explored={} state_action_transitions_checked={}",
            states.len(),
            transition_count
        );
    }

    #[test]
    fn every_reachable_state_action_edge_matches_production_trace_checker() {
        let paths = shortest_paths_to_reachable_states();
        assert_eq!(paths.len(), 252, "finite model's reachable state count changed");

        let mut compared_edges = 0usize;
        for (target_state, prefix) in &paths {
            let mut replayed_state = State::INITIAL;
            for action in prefix {
                replayed_state = transition(replayed_state, *action).next;
            }
            assert_eq!(
                replayed_state, *target_state,
                "shortest-path witness must reach its indexed state"
            );

            for action in ACTIONS {
                let mut edge_path = prefix.clone();
                edge_path.push(action);
                let generated_trace = action_history_json(&edge_path);
                let model_codes = model_violation_codes(&edge_path);
                let production_codes = super::super::first_aid_trajectory::inspect_trace_json(
                    &generated_trace,
                )
                .expect("generated abstract history must satisfy the event schema");

                assert_eq!(
                    production_codes,
                    model_codes,
                    "trace/model mismatch at state {target_state:?} via {action:?}; path={edge_path:?}"
                );
                compared_edges += 1;
            }
        }

        assert_eq!(
            compared_edges, 7_812,
            "every reachable state/action edge must be differentially checked"
        );
        println!("model_production_edge_comparisons={compared_edges}");
    }

    #[test]
    fn every_reachable_state_two_action_suffix_matches_production_trace_checker() {
        let paths = shortest_paths_to_reachable_states();
        assert_eq!(paths.len(), 252, "finite model's reachable state count changed");

        let mut compared_sequences = 0usize;
        for (target_state, prefix) in &paths {
            for first_action in ACTIONS {
                for second_action in ACTIONS {
                    let mut sequence = prefix.clone();
                    sequence.push(first_action);
                    sequence.push(second_action);

                    let generated_trace = action_history_json(&sequence);
                    let model_codes = model_violation_codes(&sequence);
                    let production_codes =
                        super::super::first_aid_trajectory::inspect_trace_json(&generated_trace)
                            .expect("generated two-action trace must satisfy the event schema");

                    assert_eq!(
                        production_codes, model_codes,
                        "two-action trace mismatch at state {target_state:?} via                          {first_action:?} then {second_action:?}; path={sequence:?}"
                    );
                    compared_sequences += 1;
                }
            }
        }

        assert_eq!(
            compared_sequences, 242_172,
            "all 252 states × 20 × 20 two-action suffixes must be compared"
        );
        println!("model_production_two_action_sequence_comparisons={compared_sequences}");
    }

    #[test]
    fn seeded_long_horizon_histories_match_production_trace_checker() {
        const SEED: u64 = 0x4c55_4d49_4e4f_5553;
        const TRIALS: usize = 4_096;
        const MAX_ACTIONS: usize = 64;

        fn next_random(state: &mut u64) -> u64 {
            // xorshift64*: deterministic, dependency-free, and reproducible from SEED.
            let mut value = *state;
            value ^= value >> 12;
            value ^= value << 25;
            value ^= value >> 27;
            *state = value;
            value.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        let mut random_state = SEED;
        let mut compared = 0usize;
        let mut action_histogram = [0usize; ACTIONS.len()];

        for trial in 0..TRIALS {
            let length = (next_random(&mut random_state) as usize % MAX_ACTIONS) + 1;
            let mut actions = Vec::with_capacity(length);
            for _ in 0..length {
                let index = next_random(&mut random_state) as usize % ACTIONS.len();
                action_histogram[index] += 1;
                actions.push(ACTIONS[index]);
            }

            let trace = action_history_json(&actions);
            let model_codes = model_violation_codes(&actions);
            let production_codes =
                super::super::first_aid_trajectory::inspect_trace_json(&trace)
                    .expect("seeded generated trace must satisfy the event schema");

            assert_eq!(
                production_codes,
                model_codes,
                "seeded model/production mismatch; seed=0x{SEED:016x}, trial={trial},                  length={length}, actions={actions:?}"
            );
            compared += 1;
        }

        assert_eq!(compared, TRIALS);
        assert!(
            action_histogram.iter().all(|count| *count > 0),
            "seeded traces must exercise every modeled action: {action_histogram:?}"
        );
        println!(
            "model_production_seeded_long_horizon_comparisons={compared}              seed=0x{SEED:016x} max_actions={MAX_ACTIONS}"
        );
    }

    #[test]
    fn resuming_without_an_interrupt_cannot_retain_a_verified_bundle() {
        let state = State {
            session_active: true,
            connectivity: Connectivity::Online,
            loaded: Some(Bundle::A),
            verified: Some(Bundle::A),
            revoked_mask: 0,
            uncertainty_pending: false,
            uncertainty_preserved: false,
        };
        let resumed = transition(state, Action::Resume);
        assert_eq!(resumed.next.verified, None);
        let rendered = transition(resumed.next, Action::RenderA);
        assert!(
            rendered
                .violations
                .contains(&Violation::RenderWithoutVerifiedBundle)
        );
    }

    #[test]
    fn revocation_and_uncertainty_obligations_survive_unrelated_transitions() {
        let state = State {
            session_active: true,
            connectivity: Connectivity::Online,
            loaded: Some(Bundle::B),
            verified: Some(Bundle::B),
            revoked_mask: Bundle::A.bit(),
            uncertainty_pending: true,
            uncertainty_preserved: false,
        };
        let started = transition(state, Action::StartSession);
        assert_eq!(started.next.revoked_mask, Bundle::A.bit());
        assert!(started.next.uncertainty_pending);
        assert!(
            started
                .violations
                .contains(&Violation::UncertaintyLostAtSessionBoundary)
        );
        let revoked_render = transition(started.next, Action::RenderA);
        assert!(
            revoked_render
                .violations
                .contains(&Violation::RenderAfterRevocation)
        );
        let handoff = transition(state, Action::Handoff);
        assert!(
            handoff
                .violations
                .contains(&Violation::UncertaintyLostAtHandoff)
        );
    }
    #[test]
    fn verification_without_active_session_is_rejected() {
        let state = State {
            session_active: false,
            connectivity: Connectivity::Online,
            loaded: Some(Bundle::A),
            verified: None,
            revoked_mask: 0,
            uncertainty_pending: false,
            uncertainty_preserved: false,
        };
        let result = transition(state, Action::VerifyAValid);
        assert_eq!(result.next.verified, None);
        assert!(
            result
                .violations
                .contains(&Violation::InvalidBundleVerification)
        );
    }

    #[test]
    fn exhaustive_orthogonal_policy_truth_tables_cover_non_lifecycle_invariants() {
        let mut network_cases = 0;
        for connectivity in [
            Connectivity::Online,
            Connectivity::Offline,
            Connectivity::Intermittent,
        ] {
            for core_network_required in [false, true] {
                network_cases += 1;
                let expected = match (connectivity, core_network_required) {
                    (Connectivity::Offline, true) => Some(Violation::OfflineCoreNetworkDependency),
                    (Connectivity::Intermittent, true) => {
                        Some(Violation::IntermittentCoreNetworkDependency)
                    }
                    _ => None,
                };
                assert_eq!(
                    network_policy_violation(connectivity, core_network_required),
                    expected,
                    "network mismatch: {connectivity:?}, required={core_network_required}"
                );
            }
        }
        assert_eq!(network_cases, 6);

        let mut resource_cases = 0;
        for freshness in [
            ResourceFreshness::Fresh,
            ResourceFreshness::Stale,
            ResourceFreshness::Unknown,
        ] {
            for presentation in [
                ResourcePresentation::Confirmed,
                ResourcePresentation::Unconfirmed,
            ] {
                resource_cases += 1;
                let expected = match (freshness, presentation) {
                    (
                        ResourceFreshness::Stale | ResourceFreshness::Unknown,
                        ResourcePresentation::Confirmed,
                    ) => Some(Violation::StaleResourceClaimedConfirmed),
                    _ => None,
                };
                assert_eq!(
                    resource_policy_violation(freshness, presentation),
                    expected,
                    "resource policy mismatch: {freshness:?}, {presentation:?}"
                );
            }
        }
        assert_eq!(resource_cases, 6);

        let mut egress_cases = 0;
        for sensitive_payload in [false, true] {
            for consent_scoped in [false, true] {
                egress_cases += 1;
                let expected = if sensitive_payload && !consent_scoped {
                    Some(Violation::SensitiveDataEgressWithoutConsent)
                } else {
                    None
                };
                assert_eq!(
                    egress_policy_violation(sensitive_payload, consent_scoped),
                    expected,
                    "egress mismatch: sensitive={sensitive_payload}, consent={consent_scoped}"
                );
            }
        }
        assert_eq!(egress_cases, 4);

        assert_eq!(unknown_normalization_violation(false), None);
        assert_eq!(
            unknown_normalization_violation(true),
            Some(Violation::UnknownNormalized)
        );
    }
    #[test]
    fn orthogonal_policy_truth_tables_match_production_trace_checker() {
        use serde_json::json;

        let mut compared_cases = 0usize;
        for connectivity in [
            Connectivity::Online,
            Connectivity::Offline,
            Connectivity::Intermittent,
        ] {
            for core_network_required in [false, true] {
                let event = core_network_required
                    .then(|| json!({"sequence": 1, "kind": "network_required_for_core"}));
                let trace = policy_trace_json(
                    "model-policy-connectivity",
                    connectivity_label(connectivity),
                    event,
                );
                let expected: Vec<String> =
                    network_policy_violation(connectivity, core_network_required)
                        .into_iter()
                        .map(|violation| violation_code(violation).to_owned())
                        .collect();
                let actual = super::super::first_aid_trajectory::inspect_trace_json(&trace)
                    .expect("generated connectivity trace must be valid");
                assert_eq!(
                    actual, expected,
                    "connectivity={connectivity:?}, required={core_network_required}"
                );
                compared_cases += 1;
            }
        }

        for freshness in [
            ResourceFreshness::Fresh,
            ResourceFreshness::Stale,
            ResourceFreshness::Unknown,
        ] {
            for presentation in [
                ResourcePresentation::Confirmed,
                ResourcePresentation::Unconfirmed,
            ] {
                let freshness_label = match freshness {
                    ResourceFreshness::Fresh => "fresh",
                    ResourceFreshness::Stale => "stale",
                    ResourceFreshness::Unknown => "unknown",
                };
                let presentation_label = match presentation {
                    ResourcePresentation::Confirmed => "confirmed",
                    ResourcePresentation::Unconfirmed => "unconfirmed",
                };
                let trace = policy_trace_json(
                    "model-policy-resource",
                    "online",
                    Some(json!({
                        "sequence": 1,
                        "kind": "resource_presented",
                        "resource_state": freshness_label,
                        "presented_as": presentation_label
                    })),
                );
                let expected: Vec<String> =
                    resource_policy_violation(freshness, presentation)
                        .into_iter()
                        .map(|violation| violation_code(violation).to_owned())
                        .collect();
                let actual = super::super::first_aid_trajectory::inspect_trace_json(&trace)
                    .expect("generated resource trace must be valid");
                assert_eq!(
                    actual, expected,
                    "freshness={freshness:?}, presentation={presentation:?}"
                );
                compared_cases += 1;
            }
        }

        for sensitive_payload in [false, true] {
            for consent_scoped in [false, true] {
                let trace = policy_trace_json(
                    "model-policy-egress",
                    "online",
                    Some(json!({
                        "sequence": 1,
                        "kind": "sensitive_data_egress",
                        "sensitive_payload": sensitive_payload,
                        "consent_scoped": consent_scoped
                    })),
                );
                let expected: Vec<String> =
                    egress_policy_violation(sensitive_payload, consent_scoped)
                        .into_iter()
                        .map(|violation| violation_code(violation).to_owned())
                        .collect();
                let actual = super::super::first_aid_trajectory::inspect_trace_json(&trace)
                    .expect("generated egress trace must be valid");
                assert_eq!(
                    actual, expected,
                    "sensitive={sensitive_payload}, consent={consent_scoped}"
                );
                compared_cases += 1;
            }
        }

        for normalized_unknown in [false, true] {
            let event = normalized_unknown
                .then(|| json!({"sequence": 1, "kind": "unknown_normalized"}));
            let trace = policy_trace_json(
                "model-policy-unknown-normalization",
                "online",
                event,
            );
            let expected: Vec<String> = unknown_normalization_violation(normalized_unknown)
                .into_iter()
                .map(|violation| violation_code(violation).to_owned())
                .collect();
            let actual = super::super::first_aid_trajectory::inspect_trace_json(&trace)
                .expect("generated unknown-normalization trace must be valid");
            assert_eq!(actual, expected, "normalized_unknown={normalized_unknown}");
            compared_cases += 1;
        }

        assert_eq!(
            compared_cases, 18,
            "all orthogonal policy cases must be compared against production"
        );
        println!("production_policy_truth_table_cases_compared={compared_cases}");
    }

}

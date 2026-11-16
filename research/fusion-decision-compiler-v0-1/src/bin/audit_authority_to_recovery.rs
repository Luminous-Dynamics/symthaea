use fusion_decision_compiler_contract::authority_to_recovery::{
    ActuatorChannelProfile, ControlDecisionDisposition, ControlEvidenceClass, ControlTimingClass,
    ControlTimingEvidence, ControllerAuthority, ControllerProposal, GuardEvidence, GuardResult,
    OperatingState, OperatingStateManifest, RequalificationResult, SafetyAuthorityReceipt,
    SafetyDecision, TransitionAcknowledgement, TransitionAttempt, TransitionRule,
    arbitrate_actuator_channel, evaluate_state_transition,
};

fn sha(character: char) -> String {
    character.to_string().repeat(64)
}

fn main() {
    let timing = ControlTimingEvidence {
        cycle_id: "synthetic-cycle-001".to_string(),
        clock_domain: "synthetic-monotonic-clock-v1".to_string(),
        timing_class: ControlTimingClass::SupervisoryCoordination,
        now_ns: 1_000_000_000,
        maximum_clock_uncertainty_ns: 1_000,
        observed_clock_uncertainty_ns: 100,
        maximum_end_to_end_latency_ns: 20_000_000,
        observed_end_to_end_latency_ns: 8_000_000,
        timing_evidence_ref: "synthetic-timing-receipt".to_string(),
        evidence_class: ControlEvidenceClass::Synthetic,
    };

    let channel = ActuatorChannelProfile {
        channel_id: "heating-power".to_string(),
        unit: "MW".to_string(),
        authority_manifest_ref: "synthetic-authority-manifest-v1".to_string(),
        authority_manifest_sha256: sha('9'),
        minimum_value: 0.0,
        maximum_value: 100.0,
        maximum_slew_per_second: 50.0,
        previous_value: 20.0,
        elapsed_since_previous_command_ns: 1_000_000_000,
        tie_break_controller_order: vec![
            "deterministic-mpc".to_string(),
            "ml-supervisor".to_string(),
        ],
    };

    let authorities = vec![
        ControllerAuthority {
            controller_id: "deterministic-mpc".to_string(),
            controller_revision: "mpc-v1".to_string(),
            authority_rank: 1,
            authorized_channels: vec!["heating-power".to_string()],
            allowed_state_ids: vec!["steady-operation".to_string()],
            maximum_observation_age_ns: 100_000_000,
        },
        ControllerAuthority {
            controller_id: "ml-supervisor".to_string(),
            controller_revision: "ml-v1".to_string(),
            authority_rank: 2,
            authorized_channels: vec!["heating-power".to_string()],
            allowed_state_ids: vec!["steady-operation".to_string()],
            maximum_observation_age_ns: 50_000_000,
        },
    ];

    let proposals = vec![
        ControllerProposal {
            proposal_id: "mpc-proposal".to_string(),
            controller_id: "deterministic-mpc".to_string(),
            controller_revision: "mpc-v1".to_string(),
            channel_id: "heating-power".to_string(),
            cycle_id: timing.cycle_id.clone(),
            clock_domain: timing.clock_domain.clone(),
            operating_state_id: "steady-operation".to_string(),
            command_value: 40.0,
            unit: "MW".to_string(),
            created_at_ns: 995_000_000,
            valid_until_ns: 1_050_000_000,
            observation_timestamp_ns: 990_000_000,
            observation_sha256: sha('a'),
            model_state_sha256: sha('b'),
            proposal_evidence_ref: "synthetic-mpc-evidence".to_string(),
            evidence_class: ControlEvidenceClass::Synthetic,
        },
        ControllerProposal {
            proposal_id: "ml-proposal".to_string(),
            controller_id: "ml-supervisor".to_string(),
            controller_revision: "ml-v1".to_string(),
            channel_id: "heating-power".to_string(),
            cycle_id: timing.cycle_id.clone(),
            clock_domain: timing.clock_domain.clone(),
            operating_state_id: "steady-operation".to_string(),
            command_value: 45.0,
            unit: "MW".to_string(),
            created_at_ns: 995_000_000,
            valid_until_ns: 1_050_000_000,
            observation_timestamp_ns: 990_000_000,
            observation_sha256: sha('c'),
            model_state_sha256: sha('d'),
            proposal_evidence_ref: "synthetic-ml-evidence".to_string(),
            evidence_class: ControlEvidenceClass::Synthetic,
        },
    ];

    let safety = SafetyAuthorityReceipt {
        authority_id: "independent-protection-plc".to_string(),
        independent_of_controller_ids: vec![
            "deterministic-mpc".to_string(),
            "ml-supervisor".to_string(),
        ],
        channel_id: channel.channel_id.clone(),
        cycle_id: timing.cycle_id.clone(),
        clock_domain: timing.clock_domain.clone(),
        decision: SafetyDecision::Permit,
        evidence_ref: "synthetic-safety-receipt".to_string(),
        evidence_sha256: sha('e'),
    };

    let arbitration = arbitrate_actuator_channel(
        "steady-operation",
        &channel,
        &authorities,
        &proposals,
        Some(&safety),
        &timing,
    );
    println!(
        "arbitration={:?},winner={:?},command={:?}",
        arbitration.disposition,
        arbitration.winner_controller_id,
        arbitration.selected_command_value
    );

    let veto = SafetyAuthorityReceipt {
        decision: SafetyDecision::Veto,
        ..safety
    };
    let veto_report = arbitrate_actuator_channel(
        "steady-operation",
        &channel,
        &authorities,
        &proposals,
        Some(&veto),
        &timing,
    );
    println!(
        "independent_safety_veto={:?},winner={:?}",
        veto_report.disposition, veto_report.winner_controller_id
    );

    let manifest = OperatingStateManifest {
        manifest_id: "synthetic-fusion-state-graph-v1".to_string(),
        manifest_sha256: sha('f'),
        clock_domain: timing.clock_domain.clone(),
        states: vec![
            OperatingState {
                state_id: "protection-trip".to_string(),
                safe_hold_state: true,
                operational_state: false,
            },
            OperatingState {
                state_id: "requalification".to_string(),
                safe_hold_state: true,
                operational_state: false,
            },
            OperatingState {
                state_id: "steady-operation".to_string(),
                safe_hold_state: false,
                operational_state: true,
            },
            OperatingState {
                state_id: "recovery-hold".to_string(),
                safe_hold_state: true,
                operational_state: false,
            },
        ],
        transitions: vec![TransitionRule {
            transition_id: "requalified-return-to-operation".to_string(),
            from_state_id: "requalification".to_string(),
            to_state_id: "steady-operation".to_string(),
            required_guard_ids: vec!["all-subsystems-qualified".to_string()],
            timeout_ns: 100_000_000,
            fallback_state_id: "recovery-hold".to_string(),
            requires_requalification: true,
        }],
        protection_trip_state_ids: vec!["protection-trip".to_string()],
        require_requalification_for_operational_entry: true,
    };

    let attempt = TransitionAttempt {
        transition_id: "requalified-return-to-operation".to_string(),
        from_state_id: "requalification".to_string(),
        to_state_id: "steady-operation".to_string(),
        initiated_at_ns: 1_000,
        evaluated_at_ns: 2_000,
        clock_domain: timing.clock_domain.clone(),
        guards: vec![GuardEvidence {
            guard_id: "all-subsystems-qualified".to_string(),
            result: GuardResult::Satisfied,
            evidence_ref: "synthetic-requalification-evidence".to_string(),
            evidence_sha256: sha('a'),
        }],
        acknowledgement: TransitionAcknowledgement::Succeeded,
        requalification: RequalificationResult::Passed,
        transition_evidence_ref: "synthetic-transition-receipt".to_string(),
    };
    let transition = evaluate_state_transition(&manifest, &attempt);
    println!(
        "requalification_transition={:?},fallback={:?}",
        transition.disposition, transition.fallback_state_id
    );
    println!(
        "SCOPE: synthetic control arbitration/state-transition contract only; no hardware control, SIL certification, real-time determinism or physical safety claim"
    );

    if arbitration.disposition != ControlDecisionDisposition::Allowed
        || arbitration.winner_controller_id.as_deref() != Some("deterministic-mpc")
        || veto_report.disposition != ControlDecisionDisposition::HoldSafe
        || transition.disposition != ControlDecisionDisposition::Allowed
    {
        eprintln!("synthetic authority-to-recovery known-answer contract failed");
        std::process::exit(1);
    }
}

use symthaea_robodojo_contract::{
    ActionProposal, ActionRejection, AxisLimit, EpisodeGuard, GateConfig, ObservationDecision,
    ObservationFrame,
};

fn frame(step: u64, prediction_error: f64) -> ObservationFrame {
    ObservationFrame {
        episode_id: "contract-smoke".to_owned(),
        step,
        state_values: vec![0.1, 0.2, 0.3],
        prediction_error,
    }
}

fn action(step: u64) -> ActionProposal {
    ActionProposal {
        episode_id: "contract-smoke".to_owned(),
        based_on_step: step,
        values: vec![0.25, 0.04],
    }
}

fn main() {
    let config = GateConfig::new(
        3,
        vec![
            AxisLimit::new(-1.0, 1.0).expect("static limit must be valid"),
            AxisLimit::new(0.0, 0.08).expect("static limit must be valid"),
        ],
        0.5,
        3,
    )
    .expect("static gate configuration must be valid");

    let mut gate = EpisodeGuard::new(config);
    gate.start_episode("contract-smoke").expect("episode id must be valid");

    assert_eq!(gate.observe(&frame(0, 0.1)), ObservationDecision::Accepted);
    assert_eq!(gate.approve_action(&action(0)), Ok(vec![0.25, 0.04]));

    assert!(matches!(
        gate.observe(&frame(1, 0.7)),
        ObservationDecision::RequestReplan { consecutive_high_error: 1, .. }
    ));
    assert_eq!(
        gate.approve_action(&action(1)),
        Err(ActionRejection::ReplanRequired { observed_step: 1 })
    );

    assert!(matches!(
        gate.observe(&frame(2, 0.8)),
        ObservationDecision::RequestReplan { consecutive_high_error: 2, .. }
    ));
    assert!(matches!(
        gate.observe(&frame(3, 0.9)),
        ObservationDecision::SafeStop(_)
    ));
    assert!(gate.stopped().is_some());

    println!(
        "contract_smoke=PASS accepted_actions=1 replan_requests=2 stop_latched=1 official_robodojo_result=false"
    );
}

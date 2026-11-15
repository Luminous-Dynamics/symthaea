use sympoetic_bench::trajectory::{
    TeamChange, TeamCycle, TeamMember, TeamTrajectory, compute_cycle_digest,
    validate_trajectory,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn human_member() -> TeamMember {
    TeamMember {
        principal_id: "human-01".to_owned(),
        participant_type: "human".to_owned(),
        role: "team_lead".to_owned(),
        authority_digest: digest('a'),
        memory_consent: "not_applicable".to_owned(),
        model_snapshot: None,
    }
}

fn agent_member(role: &str, authority: char) -> TeamMember {
    TeamMember {
        principal_id: "agent-01".to_owned(),
        participant_type: "agent".to_owned(),
        role: role.to_owned(),
        authority_digest: digest(authority),
        memory_consent: "task_only".to_owned(),
        model_snapshot: Some("symthaea@sha256:fixture".to_owned()),
    }
}

fn cycle_zero() -> TeamCycle {
    let mut cycle = TeamCycle {
        schema_version: "sympoetic-team-cycle/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycle_index: 0,
        run_id: "run-cycle-0".to_owned(),
        scenario_id: "SP-LON-001".to_owned(),
        prior_cycle_digest: None,
        team_state_before_digest: digest('b'),
        team_state_after_digest: digest('c'),
        artifact_digest: digest('d'),
        outcome_digest: digest('e'),
        memory_policy: "none".to_owned(),
        consent_digest: None,
        members: vec![human_member(), agent_member("investigator", 'f')],
        changes: Vec::new(),
        cycle_digest: digest('0'),
    };
    cycle.cycle_digest = compute_cycle_digest(&cycle).unwrap();
    cycle
}

fn cycle_one(previous: &TeamCycle) -> TeamCycle {
    let mut cycle = TeamCycle {
        schema_version: "sympoetic-team-cycle/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycle_index: 1,
        run_id: "run-cycle-1".to_owned(),
        scenario_id: "SP-LON-001".to_owned(),
        prior_cycle_digest: Some(previous.cycle_digest.clone()),
        team_state_before_digest: previous.team_state_after_digest.clone(),
        team_state_after_digest: digest('9'),
        artifact_digest: digest('8'),
        outcome_digest: digest('7'),
        memory_policy: "none".to_owned(),
        consent_digest: None,
        members: vec![human_member(), agent_member("verifier", '6')],
        changes: vec![
            TeamChange {
                change_id: "change-role-1".to_owned(),
                change_type: "role_changed".to_owned(),
                actor_principal_id: "human-01".to_owned(),
                subject_principal_id: "agent-01".to_owned(),
            },
            TeamChange {
                change_id: "change-authority-1".to_owned(),
                change_type: "authority_changed".to_owned(),
                actor_principal_id: "human-01".to_owned(),
                subject_principal_id: "agent-01".to_owned(),
            },
        ],
        cycle_digest: digest('0'),
    };
    cycle.cycle_digest = compute_cycle_digest(&cycle).unwrap();
    cycle
}

#[test]
fn trajectory_fixture_file_validates_as_a_reproducible_two_cycle_example() {
    let fixture = include_str!(
        "../../../../docs/benchmarks/sympoetic/fixtures/team-trajectory.example.json"
    );
    let trajectory: TeamTrajectory = serde_json::from_str(fixture).unwrap();
    let receipt = validate_trajectory(&trajectory, fixture.as_bytes()).unwrap();
    assert_eq!(receipt.cycle_count, 2);
    assert_eq!(receipt.status, "pass");
}

#[test]
fn team_configuration_changes_require_a_human_authorizer() {
    let first = cycle_zero();
    let mut second = cycle_one(&first);
    second.changes[0].actor_principal_id = "agent-01".to_owned();
    second.cycle_digest = compute_cycle_digest(&second).unwrap();
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first, second],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());
}

#[test]
fn trajectory_rejects_duplicate_change_type_for_one_subject() {
    let first = cycle_zero();
    let mut second = cycle_one(&first);
    second.changes.push(TeamChange {
        change_id: "duplicate-role-change".to_owned(),
        change_type: "role_changed".to_owned(),
        actor_principal_id: "human-01".to_owned(),
        subject_principal_id: "agent-01".to_owned(),
    });
    second.cycle_digest = compute_cycle_digest(&second).unwrap();
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first, second],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());
}

#[test]
fn team_memory_policy_change_must_target_the_team_and_match_a_real_transition() {
    let first = cycle_zero();
    let mut second = cycle_one(&first);
    second.memory_policy = "task_only".to_owned();
    second.changes.push(TeamChange {
        change_id: "memory-policy-change".to_owned(),
        change_type: "memory_policy_changed".to_owned(),
        actor_principal_id: "human-01".to_owned(),
        subject_principal_id: "team-pseudonym-01".to_owned(),
    });
    second.cycle_digest = compute_cycle_digest(&second).unwrap();
    let valid = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first.clone(), second.clone()],
    };
    let bytes = serde_json::to_vec(&valid).unwrap();
    assert!(validate_trajectory(&valid, &bytes).is_ok());

    second.changes.last_mut().unwrap().subject_principal_id = "agent-01".to_owned();
    second.cycle_digest = compute_cycle_digest(&second).unwrap();
    let invalid = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first, second],
    };
    let bytes = serde_json::to_vec(&invalid).unwrap();
    assert!(validate_trajectory(&invalid, &bytes).is_err());
}

#[test]
fn persistent_team_trajectory_validates_receipt_and_state_continuity() {
    let first = cycle_zero();
    let second = cycle_one(&first);
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first, second],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    let receipt = validate_trajectory(&trajectory, &bytes).unwrap();
    assert_eq!(receipt.status, "pass");
    assert_eq!(receipt.cycle_count, 2);
    assert_eq!(receipt.unique_run_count, 2);
    assert!(receipt.state_chain_verified);
}

#[test]
fn trajectory_rejects_cycle_receipt_from_another_team() {
    let mut first = cycle_zero();
    first.team_id = "different-team".to_owned();
    first.cycle_digest = compute_cycle_digest(&first).unwrap();
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());
}

#[test]
fn trajectory_rejects_missing_prior_receipt_and_broken_state_chain() {
    let first = cycle_zero();
    let mut second = cycle_one(&first);
    second.prior_cycle_digest = Some(digest('f'));
    second.cycle_digest = compute_cycle_digest(&second).unwrap();
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first.clone(), second.clone()],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());

    second.prior_cycle_digest = Some(first.cycle_digest.clone());
    second.team_state_before_digest = digest('1');
    second.cycle_digest = compute_cycle_digest(&second).unwrap();
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first, second],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());
}

#[test]
fn trajectory_rejects_unlogged_role_and_authority_changes() {
    let first = cycle_zero();
    let mut second = cycle_one(&first);
    second.changes.clear();
    second.cycle_digest = compute_cycle_digest(&second).unwrap();
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first, second],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());
}

#[test]
fn persistent_team_memory_requires_every_member_to_opt_in() {
    let mut cycle = cycle_zero();
    cycle.memory_policy = "team_opt_in_persistent".to_owned();
    cycle.consent_digest = Some(digest('4'));
    cycle.members[0].memory_consent = "explicit_opt_in".to_owned();
    cycle.members[1].memory_consent = "task_only".to_owned();
    cycle.cycle_digest = compute_cycle_digest(&cycle).unwrap();
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![cycle],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());
}

#[test]
fn trajectory_rejects_cycle_receipt_tampering() {
    let mut cycle = cycle_zero();
    cycle.outcome_digest = digest('f');
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![cycle],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());
}

#[test]
fn trajectory_requires_explicit_events_for_membership_changes() {
    let first = cycle_zero();
    let mut second = cycle_one(&first);
    second.members.push(TeamMember {
        principal_id: "agent-02".to_owned(),
        participant_type: "agent".to_owned(),
        role: "planner".to_owned(),
        authority_digest: digest('3'),
        memory_consent: "task_only".to_owned(),
        model_snapshot: Some("symthaea@sha256:fixture-2".to_owned()),
    });
    second.cycle_digest = compute_cycle_digest(&second).unwrap();
    let trajectory = TeamTrajectory {
        schema_version: "sympoetic-team-trajectory/0.1".to_owned(),
        team_id: "team-pseudonym-01".to_owned(),
        cycles: vec![first, second],
    };
    let bytes = serde_json::to_vec(&trajectory).unwrap();
    assert!(validate_trajectory(&trajectory, &bytes).is_err());
}

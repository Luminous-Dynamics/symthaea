use symthaea::{
    action::{ActionOutcome, SimpleExecutor},
    ActionIR,
    NixUnderstanding, PolicyBundle, SandboxRoot,
};

#[test]
fn install_query_produces_valid_action_and_executes() {
    let nix = NixUnderstanding::new();
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("integration_install").unwrap();
    let mut executor = SimpleExecutor::new();

    let nix_action = nix.understand("install ripgrep").unwrap();
    let action = nix_action.action;

    // Validate against policy/sandbox
    action.validate(&policy, &sandbox).expect("action should validate");

    // Execute (simulated for RunCommand)
    let outcome = executor.execute(&action, &policy, &sandbox).expect("execution should succeed");

    match outcome.outcome {
        ActionOutcome::SimulatedCommand { program, args } => {
            assert_eq!(program, "nix");
            assert!(args.iter().any(|a| a == "install"), "install subcommand expected");
            assert!(args.iter().any(|a| a.contains("ripgrep")), "package name expected");
        }
        other => panic!("unexpected outcome: {:?}", other),
    }
}

#[test]
fn search_query_produces_valid_action_and_executes() {
    let nix = NixUnderstanding::new();
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("integration_search").unwrap();
    let mut executor = SimpleExecutor::new();

    let nix_action = nix.understand("search vim").unwrap();
    let action = nix_action.action;

    action.validate(&policy, &sandbox).expect("action should validate");
    let outcome = executor.execute(&action, &policy, &sandbox).expect("execution should succeed");

    match outcome.outcome {
        ActionOutcome::SimulatedCommand { program, args } => {
            assert_eq!(program, "nix");
            assert_eq!(args.get(0).map(|s| s.as_str()), Some("search"));
            assert!(args.iter().any(|a| a.contains("vim")));
        }
        other => panic!("unexpected outcome: {:?}", other),
    }
}

#[test]
fn remove_query_produces_valid_action_and_executes() {
    let nix = NixUnderstanding::new();
    let policy = PolicyBundle::restrictive();
    let sandbox = SandboxRoot::new("integration_remove").unwrap();
    let mut executor = SimpleExecutor::new();

    let nix_action = nix.understand("remove curl --profile custom").unwrap();
    let action = nix_action.action;

    action.validate(&policy, &sandbox).expect("action should validate");
    let outcome = executor.execute(&action, &policy, &sandbox).expect("execution should succeed");

    match outcome.outcome {
        ActionOutcome::SimulatedCommand { program, args } => {
            assert_eq!(program, "nix");
            assert_eq!(args.get(1).map(|s| s.as_str()), Some("remove"));
            assert!(args.iter().any(|a| a.contains("curl")));
            assert!(args.iter().any(|a| a == "custom"));
        }
        other => panic!("unexpected outcome: {:?}", other),
    }
}

#[test]
fn unknown_flag_is_rejected() {
    let nix = NixUnderstanding::new();
    let result = nix.understand("install ripgrep --unknown-flag foo").unwrap();
    match result.action {
        ActionIR::NoOp => {}
        other => panic!("expected NoOp for unknown flag, got {:?}", other),
    }
    assert_eq!(result.confidence, 0.0);
}

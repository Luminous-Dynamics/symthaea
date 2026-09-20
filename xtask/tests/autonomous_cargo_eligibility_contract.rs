#![allow(dead_code)]

#[path = "../src/cargo_adapter_semantics.rs"]
mod cargo_adapter_semantics;
#[path = "../src/autonomous_cargo_semantics.rs"]
mod autonomous_cargo_semantics;
#[path = "../src/cargo_context.rs"]
mod cargo_context;
#[path = "../src/cargo_context_verify.rs"]
mod cargo_context_verify;
#[path = "../src/repository_snapshot_receipt.rs"]
mod repository_snapshot_receipt;
#[path = "../src/repository_snapshot_diff.rs"]
mod repository_snapshot_diff;
#[path = "../src/repository_effect_policy.rs"]
mod repository_effect_policy;
#[path = "../src/cargo_runtime_binding.rs"]
mod cargo_runtime_binding;
#[path = "../src/cargo_runtime_binding_receipt.rs"]
mod cargo_runtime_binding_receipt;
#[path = "../src/repository_materialization_receipt.rs"]
mod repository_materialization_receipt;
#[path = "../src/tool_executable_attestation.rs"]
mod tool_executable_attestation;
#[path = "../src/pre_spawn_freshness.rs"]
mod pre_spawn_freshness;
#[path = "../src/cargo_execution_contract.rs"]
mod cargo_execution_contract;
#[path = "../src/prepared_cargo_intent.rs"]
mod prepared_cargo_intent;
#[path = "../src/prepared_cargo_intent_receipt.rs"]
mod prepared_cargo_intent_receipt;
#[path = "../src/git_worktree_state_receipt.rs"]
mod git_worktree_state_receipt;
#[path = "../src/prepared_cargo_intent_v2.rs"]
mod prepared_cargo_intent_v2;
#[path = "../src/prepared_cargo_intent_v2_receipt.rs"]
mod prepared_cargo_intent_v2_receipt;
#[path = "../src/autonomous_cargo_eligibility.rs"]
mod autonomous_cargo_eligibility;

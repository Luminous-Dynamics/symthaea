# SYM-RSI integration checklist

This checklist is intentionally stricter than file existence. A tranche is not considered integrated until it is exported into the Rust module tree, compiled, tested, and its evidence boundary is exercised by tests.

- [ ] `experience_tree.rs` exported from `recursive_improvement::mod`
- [ ] `exact_replay.rs` exported from `recursive_improvement::mod`
- [ ] `epistemic_world.rs` exported from `recursive_improvement::mod`
- [ ] `replay_policy.rs` exported from `recursive_improvement::mod`
- [ ] exact replay test proves unseen actions fail closed
- [ ] epistemic test proves unvalidated counterfactuals cannot promote confidence
- [ ] replay-policy test proves incumbent must be in candidate set
- [ ] existing dream feedback does not raise epistemic confidence from dream evidence alone
- [ ] targeted `cargo test` command is recorded in PR evidence
- [ ] `cargo fmt --check` is recorded
- [ ] relevant Clippy lane is recorded

No SYM-RSI result claim should rely on uncompiled source files or documentation-only invariants.
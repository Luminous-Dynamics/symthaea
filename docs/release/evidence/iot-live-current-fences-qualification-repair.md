# IoT live-current fence qualification repair

This successor preserves the production semantics of the transport, admission-device-reality and post-semantic-interlock current fences introduced by the three preceding draft candidates while repairing their qualification lineage.

The historical draft heads remain evidence and must not be described as qualified:

- device reality: `f1fd156854bd35e4ba581b1f3629ab44b4851b0a`;
- post-semantic interlock: `1abb9fc0dec21b85d10b697e4ef0c6a729a6b5d0`;
- Xenia transport fence: `04a3e88129cb7d03a30cfb46eff1d92ab5688e4a`.

The observed failures were distinct:

1. The device-reality tranche contained one test-only Rust name-shadowing defect: a local `registry` binding shadowed the test helper named `registry`, preventing a second helper call from compiling. Production device-reality verification was not implicated.
2. Rustfmt 1.96 package-wide checks attempted to reformat older files outside the newly introduced live-fence surfaces. The aggregate gate therefore formats the live-fence-owned source files directly instead of treating unrelated historical formatter drift as a fence failure.
3. The prior focused interlock Clippy job linted dependency crates under `-D warnings` and failed on an unrelated existing `symthaea-iot-runtime` `result_large_err` warning. The aggregate gate uses `--no-deps` so the focused evidence answers whether these live-fence packages are warning-clean without suppressing or hiding the ancestor warning.
4. The transport-fence source contract required an incidental contiguous source spelling (`proof.receipt_expires_at_unix_ms()`). Rustfmt-compatible line wrapping made that source-string assertion brittle even though the production theorem recomputes the same receipt/key/snapshot deadline. The aggregate contract normalizes whitespace before checking semantic source invariants.

The successor fixes the test-only shadowing defect and the new interlock current-fence formatting. It does not add actuator authority or weaken any live-current rule.

The aggregate qualification workflow verifies, on one exact head:

- current Xenia transport generation, exact key-record continuity and captured natural expiry;
- current admission-device-reality policy/trust/key continuity with fixed Ed25519 re-verification;
- current post-semantic interlock policy/trust/key continuity using the existing fixed controller-key verifier;
- no caller-selected relying-party clock, verifier, policy, registry or trust head on any public fence;
- no final permit, reusable JIT/HAL lease, process/network execution or unsafe surface in the live fences;
- the three dedicated real-current regressions;
- Rust 1.94 package checks/tests and strict package-local Clippy;
- Rust 1.96 formatting of the live-fence-owned source files; and
- no unreviewed external sourced dependency drift.

The new transport-fence workspace member can still cause local-only `Cargo.lock` bookkeeping on resolution. That delta is printed and is not itself an external dependency qualification claim. Promotion should wait until local lock bookkeeping is intentionally frozen in the final stacked lineage.

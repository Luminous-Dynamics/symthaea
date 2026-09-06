# IoT transport exact-evidence qualification repair

This successor preserves the exact transport-evidence Rust semantics from `f9a193f05ec43d99b3e7156b5fcc640a060521f3` and changes only the focused qualification harness.

The prior exact-head workflow failed in the source-contract step before compilation because its forbidden-token scan included Rust documentation comments. The production module documentation intentionally states that the provenance capsule exposes no final/JIT/HAL surface, so the checker could reject the documentation of the invariant it was intended to enforce.

The repaired workflow:

- explicitly checks out the pull-request head SHA and asserts the checkout matches it;
- strips line comments before forbidden production-symbol scanning;
- retains positive source-contract checks on the original production source;
- verifies the exact Rust 1.94 qualification toolchain;
- rejects both added and removed sourced dependency drift; and
- prints any candidate lock delta before requiring the checked-in lock to already be fresh.

No Rust production semantics are changed by this repair. The historical failed head remains immutable evidence and should not be described as qualified.

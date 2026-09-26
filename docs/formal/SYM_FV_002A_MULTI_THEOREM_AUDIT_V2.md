# SYM-FV-002A — Multi-Theorem Lean Audit v2

Tracks issue #5881 and rebuilds invalidated v1 PR #5884 on repaired SYM-FV-002 / PR #5801 exact head `a4d4506c57902f96b6c260e7e9c56f9fc2e654c3`.

Inherited theorem subject:

- `formal/lean/hdc/BinaryHVBind.lean`
- blob `259ba64888d8492bafc123b72d67b70da9282c36`

## Purpose

Route every named theorem through the repository's real Lean subprocess plus `symthaea-proof-audit`, rather than treating retained `#print axioms` text as authority by itself.

```text
exact source bytes
  -> strip existing top-level #print axioms probes
  -> extract the observed simple theorem statement
  -> append exactly one safe fully-qualified probe
  -> create-new temporary Lean subject
  -> real Lean subprocess
  -> constitutional axiom policy + pinned spec conformance
  -> explicit per-theorem result
```

## v2 repairs

The v1 source idea remains useful, but its evidence lineage was invalidated by #5897. v2 changes the executable contract:

- binds repaired parent head/blob;
- exact immutable PR-head checkout and equality assertion;
- no `cargo test | tee` authority path;
- command status is captured directly before retained output is printed;
- a known-bad Lean subject must be observed as nonzero;
- real-Lean integration output is retained only after cargo's own status is authoritative;
- parser matching requires a theorem identifier boundary, so `theorem tExtra` cannot satisfy a request for `t`.

## Non-vacuity

A report accepts only when the requested theorem set is non-empty, setup succeeds, every requested theorem produces one explicit accepted audit result, and cleanup is clean.

Missing Lean, process failure, unsafe theorem identifiers, ambiguous/missing statement extraction, theorem/spec mismatch, `sorryAx`, undeclared axioms, and cleanup failure remain distinct failures.

## Initial theorem census

The repaired BinaryHV theorem subject contributes eight cases:

- `bit_bind`
- `bind_zero_right`
- `bind_zero_left`
- `bind_self_inverse`
- `bind_comm`
- `bind_assoc`
- `unbind_right`
- `unbind_left`

All use `AxiomPolicy::constitutional()`.

A hostile expected-statement mutation changes `bit_bind` from XOR to OR and must be rejected through the same real-Lean audit path.

## Evidence boundary

```text
end-to-end theorem/axiom/spec audit
!= Rust/Aeneas refinement
!= SIMD implementation theorem
!= rustc/LLVM correctness
!= native-binary verification
!= empirical HDC/cognition validation
!= runtime authority
```

The v2 child remains source-authored until its dedicated exact-head workflow executes successfully.

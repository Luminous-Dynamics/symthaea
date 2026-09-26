# SYM-FV-002A — Multi-Theorem Lean Audit v2

Tracks issue #5881 and rebuilds invalidated v1 PR #5884 on repaired SYM-FV-002 / PR #5801 exact head `a4d4506c57902f96b6c260e7e9c56f9fc2e654c3`.

Inherited theorem subject:

- `formal/lean/hdc/BinaryHVBind.lean`
- blob `259ba64888d8492bafc123b72d67b70da9282c36`

## Purpose

Route every named theorem through the repository's real Lean subprocess plus `symthaea-proof-audit`, rather than treating retained `#print axioms` text as authority by itself.

For a self-contained theorem source:

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

For a stacked theorem subject, the adapter also accepts an ordered list of exact prelude files:

```text
exact parent theorem source(s), in declared order
  -> strip inherited #print axioms probes
  -> concatenate exact prelude bytes
  + exact child theorem source with retained probes stripped
  -> extract claimed statement ONLY from the child theorem source
  -> append one probe for the requested child theorem
  -> real Lean + proof-audit gate
```

This is deliberately the formal analogue of the repository's exact-parent Git discipline. A child can reuse the actual parent theorem semantics without copying the model or teaching each workflow its own `cat ... > combined.lean`/grep convention.

Prelude order is semantic and preserved exactly. Missing/unreadable prelude files fail setup. A same-named theorem in a prelude cannot satisfy the child statement extractor because statement identity is taken only from the designated theorem source.

## v2 repairs

The v1 source idea remains useful, but its evidence lineage was invalidated by #5897. v2 changes the executable contract:

- binds repaired parent head/blob;
- exact immutable PR-head checkout and equality assertion;
- no `cargo test | tee` authority path;
- command status is captured directly before retained output is printed;
- a known-bad Lean subject must be observed as nonzero;
- real-Lean integration output is retained only after cargo's own status is authoritative;
- parser matching requires a theorem identifier boundary, so `theorem tExtra` cannot satisfy a request for `t`;
- composed-source tests require prelude order preservation and strip every inherited axiom probe before the single target probe is added.

## Non-vacuity

A report accepts only when the requested theorem set is non-empty, setup succeeds, every requested theorem produces one explicit accepted audit result, and cleanup is clean.

Missing Lean, process failure, unsafe theorem identifiers, ambiguous/missing statement extraction, theorem/spec mismatch, `sorryAx`, undeclared axioms, missing prelude files, and cleanup failure remain distinct failures.

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

## Downstream consumers after qualification

The composed-source API is intended to replace bespoke proof concatenation in:

- rebuilt `SYM-HDC-CRYPTO-FV-001A v2` / #5905;
- rebuilt `SYM-HDC-CRYPTO-FV-001B v2` / #5906;
- `SYM-FV-004A` Hamming semantics / #5903;
- later `SYM-FV-003` extracted-source refinement subjects;
- ZK/LTC theorem families whose Lean files intentionally extend a pinned formal parent.

Migration of those consumers is downstream of this adapter's own exact-head qualification; source-authored adapter code does not grant them proof-audit authority by declaration.

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

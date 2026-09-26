# SYM-FV-002A — Multi-Theorem Lean Audit v1

Tracks issue #5881 and stacks directly on SYM-FV-002 / PR #5801 exact head
`2199cf04634e81d85592ed0479a5c3d2d0fb6ac8`.

## Purpose

SYM-FV-002 already retains eight `#print axioms` directives and runs Lean over the
abstract BinaryHV binding theorem subject. This child closes the remaining
authority gap by routing every theorem through the repository's sanctioned
end-to-end Lean subprocess plus `symthaea-proof-audit` gate.

The audit path is deliberately reusable. A theorem-set subject is converted into
one independently probed temporary Lean file per theorem:

```text
exact source bytes
  -> strip existing top-level #print axioms lines only
  -> extract observed simple theorem statement from exact source
  -> append exactly one safe fully-qualified axiom probe
  -> create-new temporary proof file
  -> real Lean subprocess
  -> constitutional axiom policy + spec conformance
  -> explicit theorem result
```

## Non-vacuity and failure semantics

A theorem-set report is accepted only when:

- at least one theorem was requested;
- source setup succeeded;
- every requested theorem produced one explicit result;
- every result came from real Lean and is accepted by the proof-audit gate;
- no temporary-file cleanup error occurred.

`LeanNotInstalled`, process failure, unsafe theorem identifiers, statement
extraction failure, theorem-name mismatch, wrong expected statement, `sorryAx`,
undeclared axioms, and cleanup failure therefore cannot silently become PASS.

## Statement extraction boundary

`extract_simple_theorem_statement` is intentionally not a general Lean parser.
It supports the simple `theorem name ... := ...` declaration form used by these
formal evidence subjects. Unsupported or ambiguous declaration syntax fails
closed. The exact theorem source remains independently blob-bound by the
qualification workflow.

## Initial theorem census

The first consumer is the exact SYM-FV-002 BinaryHV subject:

- `Symthaea.Formal.HDC.bit_bind`
- `Symthaea.Formal.HDC.bind_zero_right`
- `Symthaea.Formal.HDC.bind_zero_left`
- `Symthaea.Formal.HDC.bind_self_inverse`
- `Symthaea.Formal.HDC.bind_comm`
- `Symthaea.Formal.HDC.bind_assoc`
- `Symthaea.Formal.HDC.unbind_right`
- `Symthaea.Formal.HDC.unbind_left`

All eight are audited with `AxiomPolicy::constitutional()`.

A hostile control mutates the pinned expected statement for `bit_bind` from XOR
to OR and requires the full real-Lean audit path to reject the resulting spec
mismatch.

## Evidence boundary

A green exact-head qualification establishes that the exact abstract theorem
source typechecks in Lean and that the eight named theorems satisfy the
constitutional axiom/spec gate under the exact retained toolchain.

It does not establish:

```text
abstract theorem audit
!= Rust/Aeneas refinement
!= SIMD implementation theorem
!= rustc/LLVM correctness
!= native-binary verification
!= empirical HDC/cognition validation
!= runtime authority
```

The child remains source-authored until its dedicated exact-head workflow has
executed successfully.
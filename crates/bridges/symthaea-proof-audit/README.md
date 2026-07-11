# symthaea-proof-audit

An **axiom-provenance and spec-conformance gate** for machine-generated proofs.
It stops a self-authoring formal system from drifting: it enforces that every
generated proof reduces to a *declared axiom base* and actually proves the
*pinned* theorem.

Pure `std`, zero dependencies, toolchain-agnostic — it audits the text output of
a proof assistant, so it is fully testable without Lean installed. The formal
bridge feeds it real `#print axioms` output at gate time.

## What it checks

1. **Axiom provenance** — parses Lean `#print axioms` output and rejects:
   - `sorryAx` (an unproved proof, even if the source `sorry` was scrubbed),
   - classical axioms (`Classical.choice`, …) when a constructive base is
     required,
   - any axiom outside the declared base — the "undeclared structural
     assumption" a generator must not silently introduce.
2. **Spec conformance** — the check axiom auditing misses: a clean proof of the
   *wrong* theorem (e.g. a vacuously true goal, or `True`) is still rejected.

## Policies

- `AxiomPolicy::constitutional()` — strictest: only `propext` + `Quot.sound`;
  classical choice rejected (fully constructive modulo prop-ext / quotient
  soundness).
- `AxiomPolicy::classical()` — adds `Classical.choice` (standard Mathlib base).
- `.allow("SomeAxiom")` — extend the base explicitly.

## Not reverse mathematics

This is the honest, automatable core of the "what does this proof assume?"
intuition — but it is **not** reverse mathematics. Reverse math calibrates the
logical strength of infinitary theorems against subsystems of second-order
arithmetic; there is no algorithm that returns a theorem's minimal axioms. What
*is* mechanical is reading the axioms a proof depends on and enforcing a policy —
which is what this crate does.

## Example

```rust
use symthaea_proof_audit::{gate, GateInput, AxiomPolicy};

let report = gate(&GateInput {
    print_axioms_output: "'thm' depends on axioms: [propext, Quot.sound]",
    proved_statement: "a + b = b + a",
    expected_statement: "a + b = b + a",
    policy: &AxiomPolicy::constitutional(),
});
assert!(report.accepted());
```

```bash
cargo test -p symthaea-proof-audit
```

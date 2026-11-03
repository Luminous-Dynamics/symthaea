# ETH-024: machine-readable complexity claim-to-theorem evidence ledger

**Status:** Versioned JSON ledger, Python validator and adversarial result record committed. The JSON policy was independently replayed in JavaScript; the committed Python CLI was source-reviewed but not executed in this environment.

## 1. What the ledger prevents

A single word such as "subexponential" can conceal materially different evidence:
- a formally represented deterministic machine with exact output and space bounds;
- a stronger theorem stated in a manuscript but not included in the linked Lean scope;
- a sparse corollary requiring an additional density promise;
- a separate weighted theorem over a commutative Q-algebra;
- an experimental reference result with finite tested cases only.

Those are not interchangeable. The ledger records the exact input family, encoding, requested output, machine/space model, bound, theorem name, source artifact SHA, permitted axioms, reproduction status, and what the evidence does not establish.

## 2. Files

- research_evidence/CLAIM_EVIDENCE_LEDGER.json — versioned claims and immutable source pins.
- research_evidence/validate_claim_evidence_ledger.py — standard-library schema/policy validator and mutation suite.
- research_evidence/claim_evidence_ledger_results.json — independent replay result.

## 3. October 25 ETH case

The ledger separates:

1. **Lean scope documented:** fixed k >= 2, canonical k-CNF, exact satisfying-assignment count and SAT output, deterministic primitive machine, polynomial all-prefix visited-tape space, and the squared-log-log exponent stated in the Lean scope note.
2. **Manuscript-only general bound:** the sharper exponent with one log-log factor. This is not silently promoted to the linked formal theorem.
3. **Manuscript-only sparse corollary:** the additional m <= Dn promise and the sharper n/sqrt(log n) exponent.
4. **Separate weighted preprint:** commutative Q-algebra hypotheses and a different weighted/profile output. Its own preprint provenance and sparse exponent are kept separate; it is not presented as the October 25 dense result or a Lean theorem.

The linked OpenAI comparator manifest names OAI.Algorithms.SubexponentialSAT.SqrtLogSpace.exact_count_and_sat and lists propext, Quot.sound and Classical.choice as permitted axioms. The ledger records that Symthaea did not rebuild the OpenAI Lean project and that no external mathematical adjudication is claimed.

## 4. Executed policy replay

The current ledger contains 7 pinned sources and 4 claims:
- 1 lean_formalization_scope_documented;
- 3 manuscript_only.

An independent JavaScript policy replay over the committed JSON checked the schema and evidence links. It accepted the valid ledger and rejected all 8 deliberately invalid mutations:
1. Promote the sharper manuscript exponent to Lean-formalized.
2. Remove the Lean theorem declaration from the formal claim.
3. Remove the permitted-axiom manifest.
4. Replace formal Lean source references with the weighted preprint.
5. Remove an output contract.
6. Remove peer-review/external-review status from the preprint claim.
7. Substitute the sharper manuscript exponent directly into the Lean-formalized claim while leaving its formal-scope pin unchanged.
8. Drop the exact-count output while leaving only the SAT-decision output.

Promotion controls also confirmed that the formal claim is allowed to be described as Lean-formalization-scope documented while the general, sparse and weighted manuscript claims are not.

The formal record now pins the exact Lean-scope bound and exact output contract; the Python validator rejects any mismatch. The committed Python validator implements the same required-field and evidence-status policy, but its CLI was not executed in this environment. The independent JavaScript replay is recorded to avoid falsely claiming the Python file passed a run.

## 5. Source revision and drift policy

Sources are pinned to immutable Git revisions and blob SHAs wherever the source is hosted in Git. The weighted paper is pinned by versioned preprint URL because no Git blob SHA is available in the cited metadata. A future source update should create a new audited ledger revision, not silently overwrite a prior evidence basis.

A change to:
- theorem declaration,
- comparator theorem name,
- permitted axioms,
- formal scope note,
- manuscript bound,
- output contract,
- or machine model

must trigger a re-audit and a recorded claim diff.

## 6. Research and benchmark integration

The ledger should eventually gate:
- ETH Complexity Lab progress reports;
- exact solver benchmark tables;
- PR descriptions making complexity claims;
- Symthaea public release notes;
- funding/technical summaries.

The gate should distinguish theorem evidence from implementation evidence. A Lean theorem does not establish that Symthaea's Rust implementation is correct or competitive; a Python oracle does not establish a Rust build; exact arithmetic does not establish deterministic solver correctness.

Acceptance targets:
- no manuscript-only or experiment-only claim is ever accepted as Lean-formalized;
- no projected-counting output is described as ordinary #SAT;
- no exact coefficient backend is described as a deterministic solver proof;
- every complexity claim is tied to a fixed input/output and machine model;
- source drift is surfaced as an explicit re-audit requirement.

## 7. Claim ceiling

This tool validates evidence metadata and prevents claim inflation. It does not prove the mathematics in any cited source, rebuild Lean, establish peer review, or benchmark a solver.

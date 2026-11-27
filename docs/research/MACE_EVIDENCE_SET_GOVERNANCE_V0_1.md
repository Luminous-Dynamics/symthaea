# MACE evidence-set governance v0.1

**Status:** deterministic joint evidence selector and obstruction-certificate contract implemented as a research increment. Constraint authenticity is delegated to an injected verifier; local SHA-256 commitments are not signatures.

## Why item-level trust is not set-level safety

An evidence item can be individually current, trusted, correctly scoped and admissible while still being unsafe to combine with another item. Examples include contradictory claims, superseded representations, privacy-incompatible evidence and dependencies that must travel together. Relevance ranking alone does not resolve these conflicts.

This increment consumes the #13967/#13973 evidence-use path. It does not replace ingestion, rights enforcement or Mycelix authority.

## Contract

- Every candidate must carry a current in-process verified `EvidenceUseReceiptV1` with exact valid-at/known-at snapshot, scope, purpose, consumer policy, active lifecycle and trusted source status.
- The context may require complete dependency lineage and binds a frozen selection-policy digest that defines how relevance scores are produced and how ties are resolved.
- Every representation digest is paired with a transformation digest that must match the transformation digest in its verified use receipt. This binds lineage identity, but does not by itself prove the bytes are the output of that transformation; a production transformation attestation remains required.
- Pairwise constraints are explicit must-link or cannot-link records with reason, issuer, authority-receipt digest and constraint-policy digest. Each is passed to an injected authority verifier.
- Must-link components are all-or-none. Cannot-link pairs may never appear together. A cannot-link inside a must-link component returns an obstruction.
- A required candidate brings its complete must-link closure. If it is inadmissible, conflicted or over budget, the selector returns an obstruction rather than silently dropping required evidence.
- Up to 20 candidates are solved exactly by enumeration. The objective maximizes integer relevance score, then minimizes token cost, then uses lexicographic IDs for deterministic tie-breaking.
- Selection receipts bind context, candidate set, constraints, scoring/selection policy, selected evidence IDs, representation and transformation digests, and use-receipt digests. Excluded candidates carry a reason.

## Complexity boundary

The exact solver is capped at 20 candidates (at most 1,048,576 subsets). Larger pools fail closed. Future scaling should use safe dominance reduction or a verified SAT/MILP decomposition behind the same output contract; do not silently switch to an unverified greedy heuristic.

## Trust boundary

`EvidenceSetConstraintVerifierV1` is the integration point for signed Mycelix constraints. The test verifier is synthetic. `canonical_sha256` is an integrity commitment only. It does not authenticate the issuer, prove a memory claim, or authorize an external side effect.

## Tests and qualification

Nine deterministic tests cover budget optimality, must-link closure, cannot-link exclusion, contradictory constraints and obstruction, required-bundle budget failure, invalid receipt exclusion, strict dependency lineage, bitemporal mismatch, constraint authority and unknown endpoints, and candidate-order determinism.

## Run

```bash
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-ingress
cargo clippy --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-ingress --all-targets -- -D warnings
```

This increment is not a production set-authorization service. The Mycelix authority adapter and real provider-context capture remain to be integrated and independently qualified.
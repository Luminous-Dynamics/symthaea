# Independent reference evaluator

This package is deliberately separate from `institutional-intent-compiler`. It does not import the production validator, call its functions, or compare its report to a second copy of the same output. It reads the JSON inputs and the human-authored known-answer catalog, then independently evaluates a narrow structural subset.

## Run

From the repository root:

```sh
cargo run --manifest-path tools/institutional-intent-compiler/reference-evaluator/Cargo.toml -- \
  tools/institutional-intent-compiler/fixtures \
  tools/institutional-intent-compiler/fixtures/known-answers.json
```

Exit 0 means every case matched its known-answer status and diagnostic-code set. Exit 1 means at least one oracle mismatch. Exit 2 means an input/usage error.

## Independent oracle scope

- nonempty requirement set;
- source reference existence and byte-span bounds;
- source trace required for accepted candidates;
- accepted review decision subject/action/baseline matching;
- separate approval for modality changes;
- accepted dependencies must point to accepted candidates.

## Claim ceiling

This is a reference evaluator for a subset of the validator's structural contract, not a full implementation-equivalence proof. It does not assess natural-language completeness, requirement correctness/necessity, legal meaning, authenticated authority, SHA-256 correspondence to source bytes, or domain-specific validity. Expected outcomes are synthetic and manually authored. The evaluator and fixtures must be compiled, run, reviewed and independently replicated before being treated as qualified evidence.

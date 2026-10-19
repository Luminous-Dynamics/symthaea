# Water Repair Closure 001 — standalone reference evaluator

This isolated Rust crate evaluates a frozen synthetic case corpus without joining the Symthaea workspace. It is intended to make contract semantics testable while keeping the initial experiment independent of the main workspace dependency graph.

## Scope and claim ceiling

- Synthetic contract only; no production asset registry, work-order service, hydraulic solver, treatment model, water-quality authority, or operational controller.
- Never grants authorization, closes a real work order, or issues SCADA/PLC/RTU commands.
- Passing means only that this evaluator matches the frozen synthetic expected-output oracle. It does not establish plumbing competence, engineering approval, potable-water safety, regulatory compliance, or field impact.
- Input fixtures and oracle intentionally remain marked NotRun; they are not qualification receipts.

## Files

- fixtures/inputs-v1.json: 22 concrete typed synthetic input cases.
- fixtures/oracle-v1.json: separately frozen expected output per case, maintained independently from evaluator logic.
- schemas/inputs-v1.schema.json: Draft 2020-12 input schema.
- schemas/oracle-v1.schema.json: Draft 2020-12 oracle schema.
- src/lib.rs: typed input validation and pure disposition evaluator with unit tests.
- src/main.rs: deterministic CLI and SHA-256-bound result receipt.

## Run

From the repository root with Rust stable and access to the declared crates:

~~~sh
cargo test --manifest-path tools/water-repair-closure-001/Cargo.toml
cargo run --manifest-path tools/water-repair-closure-001/Cargo.toml -- \
  --inputs tools/water-repair-closure-001/fixtures/inputs-v1.json \
  --oracle tools/water-repair-closure-001/fixtures/oracle-v1.json \
  --receipt target/water-repair-closure-001/receipt.json \
  --implementation-sha <40-hex-git-sha>
~~~

The CLI compares every computed result against the independent oracle. Mismatches yield a non-zero exit and a receipt marked Failed. A 40-character lowercase exact revision yields PassedExactHead only when every case matches. An unpinned local run is labelled PassedLocalOracle rather than masquerading as exact-head evidence.

The receipt binds the raw input and oracle bytes with SHA-256. That proves which bytes were evaluated, not that an assertion or any source observation is true. The exact implementation SHA is supplied by the invoking environment; it is trustworthy only when captured by the actual execution wrapper or CI.

## Frozen behavior

The evaluator enforces ordering-sensitive safety semantics: content conflicts and privacy violations reject; absent authorization rejects while expired/revoked/out-of-scope authorization blocks; asset mismatch blocks; missing/simulated observations remain incomplete; stale/unqualified/wide-uncertainty measurements remain indeterminate; missing quality clearance blocks a required quality profile; part/media qualification, commissioning, common-mode redundancy, residual disposition and service outcomes are independently gated.

The oracle is intentionally checked by tests but is never read by evaluate_case. Do not add expected dispositions to each input fixture. New cases require an independently reviewed input plus oracle row and an explicit schema/corpus version decision.

## Known limits and next gate

- The JSON Schemas are separate from Rust semantic validation; add a standards-compliant Draft 2020-12 validation step in the independent workflow.
- The current profile-specific rules are only a synthetic semantic reference. Have domain reviewers examine rule precedence and reason codes before treating any result as suitable for real work.
- Calibration currentness, evidence revocation/supersession, permission semantics, material provenance, and model/run identities are not a production trust system here.
- Do not call the corpus passed until the exact revision workflow receipt is generated and independently inspected. A queued or empty check list is not a pass.

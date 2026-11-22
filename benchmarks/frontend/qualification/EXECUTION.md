# SFEB qualification execution protocol

## Purpose

This report format closes the gap between a mutation campaign definition and
a defensible qualification decision. It makes each case a **paired counterfactual**:
the baseline and intervention run under the same fixture digest and environment
manifest, with separate source revisions, run IDs and evidence receipts.

The report adjudicator verifies evidence bytes and receipt bindings before
calculating mutation kill rates. A claimed rejection is credited only when an
expected oracle ID is present in the candidate's observed-oracle list. A build
failure, timeout, agent assertion or unrelated rejection is not a killed mutant.

## Execution report

Input schema: `qualification-report-v1.schema.json`  
Output schema: `qualification-summary-v1.schema.json`

The report contains all 30 case results:

- 20 mutation pairs (baseline accepted, candidate expected to be rejected);
- 5 positive-control pairs (both accepted, semantic output unchanged);
- 5 metamorphic pairs (both accepted, semantic output unchanged and the named
  metamorphic oracle is recorded).

Every baseline/candidate run includes:
- source commit, fixture SHA-256 and environment-manifest SHA-256;
- a serialized `sfeb-frontend-evidence/v1` receipt plus its exact byte digest;
- actual screenshot, accessibility and layout artifact paths, lengths and SHA-256s;
- disposition and observed oracle/invariant identifiers;
- a canonical semantic-outcome SHA-256 used for equivalence checks.

Artifact paths are relative to `artifact_root`, which itself must resolve below
the report directory. Symlinks and path traversal are rejected by canonical path
checks. The adjudicator reads the files and recomputes the hashes; it does not
trust digests supplied in JSON.

## Adjudication

Run from repository root:

```bash
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- \
  --qualification-report \
  benchmarks/frontend/qualification/qualification-report-v1.schema.json \
  benchmarks/frontend/qualification/mutation-campaign-v1.json \
  path/to/qualification-report.json
```

The report's artifact root is resolved relative to the report file's directory.
The command emits a JSON summary, validates it against
`qualification-summary-v1.schema.json`, and reports per-category, per-severity
and overall mutation kill rates with 95% Wilson intervals, positive-control
false rejections, metamorphic equivalence and hard-gate decisions.

The adjudicator validates the campaign's raw-file SHA-256, campaign ID, full case
coverage, paired fixture/environment identity, receipt identity and receipt
artifact digests. It rejects duplicate, missing or unknown cases and any
baseline that fails to reach the accepted control disposition.

## Qualification rules

Targets are defined by the campaign and are immutable in the validator:

- critical mutation kill rate: 100%;
- security mutation kill rate: 100%;
- evidence-integrity mutation kill rate: 100%;
- high-severity mutation kill rate: at least 95%;
- overall mutation kill rate: at least 90%;
- positive-control false-reject rate: 0%;
- all metamorphic relations must hold.

A report can be structurally valid yet fail qualification. It must never be
promoted based on a partial run. Wilson intervals are included because small
campaigns have weak statistical precision: 20/20 successful mutants is a useful
qualification gate, not proof of a 100% real-world detection rate.

## Important limits

This is an **adjudicator**, not the browser mutation executor. It verifies
captured evidence and adjudicates the complete case matrix; the isolated runner
must still safely create each mutation, launch Chromium, replay the journey and
produce a report. The current campaign includes browser cases whose required
observations cannot be inferred from static HTML/CSS alone.

A qualification report is valid only for its exact campaign bytes, source
revision and environment manifest. Changing any of those creates a new
qualification run.


## Adjudicator trust boundary

The adjudicator recomputes the raw SHA-256 of the campaign, receipt files and
all three captured artifacts. It verifies the receipt binds the exact run,
task, fixture identifier, source commit and environment digest; it also checks
that PNG pixel dimensions, accessibility JSON and layout viewport agree with
the receipt. Artifact paths must remain under the report's artifact root after
canonical path resolution.

It does **not** cryptographically authenticate who produced the JSON report,
the disposition or observed-oracle identifiers. A self-authored report could
still lie about which oracle fired while pointing to valid browser artifacts.
Before using qualification results for external claims, the executor must run
inside a trusted isolated workflow and emit a verifiable signed attestation
(e.g. a workflow identity-bound artifact provenance statement). The verifier
must validate that attestation against an explicit trust root. Until that
signature-verification layer and the real browser mutation executor exist,
the report is a local integrity/adjudication format, not a third-party proof.

## New adjudicator command

```bash
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- \
  --qualification-report \
  benchmarks/frontend/qualification/qualification-report-v1.schema.json \
  benchmarks/frontend/qualification/mutation-campaign-v1.json \
  path/to/qualification-report.json
```

The command prints a JSON summary. It returns a non-zero exit code when the
report is structurally invalid or when any qualification hard gate fails.


## Reference fixture self-audit

Before executing the browser campaign, `contract-check --task` now also checks
the static reference markup/CSS subset described in the benchmark README. It
rejects a missing required heading, changed responsive metric breakpoint,
missing skip-link focus treatment, missing named regions, wrong metric-card
count and external CSS asset imports. These are source-fixture consistency
checks only; they do not establish the behavior of a generated app in Chromium.

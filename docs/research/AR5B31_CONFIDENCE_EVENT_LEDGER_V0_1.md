# AR-5B31 — Compositional Confidence-Event Ledger v0.1

**Status:** implementation draft; Rust tests authored but not executed.  
**Tracking:** [Issue #10069](https://github.com/Luminous-Dynamics/symthaea/issues/10069)  
**Module:** **crates/domains/symthaea-architecture-engineering/src/confidence_event_ledger.rs**  
**Branch:** **research/ar5b31-confidence-event-ledger-v0.1-20261102**  
**Dependency:** AR-5B30 direct functional transport / PR #10061.

## 1. Paradigm shift: statistical guarantees as a composable evidence artifact

Decision-confidence and transport modules should not each invent their own alpha-composition helper. The shared rule is simple, but the source events and their contracts must be explicit.

AR-5B31 models each null-side confidence failure as a typed event:
- stable event ID and registered role;
- failure probability;
- source evidence receipt SHA-256;
- functional/estimand contract SHA-256;
- target design SHA-256;
- declaration that its evidence stream is independent of the confirmatory outcome stream.

The event catalogue is frozen, sorted by event ID, hashed, and replayed. Duplicate IDs, wrong target design, missing source/contract hashes, confirmation-dependent evidence, malformed probabilities, and over-budget ledgers fail closed.

The module is deliberately scoped to the events used to qualify a top-level null Type-I claim. It is not a generic registry for every confidence interval or every safety/false-safe budget. Distinct statistical claims must retain distinct failure-budget semantics.

## 2. Mathematical composition

Let the frozen null-side confidence event family be \(E_1,\ldots,E_J\), with:

\[
\Pr(E_j^c)\le\delta_j.
\]

No independence among the \(E_j\) is assumed. By the union bound:

\[
\Pr\left(\bigcap_{j=1}^J E_j\right)
\ge 1-\sum_{j=1}^J\delta_j
=1-\delta_{\mathrm{total}}.
\]

The ledger only qualifies a nominal Type-I level \(\alpha\) when:

\[
\delta_{\mathrm{total}}<\alpha.
\]

If the confirmatory test is valid at conditional level \(\alpha_c\) whenever all registered evidence events hold, then:

\[
\Pr(\text{reject under }H_0)
\le
\delta_{\mathrm{total}}
+(1-\delta_{\mathrm{total}})\alpha_c.
\]

Solving for the largest permitted conditional level gives:

\[
\alpha_c=
\frac{\alpha-\delta_{\mathrm{total}}}
{1-\delta_{\mathrm{total}}}.
\]

The evidence-stream independence declarations are separate from independence among failure events. The former supports conditioning on the qualification evidence before confirmatory testing; the latter is not needed because the ledger uses a union bound.

## 3. Request and report

### Request

The request requires:
- non-empty ledger ID;
- nominal alpha in \((0,1)\);
- common confirmatory target-design SHA-256;
- one to 10,000 null-side failure events;
- sorted unique event IDs;
- event role limited to calibration-null confidence, calibration-to-target transport, or another null-side confidence event;
- finite event delta in \([0,1)\);
- valid source receipt, contract and target-design digests;
- event target design identical to the ledger target design;
- every evidence stream declared independent of confirmation;
- total delta strictly below nominal alpha.

The constructor sorts by event ID. Validation requires the stored request itself to be in strict canonical order.

A zero-delta event is accepted only as an explicitly receipt-bound exact event; it does not waive source or contract provenance.

### Report

The report contains:
- request and event-roster SHA-256 digests;
- target design and nominal alpha;
- the complete ordered event roster;
- event count;
- reconstructed total failure probability;
- derived conditional alpha;
- canonical report digest.

The verifier reconstructs the entire report from the original request. It does not trust stored totals, event counts, or caller-supplied digests.

## 4. Deliberate boundaries

- This is a union-bound ledger; it does not infer or exploit independence among failure events.
- It does require every underlying evidence stream to be independent of the confirmatory stream for this conditional-alpha construction.
- The ledger cannot prove that source data, IID declarations, sampling assumptions, statistical methods or contract definitions are true. It binds the corresponding receipt and assertion.
- The role enumeration prevents silently mixing a guardrail false-safe budget with experiment null Type-I alpha.
- A valid hash is not a statistical coverage proof.
- The ledger does not authenticate who has authority to declare the mandatory functional roster complete.

## 5. Tests authored

Eight Rust tests are authored:

1. Three-event union bound and conditional-alpha analytical fixture.
2. Canonical event order and deterministic replay.
3. Duplicate event ID rejection.
4. Cross-target design and confirmation-dependent evidence rejection.
5. Empty roster, invalid probability and over-budget ledger rejection.
6. Receipt-bound zero failure event.
7. Tampered event role, roster, report digest and late-added event rejection.
8. Independent reconstruction of totals, target identity and roster digest.

**Tests are authored, not run.** No cargo test, cargo check, rustfmt, Clippy, CI, or performance result is claimed.

## 6. Integration sequence

The shared API is the next common primitive for:
1. AR-5B23 decision-functional null-confidence reports;
2. AR-5B27 full-distribution transport;
3. AR-5B30 direct functional transport;
4. AR-5B26 conditional-alpha constrained optimization.

Each producer should emit a typed event with its source receipt, contract, target design and failure probability. The ledger should be the sole source of the top-level conditional alpha. Future adapters should remove their private duplicate formulas after migration, rather than run both composers as if they were independent evidence.

## 7. Example command

From the repository root:

~~~bash
cargo test --locked -p symthaea-architecture-engineering confidence_event_ledger
~~~

Run against the exact PR head when a compatible Rust toolchain is available.

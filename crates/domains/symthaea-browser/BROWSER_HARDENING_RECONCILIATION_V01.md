# Browser Hardening Reconciliation v0.1

Date: 2026-09-01

## Why this reconciliation exists

The repository contains `symthaea-browser-hardening-verification-13-18.md`, which records an authored and replay-verified Patch Set 13–18 series from July 2026. The verification note explicitly states that Cargo compilation, formatting, Clippy, unit tests, and the real-Chromium hostile-browser lane were not performed. The authored Git commits referenced by that note are not present in the current public repository history, and their hardened executor semantics are not present on `main`.

Current `main` before this branch still allowed direct `BrowserExecutor::execute()` dispatch of `Click` and `Type`, retained the full `BrowserAction` and raw `ActionOutput` inside `ActionReceipt`, and had no executor-session action budget or consecutive-failure circuit breaker.

This branch reconciles the highest-value Patch Set 15–18 semantics directly onto current `main` rather than claiming that the historical authored tree has been cargo-qualified.

## Reconciled behavior

### Exact consequential-action approval

`Click` and `Type` cannot execute through the compatibility `execute()` / `execute_with_output()` path. They require `execute_proposal()` plus a `BrowserApproval` matching:

- the current process-local executor session UUID;
- the exact domain-separated action digest;
- the consequence class; and
- an explicit expiry time.

`BrowserApproval` is intentionally documented as a process-local semantic approval object, **not** a cryptographic credential. Xenia / `symthaea-authority` integration is the future authentication layer.

### Bounded autonomy

`BrowserRuntimeLimits` bounds:

- total admitted actions;
- mutating admitted actions; and
- consecutive execution failures.

Budget is reserved before dispatch. A latched failure circuit denies further admitted work once the configured failure threshold is reached.

### Privacy-minimized durable evidence

`ActionReceipt` no longer embeds the original `BrowserAction` or raw `ActionOutput`. It retains:

- executor-session identity;
- payload-free semantic action kind;
- exact action digest;
- consequence class;
- policy/runtime outcome;
- output digest and byte length;
- previous and current trace-chain hashes; and
- elapsed time.

Raw extracted text and screenshot bytes are returned separately through `ActionExecution.output`.

### Chained execution trace

Every proposal, including denied proposals, advances a domain-separated BLAKE3 trace chain over:

- previous trace hash;
- exact action digest;
- outcome class; and
- optional output digest.

The chain therefore detects receipt deletion/reordering/substitution when the expected head is retained, without storing typed text, page contents, screenshot bytes, URL credentials, or query payloads in the receipt itself.

## Deliberately deferred from the historical 13–18 design

This first reconciliation does **not** yet claim:

- cryptographically authenticated approvals;
- CDP-session-bound observation/element references;
- redirect-time DNS pinning / DNS-rebinding prevention;
- same-host/same-origin redirect confinement before page effects occur;
- crash-durable budget persistence;
- deterministic checkpoint/recovery across process restart; or
- real-Chromium hostile-lab qualification.

Those should land as separate reviewable tranches rather than being implied by this PR.

## Required qualification gates

```bash
cargo fmt --check -p symthaea-browser
cargo clippy -p symthaea-browser --all-targets --all-features -- -D warnings
cargo test -p symthaea-browser --all-targets
```

The historical ignored hostile-browser lane should only be cited once a real Chromium runner executes it successfully against this reconciled implementation.

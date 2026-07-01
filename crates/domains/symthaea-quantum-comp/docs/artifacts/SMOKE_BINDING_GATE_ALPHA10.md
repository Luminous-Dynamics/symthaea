# Smoke Binding Gate Artifact

Crate: `symthaea-quantum-comp`

Version: `0.1.0-alpha.10`

Command:

```text
cargo run -p symthaea-quantum-comp --bin symthaea-quantum-comp -- gate smoke-binding
```

Output:

```text
release_gate_status=Warn caveat=local release gate only; does not imply peer review, hardware validation, quantum advantage, or Mycelix attestation
Warn: trials-low — trial count is low; do not report as a benchmark
Pass: boundary-conservative — claim boundary remains within conservative experimental scope
Warn: low-trial-count — trial count is low; report as smoke test rather than benchmark
Warn: fixture-intent-recorded — fixture smoke-binding has intent Smoke; preserve its caveat
Warn: smoke-replay-only — only smoke replay commands were attached; use local-research before publishing
```

Interpretation:

This artifact is a smoke gate reference only. It confirms that the local release
gate wiring, fixture lookup, preflight, audit, and replay caveats are connected.
It is not benchmark evidence, hardware validation, quantum advantage evidence,
or a consciousness claim.

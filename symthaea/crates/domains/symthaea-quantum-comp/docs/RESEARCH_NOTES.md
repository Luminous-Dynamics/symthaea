# Research Notes

## Purpose

`symthaea-quantum-comp` is a research scaffold for testing quantum-inspired and future quantum-backend hypotheses around hyperdimensional cognition. It is designed to be useful even when every result is negative.

A negative result is valuable if it is reproducible, well-scoped, and honest about assumptions.

## Alpha.3 hypothesis additions

Alpha.3 adds two guardrail-oriented probes.

### Negative control probe

The negative control asks a simple question:

Does recovery behave like HDC binding should behave?

Expected behavior:

- matched key recovery remains high;
- wrong-key recovery trends toward chance;
- random-item similarity trends toward chance;
- the control gap remains visible.

This protects the crate from mistaking deterministic implementation artifacts for meaningful substrate behavior.

### Entanglement proxy probe

The entanglement proxy is not physical entanglement.

It is a classical parity/coherence sketch that gives the project a stable place to test questions like:

- Does explicit pair-coherence metadata change noise behavior?
- Does decoherence-like degradation alter recovery gaps differently from bit-flip noise?
- Do topology proxies shift as coherence collapses?
- What should later QASM or backend adapters measure?

The proxy is intentionally conservative. It creates experiment language without claiming hardware results.

## Claim boundaries

Every serious report should carry a claim boundary:

- implementation check;
- local simulation;
- circuit export only;
- external backend observation.

A local simulation result must not be promoted into a hardware claim.

A circuit export must not be promoted into an executed quantum result.

An external backend result should include backend, transpilation, shots, noise model, date, calibration metadata, and reproducibility artifacts.

## Near-term roadmap

Alpha.4 should focus on one of:

1. a proper experiment manifest file format;
2. CSV/JSON exports behind optional features;
3. CPU/GPU parity hooks for phase-HDC;
4. beta-1 proxy comparison across noise/decoherence schedules;
5. Python notebook interop without making Python mandatory.

## Non-claims

This crate does not claim:

- quantum consciousness;
- quantum advantage;
- physical quantum state preparation;
- validated quantum backend execution;
- production cryptography;
- medical, legal, or safety relevance.

## Alpha.4 reporting discipline

Alpha.4 adds replicated comparison and robustness summaries. These features are intentionally modest. They help researchers see whether a result is stable across deterministic seed replicates and how quickly a method degrades across a noise curve.

The approximate confidence intervals in this crate are convenience summaries, not publication-grade statistical guarantees. They use a simple normal approximation around the sample mean. For papers or hardware claims, export the raw reports and analyze them with a full statistics environment.

Alpha.4 still does not execute physical quantum hardware. The entanglement proxy remains a classical parity/coherence sketch. Any hardware observation must be marked as an external backend observation and accompanied by backend metadata, calibration context, and execution receipts.

## Alpha.5 notes: provenance and reporting

Alpha.5 adds local reproducibility metadata, CSV/Markdown export helpers, and conservative audit helpers. These additions are meant to make the crate easier to use in research notes without increasing the strength of the scientific claims.

Important distinction:

- `RunEnvironment` and `ReproducibilityRecord` are lab-note conveniences.
- They are not cryptographic receipts.
- Mycelix or a real digest/signature layer should be used for artifact commitments.

Alpha.5 also adds `audit_binding_probe`, `audit_negative_control`, and `audit_robustness`. These functions are local guardrails only. They are not peer review and do not make a result publishable.

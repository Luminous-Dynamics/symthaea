# MON-019 — Causal Input-to-Effect Provenance

**Status:** research contract, synthetic adversarial scenario seed corpus, and semantic prototype pending.

MON-019 extends MON from action/effect evidence toward end-to-end causal assurance. It composes upstream source evidence, agent/tool invocations, transformations, authority, dispatch, destination commit, independent observation and later corrections into a verifiable causal graph.

## Research question

Can an independent evaluator reconstruct the declared causal dependencies of a consequential autonomous action, verify the exact evidence bytes used at each boundary, detect omissions or substitutions within an independently captured boundary, and preserve the distinction between authentic, accurate, complete and causally relevant evidence?

## Files

- [CAUSAL-PROVENANCE-CONTRACT-v1.md](./CAUSAL-PROVENANCE-CONTRACT-v1.md) — architecture, graph model, claim semantics, completeness, threat model and benchmarks.
- [SCENARIOS-v1.json](./SCENARIOS-v1.json) — synthetic known-answer scenario seed corpus.
- [SCENARIOS-v1.schema.json](./SCENARIOS-v1.schema.json) — strict JSON Schema.
- [causal-graph/](./causal-graph/) — standard-library Go graph-structure and bounded-manifest reference.

## Standards and prior art

- [W3C PROV-DM](https://www.w3.org/TR/prov-dm/) and [PROV Constraints](https://www.w3.org/TR/prov-constraints/).
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) and [OpenTelemetry](https://opentelemetry.io/docs/).
- [OpenLineage](https://openlineage.io/).
- [in-toto Attestation Framework](https://github.com/in-toto/attestation).
- [SLSA provenance](https://slsa.dev/spec/v1.1/).
- [C2PA Content Credentials](https://spec.c2pa.org/).
- [SCITT RFC 9943](https://www.rfc-editor.org/rfc/rfc9943).
- [Agent Action Capsule -06](https://datatracker.ietf.org/doc/html/draft-mih-scitt-agent-action-capsule-06).

## Local reference preflight

The `causal-graph/` package has 29 named adversarial tests and one fuzz target. Local draft-package checks: race tests ×5, `go vet`, statement coverage 90.8%, and a 2-second digest-profile isolation fuzz run with 54,635 executions. These were run on a local reconstructed/gofmt'd copy of the source; the remote GitHub branch was not cloned/replayed in this environment, so this is not an exact-head hosted or checkout qualification.

## Claim ceiling

This is a research contract and synthetic corpus, not a production provenance ledger or end-to-end instrumentation deployment. Provenance does not prove truth. A signature does not prove source accuracy. A trace ID does not authenticate a causal edge. A complete event capture is complete only within its declared independently observed boundary. No production, conformance, independent field evaluation, or SOTA-superiority claim is made.

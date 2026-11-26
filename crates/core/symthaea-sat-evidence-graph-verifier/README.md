# SAT Evidence Graph Verifier

A read-only Rust semantic verifier for SAT benchmark-augmentation evidence graphs.

This crate is deliberately narrower than a full SAT qualifier. It verifies:
- strict restricted-JSON parsing (duplicate keys, floats and unsafe integers rejected);
- unique observation and claim identities;
- evidence-reference closure;
- claim-edge endpoint integrity;
- agreement between each claim's \`depends_on\` set and its typed \`depends_on\` edges;
- acyclicity of the dependency subgraph;
- conservative admissibility requirements for claims marked \`supported\`;
- deterministic graph normalization and the domain-separated evidence-graph root.

It does **not** perform complete JSON Schema Draft 2020-12 validation, verify signatures, fetch evidence artifacts, authenticate issuers, prove witness independence, validate external sink truth, or qualify a system. Those remain separate controls.

## Commands

From the Symthaea workspace:

    cargo test -p symthaea-sat-evidence-graph-verifier
    cargo clippy -p symthaea-sat-evidence-graph-verifier --all-targets -- -D warnings
    cargo run -p symthaea-sat-evidence-graph-verifier --bin sat-evidence-graph-verify -- <augmentation.json>

Compute the root for an input whose root is still a placeholder:

    cargo run -p symthaea-sat-evidence-graph-verifier --bin sat-evidence-graph-verify -- --print-root <augmentation.json>

The root is:

    SHA-256(UTF8("SAT-EVIDENCE-GRAPH-v0.1\n") || canonical_json(graph_payload))

where \`graph_payload\` contains exactly:

    {schema_version, augmentation_id, native_benchmark, subject, run_context, observations, claims, claim_edges}

Observations and claims are sorted by their IDs using UTF-16 code-unit ordering. Claim edges are sorted by \`(from_claim_id, to_claim_id, relation, rule_ref)\`. The claim \`evidence_refs\`, \`depends_on\`, and \`invalidation_triggers\` sets are sorted and must be unique. Object keys use UTF-16 ordering. Only safe integer numbers are accepted; strings are not Unicode-normalized. This is the versioned SAT restricted canonicalization profile, not a claim of full RFC 8785 conformance.

## Semantic rules

1. Every observation ID and claim ID is unique.
2. Every claim evidence reference resolves to exactly one observation in the same augmentation.
3. Every edge endpoint resolves to a claim.
4. Each \`depends_on\` declaration must have exactly one corresponding typed dependency edge, and each typed dependency edge must appear in the source claim's dependency set.
5. The dependency subgraph must be acyclic.
6. A claim marked \`supported\` must have current validity and evidence. Evidence marked invalid, unknown-authenticity, unverified-independence or not-satisfied-independence cannot silently support it.
7. A critical supported claim requires every referenced observation to have verified authenticity and satisfied independence.
8. Effect-correspondence claims require an \`external_effect_readback\` from an \`independent_sink\`, with verified authenticity and satisfied independence.
9. A supported claim cannot depend on a claim that is not both supported and current.

These checks are conservative graph semantics, not universal truth conditions. Adapter-specific policies may impose stronger requirements. The verifier reports \`qualification_assessed=false\` unconditionally.

## Exit codes

- \`0\`: semantic checks pass and the recorded root matches.
- \`2\`: malformed input or semantic failure.
- \`3\`: semantic checks pass, but the recorded root does not match the computed root.
- \`64\`: command usage error.
- \`66\`: input file cannot be read.

\`--print-root\` is a canonicalization aid only. It does not assert semantic validity or qualification.

## Evidence boundary

A passing result proves only that the submitted graph is internally well-formed under this implementation and that its recorded root matches its normalized graph payload. It does not prove that an observation is truthful, a source is honest, two witnesses are independent, a sink applied an effect, or a system is safe. The caller must first validate the augmentation against its pinned schema and separately authenticate and evaluate referenced evidence.

The AgentDojo-shaped augmentation example in the SAT benchmark-bridge research increment intentionally uses placeholder evidence and an uncomputed graph root; \`--print-root\` is available to derive its root after a semantic evaluator has accepted its graph structure.

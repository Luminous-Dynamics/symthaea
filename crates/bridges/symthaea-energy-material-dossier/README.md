# Symthaea Energy-Material Evidence Dossier

This crate assembles independent Tier-1 material evidence into one candidate dossier without erasing identity assumptions.

It does not know or trust adapter-specific receipt types. Each contribution supplies:

- one canonical Tier-1 evidence dimension;
- one generic `symthaea-discovery::Prediction`;
- the Symthaea candidate ID;
- exact source-receipt SHA-256;
- the identity assertion used to connect that receipt to the candidate.

## Partial evidence stays partial

A dossier may contain any subset of the seven Tier-1 dimensions. Missing contributions are not filled with zeros, priors, heuristics or favorable defaults.

The assembled predictions are passed through `symthaea-energy-material-screening`; therefore an incomplete dossier remains `Incomplete` and produces no generic evaluation until the screening policy's evidence requirements are met.

## Identity assertions

Three assertion modes are available:

- `internal_candidate`: evidence was generated directly from the Symthaea candidate;
- `exact_external_id`: external subject ID exactly equals the candidate ID;
- `explicit_mapping`: a differently named external subject is caller-mapped to the candidate with a mandatory review note.

Every contribution references one assertion. The assertion and contribution must bind the exact same source-receipt SHA-256.

This prevents a stability receipt for material A from being paired with the identity note for material B merely because their metric names match.

## Policy contract enforcement

Before assembly, each contribution must match the exact metric/unit declared for its evidence dimension by the supplied Tier-1 screening policy.

Duplicate dimensions fail closed. Candidate IDs must agree across dossier, assertions and contributions.

## Determinism

Identity assertions are sorted by assertion ID and contributions by canonical evidence dimension before the dossier is emitted. Caller input order therefore does not change dossier identity.

The dossier receives a domain-separated SHA-256.

## Trust boundary

The dossier digest is a content identity, not independent proof that upstream source documents, mappings or timestamps are authentic. Those claims remain in the underlying evidence receipts/source digests and can later be strengthened with signed or externally registered evidence.

The dossier does not grant synthesis, procurement, fabrication, certification or deployment authority.

# Symthaea DEVART/VART Benchmark Firewall v1

Status: development/evaluation contamination-control contract.

## Objective

Development feedback and hidden scientific evaluation MUST be different information domains.

- **DEVART** is visible, reusable, debuggable, and explicitly non-confirmatory.
- **VART** is held out, prospectively committed, and unavailable as optimization feedback before the corresponding campaign is sealed.

A VART campaign is invalid for claim admission if hidden benchmark identities, exact fixture artifacts, exact seeds, expected answers, trap labels, or target solutions were exposed to the subject/development process before the frozen reveal point.

## Four-party boundary

The preferred execution model separates:

1. `subject` — Symthaea under evaluation;
2. `benchmark_custodian` — creates/holds hidden VART material and publishes commitments;
3. `measurement_instrument` — executes and verifies the frozen protocol;
4. `analyst` — opens sealed outcomes under the frozen analysis contract.

One person or organization may fill multiple roles during early research, but the artifacts and authority boundaries remain distinct.

## Public pre-launch VART surface

Before launch, the repository may contain only the VART **commitment manifest**, not hidden plaintext benchmark material. The public manifest may expose:

- benchmark/campaign identity;
- number of hidden clusters/families;
- SHA-256 commitments to fixture artifacts and seeds;
- generator-policy/version digest;
- scoring-contract digest;
- reveal policy;
- custodian identity/receipt digest;
- domain-separation tag.

It MUST NOT expose plaintext hidden fixture IDs, exact seeds, expected solutions, target defects, trap labels, or hidden generator parameters.

## Contamination checks

A firewall verifier MUST reject when:

- any DEVART fixture commitment equals a hidden VART fixture commitment;
- any DEVART seed commitment equals a hidden VART seed commitment;
- a VART plaintext path resolves inside the subject/instrument repository before reveal;
- hidden VART material is marked revealed before the prospective reveal condition;
- a campaign reuses a VART commitment already used as development feedback;
- subject-visible configuration contains benchmark secret fields;
- benchmark and development domains omit explicit cryptographic domain separation.

## Reveal

Reveal occurs only according to the preregistered campaign policy. A post-campaign reveal receipt binds the plaintext artifacts to the pre-launch commitments.

After reveal, that benchmark family is **spent** for future confirmatory evaluation and becomes historical/DEVART material unless a new independently hidden family is generated.

## Claim boundary

Passing the firewall proves only contamination control for the declared artifacts. It does not prove absence of all semantic similarity, benchmark memorization from external sources, or generalization. Those require separate evaluation design and independent replication.
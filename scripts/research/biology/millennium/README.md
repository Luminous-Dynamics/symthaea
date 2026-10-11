# Millennium Biology 001C — independent contract oracle

This directory contains the minimal, stdlib-only contract oracle for the Millennium Biology research boundary.

The oracle validates **contract and provenance integrity only**. It does not evaluate biological correctness, model quality, experimental results, or challenge completion.

## Run

From the repository root:

```bash
python3 scripts/research/biology/millennium/contract_oracle.py \
  --mapping docs/research/biology/millennium/claim-mapping-2026-09-28.json \
  --event-contract docs/research/biology/millennium/evidence-event-contract-1.0.0.json \
  --mutation-suite
```

The mutation suite intentionally corrupts eight boundary conditions and requires each corruption to be rejected with a structured reason code.

The eighth fixture models a prospective prediction being rewritten after commitment: immutable event identity/payload changes are rejected; supersession must be represented by a new event.

No network access, biology runtime, FEP/HDC/causal-reasoning dependency, or wet-lab tooling is required.

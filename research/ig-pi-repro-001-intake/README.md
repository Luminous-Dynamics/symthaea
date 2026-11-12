# IG-PLURALISTIC-IGNORANCE-REPRO-001 — metadata intake v0.1

**Status:** metadata-only / blocked; external model results not reproduced  
**Owning issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11689  
**Paper:** https://arxiv.org/abs/2608.02758  
**Upstream code:** https://github.com/YashwanthYS/pluralistic-ignorance  
**Upstream dataset:** https://huggingface.co/datasets/yashwanthys/pluralistic-ignorance

## Purpose

This is a rights-aware, fail-closed inventory of an external population-level pluralistic-ignorance benchmark. It does not import upstream scenario payloads, participant records, model weights or result files. The manifest pins exact Git blob identities and records the reproducibility, scoring, schema and license blockers observed on upstream commit `b4d79c8229eb5d7bedfea824a9988115cfe53ef6`.

## What was audited

- 100 scenarios, 10 domains and five authority levels as described in the paper.
- A controlled 20-agent population with 16 prompt-assigned dissenters and four prompt-assigned supporters.
- Two baseline discussion rounds and a two-round norm-entrepreneur phase.
- Python's built-in salted string `hash()` used in random-seed derivation.
- A keyword scorer where unmatched text returns 0 and the conformity rule counts scores at or above zero.
- A published scorer-validation claim (100 cases / 82% human agreement) alongside a committed validation artifact (36 rows / 55.56% keyword-versus-judge agreement); reconciliation is unresolved.
- Hugging Face JSON dataset viewer schema-cast failure across scenario and result artifacts.
- README's MIT license assertion with no LICENSE file in the inspected repository tree and no detected repository license metadata.

## Reproduce local metadata qualification

From the Symthaea repository root:

```bash
cargo test --manifest-path research/ig-pi-repro-001-intake/qualifier/Cargo.toml

cargo run --manifest-path research/ig-pi-repro-001-intake/qualifier/Cargo.toml -- \
  research/ig-pi-repro-001-intake/PI_BASELINE_INTAKE_V0_1.json \
  research/ig-pi-repro-001-intake/PI_BASELINE_INTAKE_V0_1.schema.json
```

The CLI exits successfully only if the **metadata-only blocked state** remains internally consistent. Its success is not a baseline admission, rights decision, model reproduction or endorsement of the upstream findings.

## Next gated steps

1. Resolve exact code and data licensing with the upstream authors; paper CC BY 4.0 does not license code/data.
2. Separate scenario-input and experiment-result datasets into distinct schemas and snapshots.
3. Implement a deterministic seed derivation in a separately named replication harness; preserve the upstream behavior as a different condition.
4. Add scorer fixtures for explicit neutral, ambiguous, unclassifiable and error/missing responses; use independent blinded annotations.
5. Reconcile the published human-validation claim with the committed keyword-versus-judge artifact.
6. Reproduce pinned models/prompts/conditions, including multiple private-stance compositions and exact token/latency/cost receipts.
7. Compose public/private results with Symthaea #11320, perceived peer-private beliefs #11682, and Mycelix #5999/#6011/#6023.

## Claim ceiling

This slice establishes only that the exact source snapshot and its known blockers are recorded in a machine-checkable manifest. It does not establish scorer validity, faithful reproduction, human pluralistic ignorance, human population prevalence, real-world social prediction, democratic legitimacy, rights to redistribute upstream artifacts or production readiness.

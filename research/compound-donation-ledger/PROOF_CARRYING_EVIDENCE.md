# Proof-carrying Bernoulli e-process evidence envelopes

## Purpose

The donation ledger verifies how calibrated evidence is reserved and transferred. It does not calculate the upstream evidence process or prove that its inputs obey the declared global filtration. This workstream adds a separate, replayable evidence envelope for a registered Bernoulli likelihood-ratio process.

It creates an inspectable boundary between:

1. raw, precommitted observation batches;
2. exact likelihood-ratio arithmetic;
3. running-max time-uniform adjustment;
4. the integer evidence values consumed by the source-only donation ledger.

This is a verifiable computation contract, not a proof of the real-world probability law.

## 1. Registered process contract

Each candidate has precommitted rational null and betting probabilities \(p_{0,i,t}\) and \(p_{1,i,t}\). The first prototype uses constants

\[
p_0=\tfrac12,\qquad p_1=\tfrac7{10}.
\]

For a binary observation \(Y_{i,t}\), the likelihood-ratio factor is

\[
L_{i,t}=
\begin{cases}
p_1/p_0=7/5,&Y_{i,t}=1,\\
(1-p_1)/(1-p_0)=3/5,&Y_{i,t}=0.
\end{cases}
\]

Under the registered null conditional model,

\[
\mathbb E[L_{i,t}\mid\mathcal F_{t-1}]
=\tfrac12(7/5)+\tfrac12(3/5)=1.
\]

The product \(M_{i,t}=\prod_{s\le t}L_{i,s}\), held at one while dormant, is therefore a nonnegative test martingale **provided** the null conditional law holds relative to the declared global filtration. The writer and independent verifier recompute the factors and product exactly using rational arithmetic.

## 2. Atomic global-round batches

All observations becoming visible at round t are treated as one atomic batch. A candidate's round-t observation is not made available as prior information when computing another candidate's round-t factor.

Each receipt binds:
- process-specification digest;
- global filtration ID;
- round and sequence number;
- previous batch digest;
- precommitted active roster;
- ordered raw observations;
- exact per-candidate LR factors and products;
- running maximum and adjusted integer;
- arrival-policy and source-completeness attestation digests.

The independent verifier rejects missing or duplicated candidates, reordered observations, gaps or rollback in rounds, changed process digests, stale batch hashes, and altered factor/product/adjuster claims.

The roster in this prototype is predictable and precommitted by a deterministic arrival schedule. The receipt can establish consistency with that schedule; it cannot prove that a production data source disclosed every real observation or that the schedule was honestly enforced.

## 3. Time-uniform adjustment

The prototype applies

\[
A(x)=\sqrt{x}-1
\]

to the running maximum of the LR martingale and stores

\[
X_{i,t}
=\frac{\left\lfloor S\left(\sqrt{\max_{s\le t}M_{i,s}}-1\right)\right\rfloor}{S}
\]

for fixed scale \(S=10^6\).

The calibrator obeys

\[
\int_1^\infty \frac{\sqrt{x}-1}{x^2}\,dx=1.
\]

The floor is a pointwise lower approximation to A, so it does not increase the Ville-integral budget. The verifier recomputes the integer square root with exact rational input; it does not trust the writer's adjusted value.

The code does not implement a complete e-BH procedure. It produces adjusted evidence values for the separate ledger and multiple-testing layer.

## 4. Independent replay and negative controls

The committed standalone runner is benchmarks/proof_carrying_evidence.py. It has distinct writer and verifier paths:
- the writer applies the registered LR update and emits immutable round receipts;
- the verifier independently derives the success/failure factor from the registered rational pair, recomputes the product/running maximum/adjuster, validates the roster and hash chain, and compares the complete claim list.

The local rerun of the committed runner logic used 100 traces, 12 candidates and 30 rounds (32,400 evidence updates), rotating among independent, shared-uniform and balanced-negative batches. All 100 traces replayed successfully. Median verifier time was approximately 2.1 ms per 12-candidate, 30-round trace in the local Python environment.

Seven deliberate tamper controls were rejected:
1. observation bit changed with stale digest;
2. adjusted evidence claim changed;
3. previous batch hash changed;
4. active roster omitted a candidate;
5. round sequence rolled back;
6. observation order reversed;
7. process specification digest changed.

This measures a small reference Python verifier. It is not a Rust throughput result, and it is not a benchmark against a production streaming platform.

## 5. Critical trust boundary

The verifier establishes arithmetic and declared-envelope consistency. It does **not** establish:

- that the real observations obey the registered conditional Bernoulli null law;
- that the filtration receipt contains every piece of information available to the data-generating system;
- that a source-completeness attestation is truthful;
- that the source supplying the observations is authentic;
- that a SHA-256 digest chain was signed by an authorized party.

The hash chain detects accidental mutation, inconsistent serialization and stale/tampered receipts when the attacker does not recompute the whole chain. It is not authentication: an attacker who can replace all batches can recompute an unsigned hash chain. Production use needs authenticated source envelopes (e.g. signatures rooted in trusted source keys), timestamp/sequence attestations, and an independently governed completeness policy. Even then, the statistical conditional-null assumption remains a model claim to be validated.

A valid e-process theorem remains conditional on the registered null model and its global filtration. Arithmetic replay is not a substitute for that probability statement.

## 6. Integration with process-valued donation ledger

The next integration contract should make a ledger evidence update refer to a verified evidence receipt:

- candidate ID and round;
- prior evidence-state digest;
- adjusted-evidence integer and scale;
- process-specification digest;
- round-batch digest;
- independent-verifier result reference.

The donation ledger then uses the adjusted evidence value, while retaining its own independent source-cap and escrow verifier. Evidence-process replay and capital-transfer replay remain separate trust domains.

Do not let the donation router inspect same-round observations when deciding routes. Its round-t allocation decisions must be frozen before the evidence batch is revealed.

## 7. Executable artifacts and next gates

- benchmarks/proof_carrying_evidence.py: writer, independent verifier and exact replay.
- benchmarks/test_proof_carrying_evidence.py: regression tests for LR factors, A2 floor, full roster, monotone rounds and tamper controls.
- schemas/proof_carrying_evidence.schema.json: structural receipt schema.
- benchmarks/proof_carrying_evidence_2026-11-02.json: local run manifest.

Before integrating with the Rust ledger:
1. Add schema validation to the Python/Rust boundary and canonicalize the wire format.
2. Sign observation-source envelopes and pre-observation arrival receipts.
3. Make the global filtration and within-round batch semantics machine-readable.
4. Differentially replay evidence outputs in a Rust independent verifier.
5. Add mutation/property tests for rational arithmetic, hash chaining, roster completeness and evidence rollback.
6. Benchmark verification latency and receipt size at 10, 100, 1,000, 10,000 and 1,000,000 candidates.

## Claim boundary

The 100-trace run establishes that this small exact-arithmetic reference prototype and its independent verifier agree on the tested receipts and reject the specified tampering. It does not prove the real-world null model, authenticate source data, establish production completeness, prove SupFDR, or demonstrate a power gain.
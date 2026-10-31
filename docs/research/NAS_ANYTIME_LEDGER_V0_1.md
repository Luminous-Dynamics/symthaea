# NAS anytime confidence family ledger v0.1

Tracking: [NAS-LEDGER-001 / #9464](https://github.com/Luminous-Dynamics/symthaea/issues/9464)

Implementation branch: \`research/nas-confidence-ledger-v0.1\`

## 1. Purpose

PR #9445 adds a bounded-mean confidence receipt for one stream at one look. This ledger adds the protocol that makes a sequence of receipts interpretable: a fixed family manifest, all declared decision streams, immutable stream support and threshold, monotonically derived looks, ordered replicate batches, exact cumulative sample counts, prior-receipt links, and independent replay.

A confidence receipt must not be considered standalone operational authority. The ledger makes the relationship among a series of receipts auditable and makes silent family/stream drift detectable.

## 2. Frozen family manifest

The manifest declares:

- one non-empty family ID and a finite family-wise alpha in (0, 1);
- a complete list of unique stream IDs, strictly sorted before the manifest is hashed;
- one fixed support interval and decision threshold per stream;
- a SHA-256 digest over the canonical serialized manifest.

The family size is derived from the number of declarations. It is not supplied independently by a caller at each look. The per-stream alpha is always family_alpha / declared_stream_count.

The exact manifest bytes or digest must be committed to an external preregistration record before outcomes are observed. The ledger's hash cannot prove wall-clock chronology or that an adversary with write access did not replace the entire manifest and recompute every hash. Use a signed commit, trusted timestamp, immutable object store, or transparency log when external custody matters.

## 3. Append contract

A caller appends a non-empty batch of newly observed independent replicate-level values to one declared stream. The append function:

1. verifies the complete existing ledger before mutation;
2. rejects undeclared stream IDs, empty batches, non-finite/out-of-support values, and size overflow;
3. derives the next one-based look index from the current stream history;
4. rebuilds the exact cumulative sample prefix from prior batches plus the new batch;
5. recomputes the anytime bounded-mean receipt with the fixed family alpha, stream count, support, threshold, and look index;
6. hashes the new batch and extends a prefix chain that is bound to the manifest digest and stream ID;
7. links to the preceding receipt digest and hashes the complete look record;
8. updates the top-level ledger digest.

The caller cannot set a look index, previous receipt hash, family size, threshold, or support interval during append.

## 4. Independent verification

The verifier checks the manifest schema/order/digest and the exact one-to-one correspondence between manifest streams and ledger streams. It then replays every stream:

- look indices must be contiguous and one-based;
- each look must append at least one value;
- cumulative sample counts must exactly match batch lengths;
- batch digests and prefix-chain digests must recompute;
- previous-receipt links must point to the prior look and nowhere else;
- every confidence receipt must recompute from the exact cumulative ordered sample prefix and immutable stream declaration;
- look-record and whole-ledger hashes must match.

The verifier also supports ordinary JSON round-trip. Unknown JSON object fields are rejected by the serde contracts.

Hashes are integrity bindings. They are not signatures, proof of authorship, proof of chronology, or a certificate of sample independence.

## 5. Statistical interpretation

The Hoeffding alpha-spending guarantee remains conditional on independent bounded replicate-level observations. The fixed family alpha is Bonferroni-split across all streams. Within a stream, alpha_t = alpha_stream / (t(t+1)); these allocations sum to alpha_stream over all looks.

For randomized QMC, one complete independently randomized replicate is one observation. Points inside one Halton/QMC replicate are dependent and must not be treated as independent sample units.

The current QMC code uses deterministic SplitMix64-generated random shifts from a fixed seed. It is reproducible, but the hash ledger cannot prove that its pseudo-random randomizations behave like independent uniform shifts. Record a seed manifest and randomization implementation identity, and independently review PRNG assumptions before interpreting the confidence guarantee as operational.

## 6. Reference simulation

A separate Python statistical mirror used 20,000 complete decision families, four preregistered streams per family, 50 looks per stream, and 16 new Bernoulli(0.5) observations at each look. The decision threshold was the true mean 0.5. A false decision was any look on any stream whose interval excluded 0.5.

- Repeated pointwise plug-in normal intervals using 1.96 standard errors: 16,379 / 20,000 families (81.895%; 95% Wilson interval 81.355%–82.423%).
- Hoeffding intervals with 5% family alpha, Bonferroni stream allocation, and the declared alpha-spending schedule: 49 / 20,000 families (0.245%; 95% Wilson interval 0.185%–0.324%).

This deliberately simple boundary simulation illustrates the repeated-look / multiple-stream hazard in a naive pointwise interval. It is not a benchmark of NAS quality, randomized-QMC integration, physical hardware, the Rust implementation, or empirical calibration. The observed false-decision rate below 5% is not a proof of the finite-sample bound; the mathematical guarantee is conditional on the stated independent bounded-observation assumptions.

## 7. Complexity and limitations

Version 0.1 stores the scalar replicate batches in the ledger and replays them to verify receipts. Appending and verifying a long history recomputes cumulative prefixes at each look; the reference implementation therefore has work proportional to the sum of cumulative prefix lengths, not merely the number of new samples. It is intended as a small, auditable reference ledger, not yet a high-throughput distributed evidence store.

Future optimizations should preserve independent replay semantics. Possible directions include an incremental, domain-separated observation hash chain plus a separately verified running-mean accumulator, periodic prefix checkpoints, and a content-addressed batch store. Any optimization must preserve exact reproducibility against the v0.1 reference verifier.

The ledger does not prove:
- external preregistration time or trusted custody;
- independence of replicate observations or PRNG quality;
- correctness/calibration of the posterior model;
- representativeness of architecture/search-space samples;
- physical accuracy of device measurement instrumentation;
- permission to deploy, promote, or prune.

Operational authorization must still consume #9388 qualification capability and #9329/#9104 run-wide pruning-risk evidence. A statistical interval alone must never authorize a high-consequence action.

## 8. Validation and adversarial benchmark plan

**Nineteen authored unit tests** cover manifest uniqueness/order/digest, family and threshold/support immutability, unknown stream rejection, independent stream sequencing, multi-look parent links, exact cumulative sample counts, tampered batches even when the outer batch/ledger hashes are recomputed, look rollback/duplication, missing/reordered streams, broken prefix chains, corrupted receipts, final ledger digest mismatch, JSON round-trip, and rejection of unknown serialized fields.

Before qualification, run:
- exact-head Rust unit and crate integration tests;
- 1, 10, 100, and 1,000 look serialization / append / replay measurements;
- corruption tests that alter one byte in each manifest, batch, receipt, record, and ledger field;
- sequential family-wise coverage campaigns across true means at, above, and below thresholds;
- alpha allocations with varied stream counts and look horizons;
- a comparison of Hoeffding, empirical-Bernstein, and betting confidence sequences on the same independent replicate streams.

## 9. Benchmark programme and industry comparators

For hardware-aware NAS, compare at equal vector-valued budgets: accelerator seconds, target-device occupancy, instrument time, energy, money, storage, CPU time, and operator time. Do not reduce unlike resources to a weighted scalar unless the benchmark explicitly preregisters the exchange-rate policy.

Use these benchmark families:
- NATS-Bench-MCU v2: 15,625 topologies, 14,281 physical nRF5340 profiles, 1,344 flash-infeasible architectures, and the v2 correction overlay. Start with the 6.9 MB publication CSV export; selectively acquire raw traces for audit.
- ConNAS (ICLR 2026): constrained gradient-based architecture search, reporting architecture quality relative to the best feasible benchmark candidate.
- EdgeVolution (Communications Engineering, 2026): end-to-end microcontroller hardware-in-the-loop search and deployment, measuring actual latency and power.
- BoTorch qEHVI/HVKG and well-established evolutionary / multi-fidelity baselines: matched objective sets, constraints, candidate spaces, and measurement budgets.

Primary outcomes are held-out hypervolume and portfolio-class regret, feasible-architecture recall, false-feasible rate, fixed-regret resource cost, measurement count, physical latency/power/energy, flash/SRAM, and correct abstention when evidence is insufficient. Rank correlation is a secondary diagnostic, not the objective.

Proposed, aspirational go/no-go targets:
- at least 95% simultaneous coverage and no more than 5% family-wise false decisions at declared 5% family alpha in synthetic sequential campaigns;
- at least 20% fewer independent measurements than Hoeffding after a variance-adaptive method is added, at the same false-decision rate;
- at least 10% lower all-in vector-resource cost to a fixed held-out Pareto-regret target than the strongest matched-budget baseline;
- no regression in calibration, false-feasibility, or deployment constraint satisfaction.

No target is claimed achieved by the ledger implementation.

## 10. References

- Howard et al., “Time-uniform, nonparametric, nonasymptotic confidence sequences,” Annals of Statistics (2021): https://doi.org/10.1214/20-AOS1991
- Waudby-Smith and Ramdas, “Estimating means of bounded random variables by betting,” JRSS B (2024 issue; first published online 2023): https://doi.org/10.1093/jrsssb/qkad009
- ConNAS, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7c9f851081499ed5c60c51e312d3235f-Abstract-Conference.html
- EdgeVolution, Communications Engineering (2026): https://doi.org/10.1038/s44172-026-00708-2
- NATS-Bench-MCU v2, Zenodo v2.0 (28 July 2026): https://doi.org/10.5281/zenodo.21644470

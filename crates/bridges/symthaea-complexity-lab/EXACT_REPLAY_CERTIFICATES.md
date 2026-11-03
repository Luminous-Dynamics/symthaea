# ETH-022: deterministic replay certificates for exact complex-weighted elimination

**Status:** Q(i) trace producer and independent-checker source committed; the Python test suite was not executed in this session. Rust, Lean, formal correctness and performance qualification remain open.

## 1. Objective

A certificate should describe the exact computation that was performed without storing all intermediate tables in the certificate. A second implementation reconstructs each bucket operation from the original immutable factors and verifies the committed output digest.

Trace format: \`symthaea.qi-elimination-trace/v1\`.

It binds:
- exact finite-domain instance and canonical digest;
- reduced-rational Gaussian coefficient encoding;
- explicit unit factors for variables absent from all input scopes;
- variable elimination order;
- exact bucket factor IDs and joined/output scopes;
- digest of each generated factor;
- final scalar-factor IDs/digests and exact result/digest.

**The digest is not the proof.** The checker recomputes exact multiplication and summation and checks the resulting factor digest. This is deterministic replay, not a succinct proof.

## 2. Prepared finite replay-certificate test plan — execution pending

Committed files:
- reference/qi_elimination_trace.py
- reference/qi_elimination_trace_checker.py
- reference/qi_elimination_trace_results.json

The intended test plan describes 120 Q(i) networks, four elimination orders each, 480 direct-enumeration comparisons, one additional positive fixture and eight certificate mutations. The scripts were not executed in this session; there is currently no measured exact-agreement count and no measured checker acceptance/rejection count.

The trace/checker architecture is prepared in source, but it is not yet a qualified deterministic replay implementation. Hashing, replay correctness, parser behavior and actual certificate sizes remain to be tested.

## 3. Why this is useful

The existing ETH-016 certificate design stores intermediate factor tables and asks a checker to replay them. This ETH-022 prototype explores a lower-certificate-volume alternative:
- the certificate stores factor digests and operation descriptors;
- the checker recomputes each intermediate result;
- the trace size is independent of the number of serialized table entries, although checker work remains proportional to the arithmetic it replays.

The next step is streaming factor generation and hashing. It is important not to overstate memory savings: a verifier may still need to hold or page a generated factor that feeds later elimination steps. A true bounded-memory claim requires an independently measured streaming/disk-backed implementation and exact peak RSS telemetry.

## 4. Prior-art boundary

Certified model counting and knowledge-compilation proofs are established:
- Fichte, Hecher & Roland, *Proofs for Propositional Model Counting*, SAT 2022: https://doi.org/10.4230/LIPIcs.SAT.2022.30
- Bryant et al., *Certified Knowledge Compilation with Application to Formally Verified Model Counting*, JAIR 2025: https://arxiv.org/abs/2501.12906
- Certifying top-down Decision-DNNF compilers: https://doi.org/10.1609/aaai.v35i7.16776
- Certifying projected knowledge compilation: https://doi.org/10.4230/LIPIcs.SAT.2025.8

ETH-022 does not claim that replayable certificates or certified model counting are new. The hypothesis is a common deterministic replay interface spanning exact Q(i) weighted inference, semiring output contracts, projected two-phase quantifier elimination, structural decomposition and algebra-qualified rewrite plans.

## 5. Next qualification steps

1. Port the wire format to Rust without changing canonical semantics.
2. Bind the Q(i) operations to a Lean field/Q-algebra model and prove the local bucket-step/global-result invariant.
3. Extend the typed certificate to Boolean/natural/MaxCount/Pareto/polynomial domains only with domain-specific law proofs.
4. Add projected-phase certificates which prove hidden-variable Boolean existential elimination precedes projected weight accumulation.
5. Implement a chunked/disk-backed factor table reader and checker; measure producer and checker peak RSS.
6. Benchmark against Ganak exact MPQ on identical complex-weighted instances and against MICE/POG-CPOG on matching counting outputs.
7. Optimize verified end-to-end cost, not producer-only arithmetic time.

Acceptance targets:
- 100% exact agreement with direct oracles on frozen tractable instances;
- zero accepted invalid certificates in mutation suites;
- no claim of deterministic proof coverage unless the checker replays the full computation;
- report solve, certificate and check time; peak RSS; factor entries; coefficient bit sizes; and certificate bytes.

An aspirational target is at least 10% lower verified end-to-end cost on a preregistered held-out slice, or a clearly valuable deterministic verification guarantee at acceptable overhead. No performance improvement is currently measured.

## 6. Claim ceiling

This prototype demonstrates finite exact Q(i) replay-certificate checks on small factor networks. It is not a formally verified solver, succinct proof system, bounded-memory checker, competition result or SOTA improvement.

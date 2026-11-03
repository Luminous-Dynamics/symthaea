# ETH Complexity Lab: integrated research and execution roadmap

**Snapshot:** 3 November 2026. This roadmap joins ETH-015 through ETH-024. Drafts are research artifacts, not merged or release-qualified implementations.

## 1. Research thesis

Symthaea should become an exact-computation compiler and assurance stack that:
1. identifies the exact requested output semantics;
2. selects an algebra whose laws support that computation;
3. searches over equivalent rewrites, elimination orders and structural decompositions;
4. produces deterministic execution certificates;
5. has an independent checker replay the computation;
6. measures the complete producer + certificate + verifier cost;
7. records the exact evidence level of every mathematical and performance claim.

The key proposed differentiation is not one new semiring, one SAT heuristic or one certificate format. It is the composition of output-aware algebra selection, algebra-qualified transformations, structural planning, independently replayable exact computation and evidence-governed claims.

Certified model counting, equality saturation, treewidth/hypertree-width algorithms and tensor contraction planning are all established prior art. Any novelty claim must concern a measured, specific combination and must be compared against the strongest compatible systems.

## 2. Research lines and current artifacts

| Line | Core question | Current repository artifact | Qualification state |
|---|---|---|---|
| ETH-015 | Which algebra matches the requested exact output? | src/semiring.rs, src/profile.rs | Rust source exists on research branch; not compiled in this environment |
| ETH-016 | Can bucket variable elimination be replayed and checked? | ELIMINATION_CERTIFICATE.md, reference results | Python reference scope documented; Rust/Lean qualification open |
| ETH-017 | Which order minimizes algebra-specific and certificate cost? | ELIMINATION_ORDER_PLANNING.md, planner references | Finite exhaustive tiny-instance studies; no held-out runtime result |
| ETH-018 | Can equivalent computation structures be rewritten with algebra guards and proof witnesses? | ALGEBRAIC_TRANSFORMATION_DISCOVERY.md, reference oracle/checker | Reference producer/checker source committed; tests not executed in this session; no Rust e-graph or Lean rule proofs |
| ETH-019 | Which structural view exposes tractability beyond primal treewidth? | STRUCTURE_AWARE_PLANNING.md, component witness prototype | Component-splitting producer/checker source committed; tests not executed; no hypertree-width or tensor portfolio |
| ETH-020 | Can exact complex-rational weights be incorporated into the proof-aware architecture? | GAUSSIAN_RATIONAL_AMC.md, Q(i) reference | Q(i) reference source committed but not executed; Ganak already has exact MPQ, so the target is certificate/planner integration |
| ETH-021 | How are hidden witnesses collapsed before distinct projections are counted? | PROJECTED_COUNTING_MIXED_ELIMINATION.md, projected reference | PMC/PWMC-style source committed but not executed; complete official semantics/parser open |
| ETH-022 | Can a compact trace be checked by deterministic replay without serializing all intermediate tables? | EXACT_REPLAY_CERTIFICATES.md, Q(i) trace/checker | Replay producer/checker source committed but not executed; no bounded-memory claim, Rust or Lean proof |
| ETH-023 | Can an input auditor resolve MC/WMC/PMC/PWMC/AMC semantics before choosing an engine? | MC_INPUT_CONTRACT_AUDITOR.md, parser reference | 34 fixture expectations and malformed-input fuzz plan committed but not executed; explicit effective output contract added; official-corpus and Ganak differential tests open |
| ETH-024 | Can every complexity claim be connected to theorem scope, output, encoding, machine and immutable evidence? | CLAIM_EVIDENCE_LEDGER.md, JSON ledger and validator | 4 claims/7 pinned sources; 8 claim-inflation mutations rejected by independent JS policy replay |

## 3. Current exactness and implementation boundary

The new work in ETH-018–024 is research-grade reference code. Do not confuse:
- Python/JavaScript reference tests with a Rust build;
- passing finite fixtures with a universal theorem;
- exact coefficient arithmetic with deterministic solver correctness;
- a digest with a proof (the verifier must replay the operation);
- a draft PR with merged/release-qualified code;
- a manuscript exponent with the linked Lean theorem's actual scope.

The environment audit for this pass found Python 3.13.5, but no cargo, rustc, lake or lean executable. GitHub Actions reports for the sampled draft PR heads show CI and PR Governance workflows completed as skipped. Therefore no new Rust or Lean pass is claimed.

## 4. Highest-priority execution sequence

### P0 — Make semantics and evidence non-negotiable
- Finish ETH-023's golden corpus, and differentially compare parsed type/output contracts against official MC2026 fixtures and Ganak's parser.
- Preserve both raw declared mode and effective output contract; route the engine by the latter.
- Finish ETH-024's ledger integration in benchmark outputs and PR templates. Any source SHA drift requires explicit re-audit.

### P1 — Qualify the base exact computation path
- Obtain a reproducible Rust/Lean build environment and capture toolchain versions, lockfile, target, exact commit SHA and full logs.
- Compile the existing exact semiring/profile code before expanding the algebra surface.
- Port the ETH-022 trace producer and checker to Rust; keep checker in a separate crate/binary with no producer dependency.
- Extend the trace with coefficient-domain identity, exact input/output digests, algebra-version hash and explicit zero-factor semantics.
- Prove the local bucket-step theorem and induction over elimination order in Lean.

### P2 — Add high-value semantics
- Complete PMC/PWMC mixed-quantifier certificates: all hidden variables must be existentially eliminated in Boolean support before projected accumulation.
- Add exact Q(i) to the runtime only after the canonical encoding is pinned and arithmetic is proved/tested; inspect actual MC2026 coefficients before supporting larger algebraic number fields.
- Preserve exact negative-weight cancellation; add fixtures where exact zero is a required output.

### P3 — Search over more than variable orders
- Port ETH-018's typed term DAG and first five guarded rewrite schemas; formalize the guards and laws before bounded rewrite search.
- Extend ETH-019 from connected-component splitting to conditioning/cutsets and verified slicing. Add tree/hypergraph/tensor structural witnesses in separate stages.
- Include proof size and verifier replay work in plan extraction; return a Pareto frontier rather than an opaque weighted sum.

### P4 — Competitive evaluation
- Freeze all public competition inputs and versioned parser profiles.
- Split train/validation/test by structural family, not random instances from one family.
- Match CPU, memory, timeout, thread count, compiler flags and output contracts.
- Include parse time, producer time, proof/trace generation, independent check time, peak RSS, largest intermediate, coefficient bit length, trace bytes, exact solved count and wrong-answer count.
- Publish raw machine-readable per-instance results and exact commit/toolchain SHAs.

## 5. Industry baselines and benchmark targets

### Model counting / weighted counting / AMC

Ganak is the essential 2026 comparison. It already supports exact arbitrary-precision MPQ, including complex-rational weights, and broader polynomial/custom-field modes. Q(i) alone is not a novelty claim. The 2026 entry reports separate exact counts for MC, WMC, PMC, PWMC and complex AMC; exact and approximate tracks must remain separate.

Other compatible baselines include D4, MICE, POG/CPOG and exact knowledge-compilation systems. Certified model counting has prior work including proof systems for dynamic programming and Decision-DNNF, certifying compilers, and formally verified POG/CPOG checkers. Compare only the same input/output semantics.

### Exact optimization

For sparse Max-Cut/QUBO, compare exact Max-Cut branch-and-cut systems such as Rehfeldt–Koch–Shinano. For a full score profile or optimum multiplicity, require a baseline that returns the same output; an optimizer returning one cut is not a full-profile baseline.

### Tensor networks / structural planning

Compare compatible contraction plans with opt_einsum and cotengra, including FLOPs, largest intermediate, slicing, memory and contraction-tree search. Compare hypergraph decompositions with generalized/fractional hypertree-width algorithms where the instances and guarantees match. These structural parameters and planners are prior art; the research hypothesis is proof-aware, algebra-aware integration with verifier cost.

### Algebraic rewrite discovery

Equality saturation (egg), rewrite inference and e-graphs are established. The existing Symthaea EML e-graph is not a safe direct backend for exact algebraic rewrites until its EML rules are separated from closed algebra domains and proof obligations. ETH-018 must validate matched terms/scopes and exact coefficient laws, not rely only on analysis metadata attached to e-classes.

| Output contract | Strong baselines | Minimum acceptance | Aspirational competitive target |
|---|---|---|---|
| SAT / feasibility | CaDiCaL, Kissat, compatible CSP tools | zero false answers; exact oracle agreement on tractable cases | held-out solved-instance improvement without time/RAM advantage |
| #SAT exact count | Ganak, D4, MICE / POG-CPOG where matching | 100% exact oracle agreement; zero invalid accepted certs | ≥10% more solved held-out instances or lower verified cost |
| WMC with signed rational weights | Ganak exact MPQ, compatible exact WMC tools | exact signed arithmetic/cancellation; matched parser semantics | ≥10% lower verified end-to-end cost on a preregistered family |
| PMC / PWMC | Ganak PMC/PWMC, projected knowledge compilation | witness multiplicity never leaks into projection count | better exact solved count with proof verification included |
| Complex AMC | Ganak exact MPQ, other exact AMC entries | exact Q(i) agreement; exact-zero cancellation | proof-carrying result at competitive solve + check cost |
| MaxCount / optimum multiplicity | exact Max-Cut/QUBO and treewidth DP baselines | optimum and multiplicity both exact | held-out structured improvement with a matched output contract |
| Full polynomial profile | enumeration/treewidth DP/profile-specific tools | all coefficients exact | wider tractable frontier under the same memory/time limits |
| Pareto frontier | exact multiobjective DP | exact nondominated vectors and multiplicities | lower verified cost and frontier growth on held-out cases |
| Tensor contraction | opt_einsum, cotengra and compatible exact planners | same exact output tensors; verify contraction tree/slices | better verified FLOPs/memory Pareto set on a structured family |

The 10% targets are aspirations, not current results. A performance claim requires both producer and independent checker costs. Report exact and approximate tracks independently.

## 6. Claim governance

The OpenAI ETH result must remain split into:
- the Lean scope note's formalized fixed-width k-CNF exact counting/SAT bound with the squared log-log factor and polynomial all-prefix visited space;
- the sharper general bound stated in the October 25 manuscript;
- the sharper sparse corollary stated in that manuscript.

The latter manuscript bounds are not promoted to Lean formalization based only on the current ComparatorChallenges theorem manifest. The October 29 weighted-counting preprint is a separate sparse weighted result with commutative Q-algebra assumptions and its own provenance; it is not the sharper dense OpenAI construction.

## 7. What would count as a real breakthrough

A defensible breakthrough would be a frozen public benchmark slice on which Symthaea:
1. solves the same exact problem/output contract as the strongest baseline;
2. returns a deterministic replayable certificate;
3. has zero accepted invalid certificates in an adversarial mutation suite;
4. has lower producer + certificate + checker cost, or a clearly valuable independent assurance guarantee at a stated overhead;
5. reproduces exactly from pinned input, code, toolchain and evidence artifacts.

Until that exists, describe the work as a promising research architecture with finite reference evidence, not as a better-than-SOTA solver.

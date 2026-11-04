# ETH-027: projected support certification and the typed arithmetic boundary

**Status:** Research/design specification. This document records an integration direction and a reproducible first experiment. It does not claim that the SCPOG adapter, a weighted proof system, or a Rust implementation exists in Symthaea.

Parent: [ETH-027 issue #10305](https://github.com/Luminous-Dynamics/symthaea/issues/10305)
Prior work: [ETH-025 #10293](https://github.com/Luminous-Dynamics/symthaea/issues/10293), [ETH-026 #10301](https://github.com/Luminous-Dynamics/symthaea/issues/10301).

## 1. Research change: certify the projection before weighting it

The key prior art is Bryant, Tan, and Heule, *Certifying Projected Knowledge Compilation*, SAT 2025, DOI [10.4230/LIPIcs.SAT.2025.8](https://doi.org/10.4230/LIPIcs.SAT.2025.8). The accompanying software is the MIT-licensed [Skolem CPOG Projected Knowledge Compilation Verifier](https://zenodo.org/records/15628075), archive `scpog.zip`.

For a CNF formula \(\varphi(D,Y)\), with data/visible variables \(D\) and quantified/hidden variables \(Y\), the compiler constructs a d-DNNF representation \(\psi(D)\) of:

\[
\psi(D) \equiv \exists Y\;\varphi(D,Y).
\]

The SCPOG/Skolem-POG proof object retains enough witness information to certify both directions of this projected equivalence. The paper's soundness framework is formally validated in HOL4. It provides a high-performance C checker and a CakeML checker whose implementation is formally verified in HOL4.

This means a compact, certified projected-support compilation path already exists. ETH-027 must integrate and benchmark it rather than treating explicit projected enumeration as the only certification route.

### The important separation

The SAT 2025 artifact certifies the Boolean projected support and d-DNNF properties. It does **not** by itself certify an arbitrary signed-rational or complex weighted result.

The right composition is:

1. Verify that \(\psi(D)\) denotes exactly the visible assignments having at least one hidden witness.
2. Independently evaluate \(\psi(D)\) under a declared exact arithmetic domain and visible literal-weight map.
3. Bind the arithmetic evaluation to the exact support-circuit digest, coefficient-domain ID, weight-map digest, evaluator version, and output bytes.
4. Independently replay the arithmetic result.

For PMC, evaluate the number of models of \(\psi(D)\). For PWMC, evaluate the weighted sum over the same distinct visible models. Hidden witness multiplicity must already have been collapsed by existential projection, before any arithmetic summation over visible assignments.

For a deterministic, decomposable support circuit, OR branches have disjoint model sets and AND children have disjoint variable scopes. This makes bottom-up weighted evaluation sound for compatible commutative semirings/rings. Exact \(\mathbb Q\) and \(\mathbb Q(i)\) arithmetic can then preserve signed and complex cancellation. This arithmetic bridge is a Symthaea implementation obligation, not a claim that the SAT 2025 SCPOG paper already emits a proof of every such weighted value.

## 2. What the SAT 2025 result actually achieved

The authors evaluated on 389 unique formulas assembled from the public and private 2024 unweighted and weighted projected model-counting tracks.

Reported results:
- Original D4 and modified D4 each compiled 175 formulas within a 1,000-second compilation limit.
- The modified D4 compiler had a median 1.11× slowdown and maximum 1.8× slowdown relative to the unmodified version.
- 172 of the 175 compiled cases completed full verification; three proof-generation runs timed out at 10,000 seconds.
- Skolem POG size overhead had a median 34× and maximum 4,857× relative to the ordinary POG.
- For the 172 fully verified cases, proof generation/checking plus counting took a median 2.95× the modified compilation time and a maximum 191.7×.
- The CakeML checker completed 156 of those 172 cases; 16 exceeded its 48 GB heap limit.

These are strong feasibility results and equally strong evidence that proof size, proof generation, and checker memory must be first-class optimization objectives. They are not performance numbers for Symthaea. The paper used a 2021 M1 MacBook Pro with 64 GB RAM; its resource configuration is not directly interchangeable with MC competition limits.

## 3. Architecture: two evidence layers, not one universal proof language

Keep source-to-support correctness separate from support-to-number correctness.

### A. Semantic envelope

A versioned Symthaea envelope should bind:
- canonical source-instance bytes and digest;
- input-format and parser-profile version;
- output contract: `#sat`, `wmc`, `pmc`, `pwmc`, or `amc-complex`;
- the complete variable universe and visible/hidden partition;
- quantifier semantics and order;
- coefficient-domain identifier and arithmetic version;
- support-circuit digest and canonicalization version;
- proof adapter ID, proof bytes digest, checker identity and result;
- arithmetic evaluator ID, weight-map digest, exact result encoding and replay result;
- unsupported guarantees and failed obligations.

A digest identifies bytes. It is not a proof of semantic equivalence.

### B. Support proof adapter

First adapter target: SAT 2025 SCPOG / Skolem POG.

Required receipt:
- source CNF digest;
- visible and hidden variable sets;
- compiler/toolchain versions and exact binary/source digests;
- projected support-circuit digest;
- SCPOG proof digest;
- checker version and exit status;
- decomposability/determinism claims established by the checker;
- explicit failure/timeout status (never converted into success).

Keep standard CPOG and projected SCPOG as distinct profiles. The public [rebryant/cpog](https://github.com/rebryant/cpog) Lean toolchain is useful standard-POG prior art, but it is not the SAT 2025 CakeML/HOL4 projected checker.

### C. Exact arithmetic adapter

Input: a verified support circuit plus a fully specified weight map.

Start with:
- Boolean unit weights for PMC;
- exact nonnegative rational visible weights;
- signed rational visible weights with exact cancellation;
- Gaussian rational visible weights in \(\mathbb Q(i)\).

The evaluator should return per-node values or a deterministic replay trace, an exact root result, and a receipt bound to the circuit and weight-map digests. It must reject unsupported domains or circuit properties, not silently coerce them.

For a PWMC result, the support circuit has variables \(D\) only. Each supported assignment \(d\) contributes its visible literal-weight product once:

\[
\operatorname{PWMC}(\varphi,w,D)
=
\sum_{d\in\{0,1\}^{D}}
[\psi(d)]\prod_{i\in D}w_i(d_i),
\qquad
\psi(D)\equiv\exists Y\,\varphi(D,Y).
\]

A negative or complex weight can make the final result zero by cancellation even when the support is nonempty. Preserve support cardinality separately from the arithmetic output so that a zero weighted sum is never confused with unsatisfiability.

## 4. ETH-026 v1 audit: why v2 needs an independent evaluator

The current v1 reference evaluator is a useful finite semantic oracle, but it enumerates all assignments in the declared variable universe. Its 5,000 randomized cases are generated as flat root sums of disjoint complete-assignment products (minterm DNF). Consequently those tests do not establish efficient bottom-up evaluation over general nested/shared circuit DAGs.

A bottom-up evaluator needs explicit scope discipline:
- product children must have pairwise-disjoint scopes;
- deterministic sum branches must have disjoint support;
- weighted model counting over a declared universe must account for variables absent from a child branch;
- either smooth sums must be required, or missing variables must be handled by explicit scope-completion factors;
- projected support must be produced before weighting visible assignments;
- shared DAG nodes must be evaluated once without changing multiplicity.

The v2 validator should be strict and machine-executable. The current v1 JSON description is not itself a complete JSON Schema. Canonical serialization must specify UTF-8, key ordering, rational reduction, positive denominators, Q(i) component encoding, stable node IDs/order, unknown-field rejection, and digest algorithm/version.

Avoid one evaluator checking its own assertions. The test triangle should be:
1. direct exhaustive assignment-set oracle for small instances;
2. recursive IR interpreter;
3. structurally independent bottom-up evaluator for the certified circuit subset.

Generate nested circuits from a bounded grammar, including shared sub-DAGs, non-smooth sums, free variables, hidden-witness duplicates, exact-zero branches, and malformed scope/determinism claims. Shrink every counterexample and retain it permanently.

## 5. Adapter portfolio

| Route | Intended guarantee | Do not claim |
|---|---|---|
| SAT 2025 SCPOG / Skolem POG | Certified equivalence of source CNF projection and Boolean data-support d-DNNF | That its proof certifies every signed/complex weighted output |
| Standard CPOG | Certified source-formula ↔ POG equivalence and count for compatible formats | That the public Lean toolchain is the projected SCPOG implementation |
| VeriPB / CakePB | Formally checked explicit projected enumeration under its supported proof contract | General compact PMC/PWMC weighted-value proofs |
| EVDD / EV-SDNNF | Weighted exact function representations over a supported coefficient algebra | That every tensor/circuit import is automatically equivalent or tractable |
| Tensor trains / tree tensor networks | Structured weighted representations and planning alternatives | A performance win without matched contraction/circuit experiments |
| ETH-022 replay trace | Independently replayed elimination computation | A universal proof format |

Do not create a new universal proof language. Use the semantic envelope to bind external proof artifacts and exact evaluators while preserving each system's guarantee boundary.

## 6. First reproducible experiment

### Reproduction
1. Obtain SAT 2025 Zenodo record 15628075 and record size, SHA-256, license, and tool versions. Zenodo lists a 3.9 MB MIT-licensed `scpog.zip`; its listed MD5 is `8cd11c2b6d718e764335101b85d6e124`. Recompute stronger hashes on the downloaded bytes.
2. Reproduce one supplied small test, then one 2024 public PMC/PWMC benchmark from the paper's dataset.
3. Run modified D4 → SCPOG generation → unverified C checker → CakeML verified checker. Record each outcome independently.
4. Export the projected d-DNNF support into a narrow Symthaea Boolean-support profile.
5. Compare PMC and exact rational PWMC arithmetic against direct enumeration on small examples.
6. Keep signed/complex arithmetic as a separate extension test, not as a claimed property of the upstream SCPOG proof.
7. Pin every input, compiler, proof, checker, circuit, weight map, output, and log by digest.

The archive was located but its download timed out in this research session; it has not been downloaded or built here.

### Performance measurements

Report per instance:
- compiler wall time and peak RSS;
- proof generation time;
- unverified and verified checker time/RSS;
- SCPOG bytes, POG nodes/edges, Skolem nodes/edges;
- support circuit nodes/edges and scope width;
- arithmetic evaluation time/RSS;
- coefficient bit-height and weight-map size;
- total verified end-to-end time;
- exact result, support cardinality when feasible, and whether the run was verified, failed, unsupported, or timed out.

Use the same hardware, CPU/thread limit, memory limit and timeout for Symthaea and baseline runs. Keep compiler time, proof generation, checking and arithmetic time separately visible.

## 7. Benchmarks and competitive strategy

Use the SAT 2025 public/private projected dataset used by the paper for direct reproduction, then the official MC 2026 instance families for held-out evaluation. Split by structural family, not random samples from one generator, to avoid leakage.

Use the best compatible incumbent per contract:
- D4 and Ganak for exact #SAT and WMC;
- Ganak and D4 for PMC;
- DPMC/GPMC and projected-knowledge-compilation systems for projected variants where semantics match;
- Ganak, D4 and GPMC for complex AMC, with exact-vs-approximate modes explicitly separated.

The official MC 2026 mixed projected-weighted track did not receive genuine PWMC instances. Its score must not be presented as a PWMC-specific result. Construct a separate true-PWMC suite with:
- visible/hidden ratios from very small to highly projected;
- many-to-one hidden witness distributions;
- visible and hidden free variables;
- both literal weights explicitly set;
- positive rational, signed rational and Q(i) weights;
- exact cancellation and exact-zero outputs;
- sparse/dense visible scopes and structured families.

The primary competitive score should be **verified exact instances solved** under matched resource limits, with zero wrong answers. Also report producer time, proof-generation cost, independent checker cost, arithmetic cost and total verified end-to-end cost. A reasonable aspirational goal is at least 10% improvement in verified end-to-end cost or exact held-out solves on a preregistered structural family, with no correctness regressions. This is a target, not an achieved result.

## 8. Research hypotheses worth testing

1. **Proof-aware compiler portfolio:** choose among D4-style top-down decomposition, component caching, tree-decomposition/project-join, and circuit/tensor-structured compilation using structural presolve features. Optimize a Pareto frontier of solve time, proof size, checker work, memory and exactness—not one opaque cost scalar.
2. **Certificate-aware decomposition:** choose variable orders and cuts that minimize both intermediate width and the cost of proving determinism, decomposability and source equivalence.
3. **Reusable projected support:** amortize SCPOG verification once, then evaluate the certified visible support under multiple exact visible weight maps. Cache only by the full support digest and weight-map digest.
4. **Proof-size reductions:** measure whether sharing, node ordering, proof-step factoring and direct graph inference reduce Skolem POG growth without weakening the checker.
5. **Semantic mutation testing:** inject a wrong variable partition, omitted weight, hidden-witness multiplier, false determinism bit, missing scope factor or changed parser default. The qualification harness must kill every mutant.

## 9. Acceptance gates

- 100% exact small-instance equivalence across direct oracle, IR interpreter and bottom-up evaluator.
- 100% proof/checker agreement for all supported SCPOG fixtures; zero malformed certificates accepted.
- 100% canonical serialization round-trip identity.
- Zero PMC/PWMC hidden-witness multiplicity leakage.
- Exact signed-rational and Q(i) cancellation, with support cardinality reported separately.
- Explicit non-smoothness/free-variable completion handling.
- No unsupported proof guarantee silently accepted.
- Reproducible benchmark manifest and per-instance JSON receipts.
- Any claim of advantage requires a preregistered held-out structural family and matched resource limits.

## 10. Claim boundary

ETH-027 is a research integration design and reproduction plan. The SAT 2025 SCPOG implementation is established prior art. The current ETH-026 v1 Python prototype is a finite exhaustive reference evaluator, not a scalable bottom-up circuit engine. No new Symthaea SCPOG adapter, Rust build, Lean/HOL4 verification, solver differential, or SOTA advantage is claimed by this document.


## 11. Executed ETH-027 bottom-up reference prototype

A small local reference prototype now tests nested deterministic/decomposable decision circuits and bottom-up exact evaluation:
- Source: `reference/projected_support_bottom_up.py`
- Manifest: `reference/projected_support_bottom_up_results.json`
- Executed environment: Python 3.13.5, seed `0xe7027`
- Exact direct truth-table oracle comparisons: 5,000/5,000, 1,000 per contract (#SAT, WMC, PMC, PWMC, complex AMC over Q(i)).
- Hidden-witness invariance cases: 500/500.
- Circuit structure traversed: 159,097 nodes, including 12,693 non-smooth OR nodes and 11,727 nodes referenced by multiple parents.
- Non-smooth OR scope completion: exact result 19/28, matching the independent oracle; support cardinality 3.
- Free-variable completion: root scope {x}, exact result 864, matching oracle; support cardinality 4.
- Signed rational WMC cancellation: exact zero with support cardinality 2.
- Complex Q(i) cancellation: exact zero with support cardinality 2.
- Invalid controls rejected: overlapping product scopes, false OR determinism, and binary floating-point weights in exact mode.
- Mutation controls killed three semantic bugs: omitting non-smooth-scope completion changes 19/28 to 17/14; omitting root free-variable completion changes 864 to 3; summing over two hidden witnesses instead of existentially projecting changes PWMC 2/3 to 4/3.

The exact locally executed source has Git blob SHA `4f35cc8e20f82a070fd4444687e2f3e317b2d69b`; fetching the committed file back from GitHub returned the same blob SHA. The result manifest binds this source SHA.

### Interpretation and limitations

This is evidence for the bottom-up arithmetic rule on small finite nested deterministic/decomposable DAGs, including non-smooth scope completion and shared subgraphs. The arithmetic evaluator traverses the DAG bottom-up and memoizes no subtree in the result representation; the reference compiler and determinism validation still use exhaustive truth tables. Shannon decomposition here is not a general source-CNF compiler. The experiment does not invoke or verify the SAT 2025 SCPOG artifact, does not implement a proof adapter, and is not a Rust/Lean/HOL4 build, solver-binary differential, large-scale benchmark, or SOTA result.
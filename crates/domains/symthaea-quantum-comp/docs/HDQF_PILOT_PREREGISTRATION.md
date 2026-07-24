# HDQF Cryptanalysis Pilot Preregistration

**Protocol ID:** `symthaea-hdqf-pilot-2026-01`  
**Protocol version:** 1.0.0  
**Decision date:** 2026-07-13  
**Literature baseline frozen:** 2026-07-13  
**Planned duration:** 12 weeks  
**Claim boundary:** controlled hypervector factorization only

## 1. Decision being made

This pilot asks:

> Across explicit instance families, access assumptions, and codebook-reuse
> horizons, does Hyperdimensional Quantum Factorization (HDQF) occupy a useful
> query-time-memory Pareto region relative to structure-aware classical
> factorization?

The pilot makes a go/no-go decision for further **HDQF factorization** work. It
does not make experimental continuation decisions for side-channel HDC, toy
algebraic cryptanalysis, or reversible-circuit synthesis. Those tracks receive
readiness recommendations only.

The pilot does not test integer factorization, elliptic-curve key recovery, or
any deployed cryptosystem.

## 2. Hypotheses

### Null hypothesis H0

After charging for codebook access, oracle construction, state preparation,
uncomputation, measurement, and classical verification, HDQF has no useful
resource tradeoff over structure-aware classical algorithms.

### Alternative hypothesis H1

For at least one preregistered instance family and reuse horizon, HDQF is
Pareto-nondominated and retains a favorable trend across at least three
increasing codebook sizes.

Query advantage, memory advantage, and end-to-end advantage are separate
claims. Evidence for one must not be relabeled as another.

## 3. Frozen problem definition

For dimension \(D\), factor count \(F\), and codebooks
\(C_1,\ldots,C_F\), each codebook contains \(N\) bipolar hypervectors in
\(\{-1,+1\}^D\). Binding is component-wise multiplication:

\[
  t = c_{1,i_1} \odot c_{2,i_2} \odot \cdots \odot c_{F,i_F}.
\]

This is equivalent to bitwise XOR under the fixed mapping
\(-1\leftrightarrow1\), \(+1\leftrightarrow0\). Implementations may use either
representation, but conversion time and memory must be recorded.

For noisy instances, each target coordinate is independently flipped with
probability \(\epsilon\). The objective is then minimum Hamming-distance
factorization. Ties are retained; they are not resolved using knowledge of the
planted tuple.

Candidate multiplicity \(\mu\) is the number of tuples that attain the exact
target for noiseless instances or the globally minimum observed Hamming
distance for noisy instances. For cells where exhaustive enumeration is not
feasible, `mu` must be reported as unknown rather than estimated without a
separate method label.

Two success outcomes are recorded:

1. `any_valid_factorization`: the returned tuple attains the global optimum;
2. `planted_factorization`: the returned tuple equals the planted tuple.

The first is the primary correctness outcome. The second diagnoses collisions
and is never substituted for the first.

## 4. Preregistered instance families

Every algorithm is evaluated on identical serialized instances from these
families:

1. **Planted unique:** independently random codewords; resample the complete
   instance until \(\mu=1\) at sizes where enumeration is feasible.
2. **Random:** independently random codebooks with no uniqueness conditioning.
3. **Collision rich:** choose \(N^F>2^D\), or inject documented duplicate
   products, so multiple valid answers occur.
4. **Correlated:** derive codewords from shared prototypes at a preregistered
   correlation rate.
5. **Adversarial:** construct near-collision products separated by a small
   Hamming margin, without changing the target after observing algorithm output.

Instance generation must finish before outcome analysis. Each instance records
its generator version, seed, realized multiplicity when known, correlation
parameters, and target margin.

## 5. Pilot matrix

### Correctness matrix

The complete Cartesian product is:

- \(D\in\{8,16\}\);
- \(F\in\{2,3\}\);
- \(N\in\{4,8\}\);
- \(\epsilon\in\{0,0.05\}\);
- all five instance families;
- 20 deterministic instance seeds.

Every cell is exhaustively enumerated and every non-heuristic implementation
must agree with that ground truth.

### Scaling matrix

The complete Cartesian product is:

- \(D\in\{16,32,64\}\);
- \(F\in\{2,3,4\}\);
- \(N\in\{4,8,16\}\);
- \(\epsilon\in\{0,0.01,0.05,0.10\}\);
- random, collision-rich, and correlated families;
- 20 deterministic instance seeds.

Planted-unique and adversarial cells are added where ground-truth construction
is feasible, but are reported as a separately labeled extension rather than
silently omitted from the frozen matrix.

Quantum simulation may hit a preregistered resource ceiling. The initial
ceiling is 20 statevector qubits or 16 GiB resident memory, whichever comes
first. Such cells are `resource_censored`, retained in the output, and never
dropped from scaling plots. Raising the ceiling is a protocol deviation.

The accounting-only codebook-reuse horizons are
\(R\in\{1,10,100,1000\}\). A quantum circuit is not rerun \(R\) times merely to
calculate amortized construction cost.

## 6. Algorithms and baseline rules

Required classical baselines are:

- exhaustive enumeration;
- hash-based meet-in-the-middle using the best balanced split for \(F\);
- a generalized-birthday or k-XOR method where its preconditions apply;
- a published resonator-network implementation or a faithful reproduction;
- noise-aware nearest-neighbor retrieval for \(\epsilon>0\).

Required quantum baselines are:

- ordinary Grover/amplitude amplification over the same candidate index space;
- reproduced HDQF using the paper's oracle and iteration rule.

The generalized-birthday implementation must state its list-size, independence,
and solution-density assumptions. A method is marked `not_applicable` when its
preconditions fail; it must not be assigned an infinite or zero cost that
distorts Pareto analysis.

All heuristic algorithms receive the same stopping policy and success budget.
Hyperparameters are selected using disjoint calibration instances and frozen
before the scaling matrix is evaluated.

## 7. Access and amortization models

Every quantum result is labeled with exactly one access model.

### A. Ideal oracle

The marking oracle is treated as one query. This model supports query-complexity
claims only. It reports no constructed gate count.

### B. QRAM-like access

Codeword lookup is modeled using explicit latency, width, coherence, and build
assumptions. Hardware construction is not implied. Results remain conditional
on those assumptions.

### C. Explicit reversible ROM

Codebook lookup, product construction, target comparison, phase marking,
workspace cleanup, and inverse operations are constructed using a fixed logical
gate set. Counts must be extracted from the constructed circuit rather than an
asymptotic label.

For models B and C, report both:

- construction charged to every isolated target;
- construction amortized over \(R\) targets sharing an unchanged codebook.

For each resource coordinate \(x\):

\[
  x_{\text{per-target}}(R) = x_{\text{build}}/R + x_{\text{query}}
  + x_{\text{readout}} + x_{\text{verify}}.
\]

Break-even reuse horizons are reported only when both compared methods use
compatible units and accounting boundaries.

## 8. Resource and evidence vocabulary

No scalar "advantage score" is permitted. Preserve at least these coordinates:

- wall time and peak resident memory;
- oracle queries and codebook accesses;
- logical qubits and clean/dirty ancillas;
- Clifford, T, Toffoli, measurement, and reset counts;
- logical depth and T-depth;
- preparation, oracle-build, uncomputation, readout, and verification cost;
- shots, success probability, and confidence interval.

Every numerical resource claim has one evidence level:

1. `asymptotic_accounting`;
2. `symbolic_resource_formula`;
3. `measured_classical_execution`;
4. `constructed_logical_circuit`;
5. `transpiled_circuit`;
6. `simulated_circuit`;
7. `physical_backend_observation`.

Evidence levels are not interchangeable. In particular, symbolic or ideal
models must not populate fields described as measured constructed-circuit
counts.

## 9. Primary and secondary endpoints

The primary endpoint is Pareto status over the frozen coordinates. Method A
dominates method B only if A is no worse on every compared coordinate, strictly
better on at least one, and the methods have compatible evidence levels and
success targets.

H1 receives support only when HDQF is nondominated for at least three increasing
values of \(N\) at fixed \((D,F,\epsilon,\text{family},R)\), with uncertainty
intervals that do not reverse the claimed ordering.

Secondary endpoints are:

- empirical scaling exponents with confidence intervals;
- break-even reuse horizon \(R^*\);
- success degradation under noise;
- sensitivity to multiplicity and target margin;
- divergence between ideal, QRAM-like, and explicit-ROM conclusions.

Simulator wall-clock performance is an implementation diagnostic, never a
quantum-advantage endpoint.

## 10. Statistical protocol

- Use 20 preregistered instance seeds per matrix cell.
- Use at least 1,024 shots for stochastic quantum cells, increasing shots until
  the Wilson 95% interval half-width is at most 0.03 or 65,536 shots are reached.
- Bootstrap instance-level medians and paired differences with 10,000 resamples.
- Fit scaling only across cells with identical instance family, access model,
  success target, and evidence level.
- Report failures, timeouts, resource censoring, and non-convergence in the
  denominator.
- Correct families of confirmatory comparisons using Holm's method. Exploratory
  comparisons are labeled exploratory.

## 11. Go/no-go rule

Continue HDQF work when at least one of these outcomes is documented:

1. a nondominated time-memory-query frontier persisting across three increasing
   \(N\) values;
2. a favorable repeated-codebook regime with a finite, reproducible \(R^*\);
3. statistically supported noise robustness over resonator and nearest-neighbor
   baselines at matched resources;
4. an explicit access regime whose assumptions are technically plausible and
   whose advantage survives complete accounting;
5. a rigorous negative result that localizes the loss to access, construction,
   measurement, verification, or classical structural algorithms.

Only outcomes 1--4 justify a positive continuation. Outcome 5 justifies a
negative-results paper and stopping the factorization track.

## 12. Twelve-week execution plan

- **Weeks 1--2:** freeze this protocol, schema, generators, resource units, and
  deviation log.
- **Weeks 3--5:** implement and cross-check all classical baselines.
- **Weeks 6--8:** reproduce ideal HDQF and Grover; validate every correctness
  cell against enumeration.
- **Weeks 9--10:** add QRAM-like accounting and explicit reversible-ROM
  construction with uncomputation.
- **Weeks 11--12:** execute the matrices, compute Pareto/scaling analyses, and
  issue the HDQF go/no-go memo plus readiness notes for the other three tracks.

## 13. Deviations and exclusions

Once the first outcome-bearing scaling run begins, this document is frozen.
Any change requires a machine-readable deviation record containing:

- timestamp and author;
- affected fields and cells;
- reason discovered;
- whether outcomes were inspected;
- expected direction of bias;
- decision to restart, version, or label the affected analysis exploratory.

No cell may be removed because it is slow, unfavorable, collision-rich, noisy,
or inconsistent with the expected claim.

## 14. Responsible-research boundary

All instances are synthetic bound hypervectors. No live system, production key,
third-party trace, or deployed cryptographic target is in scope. Any accidental
operational vulnerability follows the project's coordinated-disclosure policy.

## 15. Machine-readable contract

Run manifests and completed results use
`schemas/hdqf-pilot-experiment-v1.schema.json`. The canonical smoke fixture is
`fixtures/hdqf-pilot-smoke.json`. A run is inadmissible when required fields are
missing, its protocol version differs, or an unlogged deviation changes a frozen
field.

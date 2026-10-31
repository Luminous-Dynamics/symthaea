# AR-5B5 deterministic MOEA/D-DE baseline v0.1

**Status:** deterministic reference implementation committed; tests authored but not compiled or executed. No convergence, runtime or SOTA result is claimed.

## Purpose and scope

NSGA-II is a Pareto-ranking baseline. MOEA/D provides a different design point: decompose the multi-objective problem into scalar subproblems with reference weight vectors, then evolve neighbouring subproblems through shared candidate evaluations.

File:
\`crates/domains/symthaea-architecture-engineering/src/pareto_moead.rs\`

Implemented variant: **MOEA/D-DE v0.1**, using:
- uniform line weights for two objectives;
- integer simplex-lattice weights for three objectives;
- Euclidean nearest-weight neighborhoods;
- neighborhood/global mating controlled by a frozen probability;
- differential evolution \`DE/rand/1/bin\` variation;
- Tchebycheff scalarization relative to the running ideal point;
- randomized neighborhood replacement with a frozen maximum replacement count.

This variant uses bound clamping as its decision repair and does not add polynomial mutation. Those are explicit variant choices, not an assertion that every published MOEA/D configuration uses the same operator mix.

## Weight vectors and population

For ZDT1, the \`Uniform2d\` mode builds N evenly spaced weights:

\[
w_i=(i/(N-1), 1-i/(N-1)), \quad i=0,\dots,N-1.
\]

For three-objective DTLZ2, \`Simplex3d { divisions: H }\` builds all integer-lattice triples:

\[
(i/H,j/H,k/H), \qquad i+j+k=H,
\]

giving \((H+1)(H+2)/2\) subproblems.

Population order is subproblem order and is retained in the run report. The final population at index i corresponds to weight vector i. The nondominated archive is separately canonicalized by candidate digest for analysis.

The current implementation caps the population at 2,048 because neighborhood construction computes pairwise weight distances. Higher-dimensional/larger populations need a more scalable neighbor index and explicit complexity/approximation validation.

## Evolutionary step

For subproblem i, choose three distinct parents other than i from either its neighborhood or the global population. Generate a child using:

\[
v=x_{r_1}+F(x_{r_2}-x_{r_3})
\]

with binomial crossover against the incumbent solution for i and at least one forced crossover dimension. Clamp the child to [0,1], evaluate it once, update the ideal point, then compare it against neighbouring subproblems using Tchebycheff scalarization:

\[
g^{te}(x\mid w,z^*)=\max_j \tilde{w}_j |f_j(x)-z_j^*|.
\]

A zero weight coordinate is replaced by the documented floor \(10^{-6}\) so an objective is not completely ignored in scalarization. The offspring replaces a neighbour only when its scalarized value is strictly lower; at most \`max_replacements\` neighbors are updated.

## Resource and evaluation budget

The generational/evolutionary accounting is:

\[
N_{\mathrm{eval}}=N_{\mathrm{pop}}(1+G)
\]

(initial population plus one evaluated offspring per subproblem per generation).

The report preserves weight vectors, final subproblem-ordered population, nondominated archive, problem ID, dimensions, config digest, seed and report digest. It records objective evaluations, not machine CPU/GPU/energy telemetry; the benchmark harness must measure hardware resources separately.

## Shared Pareto indicators and replay

\`verify_moead_report\` reconstructs the run from problem ID, dimensions, frozen config and seed, then compares the full report and digest. \`analyze_moead_report\` uses only the replay-verified archive and sends it through the same Pareto plan/analysis path as Symthaea and NSGA-II.

Repeated identical decision vectors may occur in a population. They remain in the full run report and evaluation count; they are deduplicated by decision digest only in the analysis plan.

## Tests authored

- 2D and 3D weight-vector normalization and exact population counts;
- deterministic ZDT1 run and exact evaluation budget;
- DTLZ2 simplex population size, objective dimension and decision bounds;
- report replay verification and shared-indicator analysis;
- preservation of decomposition subproblem order in final population;
- Tchebycheff handling of zero weight coordinates;
- neighborhoods contain self and parent selection excludes target while choosing three distinct parents;
- invalid parameters and problem/weight-mode mismatch rejection.

These tests are authored but not run. Exact-head commands:

\`\`\`bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_moead -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_nsga2 -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
\`\`\`

## Comparison protocol

Run MOEA/D-DE, NSGA-II and Symthaea with:
- the same objective function and decision bounds;
- matched objective-evaluation budgets and seed blocks;
- frozen normalization anchors, HV reference point and IGD+/epsilon reference front;
- full archive retention and common analysis code;
- separate reporting of runtime and hardware-resource usage.

Use multiple problem instances and seeds. Compare fixed-target attainment curves, not only one final score. The MOEA/D weights are a structural bias toward a set of trade-offs; report weight coverage and target-front coverage as well as indicators.

## Limitations and next steps

- The implementation is a compact MOEA/D-DE variant, not yet cross-checked against a pinned PlatEMO 4.16 or another reference implementation.
- It covers two and three objectives only.
- It has no constraints, uncertainty-aware scalarization, expensive-evaluation acquisition or parallel evaluation.
- The 3D simplex-lattice design is capped by the current 2,048 population guard.
- Next baseline: SMS-EMOA, followed by NSGA-III/RVEA for higher objective counts; extend test functions to WFG, constrained DTLZ, noisy/expensive and multimodal settings.

No performance or SOTA claim is made.

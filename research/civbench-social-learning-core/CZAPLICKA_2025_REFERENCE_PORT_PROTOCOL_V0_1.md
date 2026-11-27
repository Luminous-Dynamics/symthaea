# Czaplicka, Baumann & Rahwan (2025): reference cultural-mediation model

**Status:** source-informed Rust semantic port with deterministic RNG; not bit-for-bit reproduction of the original C random stream.  
**Rust source:** src/czaplicka_2025.rs  
**Sweep:** src/bin/czaplicka_2025_sweep.rs  
**Pinned upstream:** aga-cz/Mutual-benefits-of-SL-and-RS, main branch.  
**C source blob:** 8555ec4ad8bd7eec0605bcda28565fbfe2645957 (modelQinn.c).  
**Runner script blob:** e94d5999cc5a7c60ff072186d50febb6d87326a5 (script_runQinn.sh).

## Citation

Agnieszka Czaplicka, Fabian Baumann & Iyad Rahwan (2025). Mutual benefits of social learning and algorithmic mediation for cumulative culture. Journal of the Royal Society Interface 22(225):20240686. DOI: 10.1098/rsif.2024.0686.

- Article: https://pmc.ncbi.nlm.nih.gov/articles/PMC11978438/
- Author source repository: https://github.com/aga-cz/Mutual-benefits-of-SL-and-RS
- Archived source: https://zenodo.org/records/14718450

The paper asks how local peer social learning and global algorithmic mediation interact in cumulative culture. The article reports that the value of mediation depends on network density; the optimal balance is not equivalent to maximizing recommendations. The authors track group payoff, cultural level/complexity, and discovered items.

## Source runner defaults and sweep

The pinned shell script sets:
- population N=100;
- time horizon T=20N=2,000 sequential update steps;
- declared branch/item parameter X=100;
- per-agent effort budget L=1,000;
- social-learning cost Cs=5;
- innovation cost Ci=10;
- recommendation cost Cr=1;
- Q=200 independent realizations;
- recommendation probability r in {0.0, 0.1, ..., 1.0};
- Erdős–Rényi edge probability pER in {0,.01,...,.10,.2,.3,...,1.0}.

The C source allocates the X-sized branch array but does not use it in the update rules. The Rust config retains the declared parameter for source parity; it does not invent branch-specific behavior.

## Reference transition semantics implemented in Rust

1. Generate one undirected, static Erdős–Rényi network per replication.
2. Start all agents with effort budget L, level 0, payoff 0 and no acquired traits.
3. Give every agent one sequential initial innovation attempt; charge Ci, sample a payoff Y=round(2Z^2), with Z~Exponential(1), and register positive discoveries in a global best-payoff-per-level archive.
4. For each of T-1 sequential time steps, process agents in index order. With probability 1/N, an agent dies and is reborn with its original budget and no personal traits/payoff; the global discovery archive persists.
5. With probability r, the algorithmic recommender offers the globally best-known trait at the agent's next level if that trait exists and the agent can pay Cr.
6. Otherwise, peer social learning chooses the adjacent agent with the highest total payoff (the C code's >= tie condition means the highest-index neighbour wins equal-payoff ties). If the selected neighbour has reached the recipient's next level, copy that neighbour's trait for the level and pay Cs.
7. If recommendation or peer copying does not succeed, attempt innovation if the remaining budget covers Ci; charge the cost even if the sampled payoff is zero.
8. Later agents in a time step can observe earlier agents' updates from the same step. This sequential/asynchronous order is preserved; the loop must not be parallelized in the reference path.

The global archive is a cultural memory that outlives individual agents. The model therefore permits a population to retain a useful discovery after its originator dies.

## Port boundary and deliberate differences

This Rust implementation is a semantic port, not an exact binary replay:
- The C runner calls srand(time(0)) and uses the platform C rand(); Rust uses a deterministic SplitMix64 stream and explicit seeds.
- The C code's exact equality branch at draw == r can leave a branch flag stale from a prior user; Rust uses an explicit fallback to innovation.
- The C code's legacy max_pay and max_lvl columns sum prefix maxima while iterating users; they are not the true population maximum. Its step-zero max-level column is hard-coded to one per agent. Rust preserves clearly named legacy metrics and additionally reports true maximum payoff/level.
- The C n_items, innovations, social_learning and recommendations counters are cumulative within each replication; Rust reports cumulative counts.
- The C source tracks the global best payoff per level, but does not retain a rich explicit lineage/provenance graph. Rust adds parent/root lineage and adoption-count metadata for additional measurement; these fields do not affect the baseline policy.
- C source writes a final file after 200 realizations and uses the final 200 time steps for reported means. The Rust sweep uses per-run averages over the final min(200,T) steps and aggregates independent run summaries with 95% normal-approximation confidence half-widths.

The upstream C source's quirks are not silently fixed in the reference path. Where the Rust model adds a clearer metric, the corresponding historical output is separately labelled rather than overwritten.

## New intervention: exposure-aware recommender

The ExposureAwareUcb treatment is a CivBench extension, not part of Czaplicka et al.'s original C implementation. It ranks an available trait by:

score(i) = payoff(i) + beta * sqrt(log(2 + total_adoptions) / (1 + adoption_count(i))).

The exploration bonus rewards promising but under-exposed trait lineages. This is a controlled intervention, not an assumption that novelty is always beneficial. The evaluation must show performance and diversity together.

Required comparisons:
- exact source-rule GlobalBestByLevel;
- ExposureAwareUcb for predeclared beta values;
- peer-only (r=0);
- recommender-only (r=1);
- matched mixed r values;
- multiple network densities;
- common random seeds across policies.

Report payoff, level/complexity, remaining budget, innovation, peer-copy and recommendation counts, active distinct traits, lineage concentration (HHI), and extinction/retention if implemented. Do not treat greater diversity as automatically better if it reduces capability; use a Pareto frontier.

## Running

Quick smoke sweep:

    cargo run --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin czaplicka-2025-sweep

Source runner's N=100, Q=200, 20-density x 11-recommendation grid:

    cargo run --release --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin czaplicka-2025-sweep -- --paper-grid

New exposure-aware recommender sensitivity sweep:

    cargo run --release --locked --manifest-path research/civbench-social-learning-core/Cargo.toml --bin czaplicka-2025-sweep -- --extension-grid

The full source grid can be computationally expensive. Its CSV output must be stored with exact commit SHA, config, seed schedule and runtime metadata before results are cited.

## Acceptance gates before any "beats SOTA" claim

1. Pin upstream C source and script hashes (recorded above) and independently review the Rust transition semantics line by line.
2. Compare the Rust reference to the C implementation using a deterministic test harness or instrumented C build with injected seed/PRNG. Compare state transitions, not just aggregate curves.
3. Verify output definitions against the paper and source; preserve the unusual prefix-max metrics as legacy compatibility values.
4. Match N,T,L,Cs,Ci,Cr,pER,r,Q and all time/seed semantics for the paper baseline.
5. Reproduce paper's reported curves/summary statistics within uncertainty before adding the new recommender.
6. Evaluate UCB vs source policy on common random seeds, reporting capability and diversity separately.
7. Use additional held-out network densities and effort/cost regimes, without tuning on the test grid.
8. Report independent replicate uncertainty, tail behavior, compute/energy cost if measured, and failures.

## Limitations

The model is an abstract cultural-trait process, not a psychological model of humans or a model of all cultural evolution. The branch parameter X is currently unused in the upstream C rules and in the Rust state transition logic. No result from the Rust port is claimed to reproduce the paper until the C/Rust conformance gate is completed.

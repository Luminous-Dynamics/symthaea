# Networked cultural learning and algorithmic mediation protocol v0.1

**Status:** executable synthetic mechanism model; not a reproduction of the reference paper.  
**Code:** `src/networked_mediation.rs` and `src/bin/networked_mediation_sweep.rs`.

## 1. Research anchor

Czaplicka, A., Baumann, F. & Rahwan, I. (2025). *Mutual benefits of social learning and algorithmic mediation for cumulative culture*. Journal of the Royal Society Interface 22(225):20240686. DOI: https://doi.org/10.1098/rsif.2024.0686.

- Article and model description: https://pmc.ncbi.nlm.nih.gov/articles/PMC11978438/
- Author code: https://github.com/aga-cz/Mutual-benefits-of-SL-and-RS
- Archived code: https://zenodo.org/records/14718450

The paper studies how localized social learning and global algorithmic mediation interact in cumulative culture. It reports that the impact of algorithmic mediation grows as social networks become less dense, and that the best ratio of peer and algorithmic learning depends on network density. The authors model a global mediator with knowledge of cultural traits and a peer-learning mechanism over a social network. Their cultural space distinguishes branches and complexity levels; agents have effort budgets and different costs for social copying, innovation and recommendation. They measure mean population payoff, number of distinct traits and maximum cultural complexity.

The archived public C reference implementation is `modelQinn.c` at repository main blob SHA `8555ec4ad8bd7eec0605bcda28565fbfe2645957`. Its runner script is pinned at blob SHA `e94d5999cc5a7c60ff072186d50febb6d87326a5`. The source-defined default grid uses N=100, T=20N, X=100, L=1000, social-copy cost Cs=5, innovation cost Ci=10, recommendation cost Cr=1 and Q=200; it sweeps recommendation probability r and Erdős–Rényi edge probability pER. A separate Rust semantic port now exists in `src/czaplicka_2025.rs`, with a dedicated protocol at `CZAPLICKA_2025_REFERENCE_PORT_PROTOCOL_V0_1.md`. The port is deterministic but not a bit-for-bit C RNG replay; no paper-result replication is claimed until source-to-source conformance and aggregate result checks pass.

## 2. Synthetic Rust model implemented here

The separate `src/networked_mediation.rs` module is a deliberately simpler stage-separated mechanism extension, not the authors' model:

- A social graph begins as a connected ring; extra undirected shortcuts are added with a configured probability.
- Each generation contains a new cohort at the same graph positions.
- An initial machine-seeded capability is available to the first cohort.
- Agents can independently innovate a small quality increment.
- Peer learners can observe previous-generation agents at adjacent network positions.
- Recommender-mediated learners can access a global previous-generation candidate pool; it is ranked by noisy demonstration quality and truncated to a top-k shortlist.
- Observation, successful acquisition, transmission corruption and active adoption are tracked separately. A successful acquisition is not counted as adoption unless it improves the learner's active capability.
- Lineage IDs are retained to distinguish the spread of an original discovery from independent innovation.
- The recommender fraction is an experimental control from 0 (peer-only) to 1 (recommender-only).

This ring-plus-shortcuts model is **not** a translation of `modelQinn.c`: it uses a different topology, generation-cohort structure, trait abstraction and recommender shortlist. It does not implement the authors' within-generation asynchronous update order, effort budget, death/birth reset, level-indexed payoff archive or payoff distribution. These differences are intentional and explicit. The module is useful for isolating the interface between peer access, global curation, acquisition and lineage concentration, but its results cannot be reported as the 2025 paper's reproduced curves. For source-faithful baseline work, use `czaplicka_2025.rs`; for the new ring/shortcut mechanism experiment, use `networked_mediation.rs`. Keep their outputs in separate benchmark tracks.

## 3. Multi-seed sweep

The `networked-mediation-sweep` binary evaluates 20 conditions:

- shortcut probability: 0.00, 0.02, 0.10 and 0.50;
- algorithmic mediation fraction: 0.00, 0.25, 0.50, 0.75 and 1.00;
- 256 common seeds per condition;
- 32 agents and 8 generations per run.

The graph is generated deterministically from the seed and shortcut probability. Within each density, the same seed range is reused across mediation fractions to support paired comparisons.

Per-generation outputs include:

- actual graph density;
- mean and maximum skill quality;
- knowledge coverage;
- original seed-lineage prevalence;
- number of distinct non-zero lineages;
- effective lineage diversity (inverse HHI) and HHI concentration;
- individual innovations;
- demonstration observations and successful acquisitions;
- peer versus mediated adoptions, separately from peer versus mediated acquisitions.


## Diversity-aware recommender comparison now implemented

The Rust model exposes two recommender objectives:
- `PerformanceOnly`: rank the global candidate pool by normalized observed capability quality.
- `PerformanceDiversity { novelty_weight }`: add a bounded bonus for lineages that are underrepresented among candidates with non-zero knowledge.

The rarity bonus is calculated from the frequency of each lineage in the previous-generation candidate pool. Unknown/empty candidates receive no novelty bonus. This is not an epistemic truth score and does not reward popularity or provenance; it is an explicit diversity intervention. The default remains PerformanceOnly, making the baseline comparison clear.

The sweep now crosses 4 graph shortcut probabilities × 5 mediation fractions × 2 recommender objectives = 40 conditions, with 256 common seeds per condition. Compare mean quality/maximum quality with knowledge coverage, unique lineages, effective diversity and lineage HHI. The desired result is a Pareto frontier, not one arbitrary scalar winner: identify when diversity gains are worth a capability trade-off and whether that trade-off changes with graph sparsity. The novelty bonus is a synthetic experimental treatment, not a proven optimal recommender.

## 4. Paradigm-shifting questions

1. **Complementarity frontier:** Is there an interior mediation share that maximizes cumulative capability at each network density?
2. **Sparse-network compensation:** Does global mediation reduce skill extinction in sparse peer networks, and what is its marginal cost?
3. **Diversity versus performance:** Does a high-performance recommender increase mean quality while collapsing lineages or concentrating descendants around a single source?
4. **Innovation versus copying:** Does low-cost global access crowd out independent innovation, even when it improves short-run performance?
5. **Robustness:** How do observation failures, acquisition failures, noisy rankings and transmission corruption change the best mediation ratio?
6. **Personalization versus global ranking:** Does recommending a globally highest-scoring skill outperform recommendations conditioned on the learner's current complexity level and preconditions?
7. **Plurality and sovereignty:** Can a diversity-aware recommender preserve rare high-value lineages without suppressing the learner's ability to ignore or challenge recommendations?

## 5. Proposed next comparison against the published model

Before making any comparison claim:

1. Pin the C source commit and Zenodo archive digest; inspect the versioned supplementary implementation details.
2. Port the paper's actual trait-branch/complexity-level representation, birth/death schedule, individual effort budgets, and Cs/Ci/Cr acquisition costs into a separate Rust reference model.
3. Reproduce their ER network generation and recommendation semantics: algorithmic mediation exposes the best-payoff trait at the learner's next complexity level; peer learning uses local neighbours and payoff-biased demonstrator choice.
4. Match parameter grids, run counts, random seed protocol, time horizon and output measures.
5. Reproduce mean payoff, distinct trait count, maximum complexity, and their sensitivity curves before extending the model.
6. Only then compare the simpler ring-shortcut model or new Symthaea mechanisms against the reproduced baseline.

## 6. Additional benchmark targets

Proposed targets for the future exact reproduction and extension (not achieved results):

- zero source-code/configuration drift in the pinned reference run;
- exact graph density and degree-distribution checks for every generated ER network;
- paired common-random-number comparisons across mediation shares;
- uncertainty intervals over independent full-system seeds;
- separate reporting of mean payoff, maximum quality, distinct lineages, lineage HHI, effective diversity, innovation and adoption;
- identify the mediation-share optimum by density with out-of-sample validation;
- stress test 0/5/20% transmission corruption, top-k size, recommender noise, and network rewiring;
- require performance gains not to hide destructive loss of knowledge diversity or false confidence.

## 6A. Disclosure boundary is a separate benchmark track

The ring-plus-shortcuts recommender model above still uses a global previous-generation candidate pool; its `mediation_fraction` controls peer versus recommender access, not the privacy of recipient state. Do not describe that mechanism as federated or privacy-preserving.

A separate executable disclosure comparison now exists in `adaptive_transmission.rs` and `privacy_disclosure_sweep.rs`: centralized full inventory versus a recipient-local coarse recipe-readiness signal versus no social transmission. It is documented in `PRIVACY_PRESERVING_TRANSMISSION_PROTOCOL_V0_1.md`. The two models should remain separate:
- `networked_mediation.rs`: peer/global mediator ratio, network density and lineage diversity;
- `adaptive_transmission.rs`: what recipient-state evidence the scheduler is allowed to observe.

## 7. Security and governance extension

A recommender must not become a hidden authority. The next version should separately model authenticated source identity, evidence-calibrated quality, provenance-preserving forks, revocation of invalid traits and a diversity/novelty term. Any resulting capsule remains contestable and locally rejectable; popularity is not qualification.

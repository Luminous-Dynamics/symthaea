# PHYS-CHAIN-002 — claim/dependency ledger v1

**Date:** 2026-10-29  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8923  
**Status:** Source-grounded starter ledger; not an exhaustive audit of the 754-page working draft and not an independent reproduction.

## What the current source actually tells us

The Chain Theory compendium is explicitly labelled a **working draft**. Its abstract reports 100 theorems, 109 conditional theorems and 198 open results; 64 of the 100 theorem statements are tied to named Lean declarations that check finite computations used in their derivations. These are author-reported scope counts transcribed from the abstract, not our independently recounted totals.

The separate 24-page alpha letter and `ChainAlpha.lean` provide an especially clear example of why claim status must be decomposed. The expression for the α candidate is a computation from the model's definitions and is checked by the supplied Lean artifact. The Lean file explicitly says it does **not** prove the definitions are the right physics, formalize all cited readability-cap results, establish statements over the full unitary group, or establish the number's identity with a constant of nature. The paper names that last bridge the Photon Conjecture.

Sources: [Chain Theory announcement](https://theinnermostloop.substack.com/p/chain-theory), [alpha letter](https://alexwg.org/publications/ChainTheoryAlpha.pdf), [Lean artifact](https://alexwg.org/publications/ChainAlpha.lean), [754-page working draft](https://alexwg.org/publications/ChainTheoryCompendium.pdf).

## Paradigm shift: track the bridge, not just the theorem name

A mathematical theorem, a conditional derivation, a physical identification and an empirical prediction are distinct objects. For each claim, the ledger stores the exact source locator, upstream dependencies, what the formal or combinatorial result establishes, the missing physical bridge, a prospective discriminator and a failure condition.

Do not roll these into a single Chain Theory confidence score. A correct proof of a statement inside a model does not establish the model's ontology; a successful numerical correspondence does not make several shared-premise consequences statistically independent.

## Starter claim ledger

The machine-readable [JSON ledger](PHYS_CHAIN_002_CLAIM_LEDGER_V1.json) contains nine initial claim records; [its JSON Schema](PHYS_CHAIN_002_CLAIM_LEDGER_V1.schema.json) provides a versioned interchange contract. This is a focused starting set, not an exhaustive census of all compendium claims.

| Claim family | Source status recorded | Key gap / test owner |
|---|---|---|
| Base ontology | Axiom / postulate | Must earn support through downstream consequences that distinguish alternatives; #8923 |
| α candidate arithmetic | Formal computation from definitions | Independently reconstruct 172 records/normalization; #8921 / #8922 |
| Photon → α(0) identification | Conjecture | Prospective independent α determinations; keep raw methods and CODATA adjustment distinct; #8921 / #8922 |
| Gauge-group quotient | Theorem about the stated construction | Full Standard Model representation/charge/phenomenology map; #8923 |
| Born rule and measurement | Conditional theorem / correspondence | Audit imports and operational measurement map; #8923 |
| Discrete order coordinate | Theorem within finite model | Do not silently identify construction order with measured physical time; #8941 / #8964 |
| Thermodynamic entropy/arrow | Mixed conditional and open | Boundary count vs register count W; closure/stop sensitivity; #8941 / #8942 / #8964 |
| Particle spectrum / generations | Mixed conditional and open | Split into individual claims with quantitative observables and dependency closure; #8923 |
| Shared relational grammar (Luminous hypothesis) | Our benchmark hypothesis, not a Chain Theory claim | Held-out composition/intervention transfer vs symbolic regression and graph baselines; #8919 / Symtropy #2870–#2871 |

## Three concrete research priorities

### 1. α: stop using the aggregate as the sole success criterion

The letter reports the fixed candidate inverse as 137.035999189566155… and compares it with CODATA 2022 and three measurement routes. Its own comparison places the candidate near the CODATA aggregate and the cited electron-magnetic-moment/rubidium values, but about 5.3σ from the cited cesium-recoil value. The article also states that the experimental methods disagree beyond their stated uncertainties. This is mixed evidence, not a clean ten-digit independent confirmation and not a verdict against the candidate.

Improve the benchmark by preregistering the candidate value, uncertainty treatment, method-specific comparisons, correction/correlation model, and a rejection rule before new data are viewed. Keep CODATA's adjusted value separate from each underlying experiment that contributes to it. A future measurement should be treated as genuinely prospective only if its measurement result and analysis choices were not used to select or revise the candidate.

### 2. Time and thermodynamics: test operational maps, not labels

The compendium defines an order coordinate as a discrete count and says it is a count, not a flow. Its thermodynamic discussion distinguishes a boundary-record count from a whole-register count W and records open results in the thermal-state/bath and entropy-like mappings. The source also reports closure/stop-condition cases where the counts part company. Reproducing an acyclic move graph or pointwise count monotonicity is valuable, but it does not by itself derive proper time, the standard second law, a Gibbs state or fluctuation relations.

Use the existing #8941/#8942 and #8964 tracks. Freeze the permitted move graph, closure, stopping condition and both count definitions; enumerate every edge; include reversible-process controls; and compare the proposed mapping to standard physical observables. Any claim of physical time/entropy needs units, state/dynamics/observable maps and a quantitative discriminator.

### 3. “Reality has grammar” / relational ontology: make it win only where it should

This is a separate Luminous research hypothesis inspired by structural analogies, not a consequence of Chain Theory's quantum axiom. It becomes interesting if shared grammar rules help recover laws across *related* worlds and extrapolate to unseen compositions/interventions better than independent equation fitting. It should not be expected to beat all equation-discovery methods on every fixed-formula regression task.

Use Symthaea #8919 and Symtropy #2870/#2871. Include:

- ordinary canonical physics and independently held-out symbolic-regression tasks;
- controlled law changes / “metaphysical shifts” so memory of familiar equations cannot pass as discovery;
- unseen compositions, noisy/partial observation and intervention-choice budgets;
- negative worlds whose components have heterogeneous or non-shared rules;
- strong conventional graph learners, known-equation oracles, and mature symbolic-regression baselines;
- exact equation/grammar recovery, held-out intervention prediction, description length, calibrated uncertainty, detection of no-shared-grammar cases, wall-clock/compute and sample efficiency.

The paradigm shift is **active identification under observational equivalence**: when two models predict the same observed data, select the cheapest legal intervention that separates their predictions. Score the selected experiment before seeing its outcome. A model that fits current observations but cannot choose a discriminating experiment is not a successful scientific discovery agent.

External benchmark context: [AI Feynman](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/) established a physics-informed symbolic-regression baseline; [NewtonBench](https://arxiv.org/abs/2510.07172) tests interactive law discovery under systematic law shifts across 324 tasks/12 domains in its 2025 preprint; [SRBench](https://cavalab.org/srbench/) reports a 2025 protocol comparing 25 methods with 30 runs/dataset; and [parallel symbolic enumeration](https://www.nature.com/articles/s43588-025-00904-8) reports symbolic discovery evaluations across more than 200 synthetic and experimental problem sets. These are complementary benchmarks, not directly comparable headline scores; our comparisons must match task/data/compute budgets.

## How this ledger changes our standard of evidence

1. **Dependency closure first:** a theorem with an open upstream premise stays conditional for claims that require that premise.
2. **Count independent confirmations, not theorem headings:** claims sharing the same axiom, selected construction, calibration route or data are grouped.
3. **Proof and empirical truth are separate lanes:** proof checks finite logic/calculation; physical claims need operational mapping and observations.
4. **Every physical bridge gets an observable and a failure rule:** if there is no distinct observable prediction, label it interpretation/open rather than claiming experimental support.
5. **Prospective beats retrodictive:** freeze the value, mapping, uncertainty, and rejection interval before new measurements or held-out intervention data are viewed.
6. **Negative results remain in the ledger:** changing closure, a choice rule or a physical identification after a failure creates a new version with the previous result preserved.

## Scope and next steps

- This v1 covers nine high-leverage claim families only; it is not a full 100/109/198-claim extraction.
- No independent Lean run, record reconstruction, physical simulation or new measurement analysis was executed for this ledger.
- The next slice under #8923 is to expand one claim family at a time, cite exact compendium statements and all transitive parents, then have an independent reviewer reproduce the status classification.
- Reuse existing work: #8921/#8922 for α; #8941/#8942/#8964 for order and thermodynamics; #8919 and Symtropy #2870/#2871 for relational grammar. Do not create another general scientific IR or another copy of the alpha arithmetic audit.

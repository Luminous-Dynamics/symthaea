# ROB-COMPOSITIONAL-SHIFT-001

Status: first executable reference for **unseen combinations of familiar robotics source factors**. It is a sibling measurement contract to #11830's whole-source-group holdout, not a replacement.

Parent issue: [#11897](https://github.com/Luminous-Dynamics/symthaea/issues/11897). Parent source-group implementation: [#11830 / PR #11863](https://github.com/Luminous-Dynamics/symthaea/pull/11863).

## Two different generalization questions

**Source transfer** asks whether a policy handles a previously unseen factor level, such as a new collection site, physical article, sensor family or morphology. #11830 owns that whole-source-group split contract.

**Compositional transfer** asks whether a policy handles a previously unseen *combination* of factor levels even though every individual factor level is already represented in both training and calibration. Example: the same site and morphology levels appear in training, but the site × morphology cell being evaluated has never appeared together.

These claims are not interchangeable. A composition pass is not an unseen-site or unseen-embodiment result.

## Existing compositional benchmarks: explicit reuse, not novelty claims

Two 2026 benchmarks already establish strong adjacent controls and must be included in any comparative study:

- **MESA / BiMESA (CoRL 2026)** separate in-distribution, spatial, unseen object-instance, unseen object-category and unseen subtask-composition evaluation. The released sets use Franka and bimanual YAM embodiments; MESA-70 contains 7,000 demos over 70 tasks and BiMESA-57 6,833 over 57 tasks. The strongest reported baseline in the published table is 52.0% on MESA and 48.7% on BiMESA. MESA also studies cross-arm transfer and its strongest policy shows a large gap between familiar and held-out axes. We should reuse MESA/ BiMESA task and reset variants when license and data access permit, rather than inventing a second object/task-composition suite.
  - https://www.pair.toronto.edu/MESA/
- **ATOM-Bench (June 15, 2026)** evaluates 30 atomic tasks and 24 held-out compositional tasks across paired single-arm and dual-arm tracks, with 3,000 demonstrations and 2,700 physical rollouts across five policies. Its Atomic Score and Compositional Failure Share diagnose weak motor atoms versus poor compositional reuse.
  - https://arxiv.org/abs/2606.16826

The present contract is a separate and complementary axis: **provenance/embodiment/operating-factor composition** (e.g. familiar site levels × familiar morphology levels, paired in a joint cell never used in training or calibration). It does not claim to invent compositional manipulation generalization or to outperform MESA/ATOM. The strongest next benchmark is to cross their held-out task/skill-composition protocols with our held-out source-factor cells, then add per-episode safety, terminal reserve, route redundancy and independent replay. This asks whether task composition survives a new combination of physical deployment conditions, not only whether known skills recombine in one familiar setup.

## Factorial interaction contract

Every frozen interaction spec declares exactly two or three source dimensions, for example:
- CollectionSite × RobotMorphology;
- RobotMorphology × SensorStack;
- PayloadRegime × EnvironmentRegime;
- SimulatorFamily × TaskFamily;
- Operator × RobotArticle;
- RobotMorphology × SensorStack × PayloadRegime.

For each spec, the interaction manifest assigns every canonical factor-level tuple to exactly one partition: Training, Calibration or HiddenConfirmatory.

The split intentionally permits a factor level to appear in all three partitions while forbidding an *exact interaction cell* from crossing partitions. For every factor level used by a hidden cell, the manifest must contain at least one training cell and one calibration cell that use that same level. Thus the hidden challenge tests interaction generalization rather than simple factor novelty.

The evaluator canonicalizes dimension order, rejects duplicate cell assignments, missing factor-level exposure and mismatched hidden cases, and preserves empty hidden cells as Inconclusive.

## Outcomes

Each hidden cell is reported separately:
- paired episode count;
- distinct independent-cluster count;
- candidate and baseline safe-success counts/rates;
- successful-but-unsafe counts;
- Unknown/censored counts;
- descriptive candidate-minus-baseline effect;
- Complete or Inconclusive disposition.

Each interaction also reports training/calibration/hidden cell counts, marginal factor-level exposure counts, observed hidden-cell count, candidate rate range and worst-cell effect.

There is no pooled scalar across different interaction families, no p-value or confidence interval, and no actuation authority. Independent-cluster inference remains with #11807. A multi-factor policy comparison must account for dependence where the same cluster appears in more than one interaction report.

## Known-answer corpus

The Rust reference includes deterministic fixtures for a held-out joint cell whose individual factor levels all appear in training and calibration, factor novelty in training/calibration, exact cell leakage across partitions, pair and triple interactions, canonical ordering under input permutation, missing/duplicate cells and receipts, duplicate case and scenario identities, inconsistent/reused independence receipts, Unknown/censored outcomes, matched-arm scope and resource-boundary mismatch, and worst-cell regression remaining visible.

Synthetic fixtures qualify only the deterministic split/key/report contract. They do not establish physical transfer, truthful factor labels, sealed hidden data or statistical significance.

## Running

The crate imports the exact public SourceDimension, PairedCase, Outcome, IndependenceDisposition and ArmManifest types from the parent source-holdout reference:

    cargo fmt --manifest-path research/rob-compositional-shift-001/reference/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-compositional-shift-001/reference/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-compositional-shift-001/reference/Cargo.toml --all-targets -- -D warnings

# ROB-COMPOSITIONAL-SHIFT-001

Status: first executable reference for **unseen combinations of familiar robotics source factors**. It is a sibling measurement contract to #11830's whole-source-group holdout, not a replacement.

Parent issue: [#11897](https://github.com/Luminous-Dynamics/symthaea/issues/11897). Parent source-group implementation: [#11830 / PR #11863](https://github.com/Luminous-Dynamics/symthaea/pull/11863).

## Two different generalization questions

**Source transfer** asks whether a policy handles a previously unseen factor level, such as a new collection site, physical article, sensor family or morphology. #11830 owns that whole-source-group split contract.

**Compositional transfer** asks whether a policy handles a previously unseen *combination* of factor levels even though every individual factor level is already represented in both training and calibration. Example: the same site and morphology levels appear in training, but the site × morphology cell being evaluated has never appeared together.

These claims are not interchangeable. A composition pass is not an unseen-site or unseen-embodiment result.

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

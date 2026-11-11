# Open-World Reference-Bank Stress Test — 2026-10-22

**Status:** exact finite-state local evaluation; embedded self-test passed and 5/5 unit tests passed. This is a small falsification pilot, not an ACDB/CausalArena reproduction or an SOTA claim.

## Why this experiment exists

The previous Pareto pilot showed that adding a Jensen–Shannon-divergence (JSD) term to expected information gain (EIG) can sharply improve surprise-based discrimination of an XOR collider when the action policy has reference collider models. That result could be an artifact of reference-family knowledge rather than a robust open-world effect.

This follow-up expands the held-out set and adds a larger reference bank. It also tests a key assumption failure: intervention compliance can be incorrectly treated as perfect.

## Protocol

Implementation: \`validation/open_world_reference_stress.py\`; tests: \`validation/test_open_world_reference_stress.py\`.

- Candidate set: a three-node Boolean chain and fork.
- Narrow reference bank: collider AND and collider OR.
- Broad reference bank: the two collider references plus an alternative chain mechanism and an alternative fork mechanism.
- Six held-out worlds: collider XOR, collider XNOR, collider NAND, a chain with the second edge mechanism inverted, a fork with both child mechanisms inverted, and a reversed chain.
- Three regimes: perfect compliance at mechanism noise 0.10; known 0.75 compliance at noise 0.10; and actual compliance 0.75 but the learner assumes compliance 1.0.
- Action budget: at most two distinct interventions from the six requested \(do(X_i=v)\) actions.
- Policies: EIG-only, EIG + 0.5×JSD with the narrow reference bank, and EIG + 0.5×JSD with the broad reference bank.
- Evaluation: exact weighted ROC AUC of cumulative predictive surprisal for each held-out world against the equal-prior in-class chain/fork mixture; usable expected graph-information reduction; in-class zero-support mass; expected probe count.

The reference banks contain no exact held-out model, but they are hand-authored and share causal-family knowledge with some tests. This is still only a small, structured mechanism/topology holdout, not arbitrary open-world evaluation.

## Aggregate results

Mean OOD-surprise AUC across the six held-out worlds. Chance AUC is 0.50.

| Regime | EIG-only | EIG + 0.5×JSD, narrow refs | EIG + 0.5×JSD, broad refs |
|---|---:|---:|---:|
| Perfect compliance, noise 0.10 | 0.8044 | **0.9257** | **0.9257** |
| Actual and assumed compliance 0.75, noise 0.10 | 0.7860 | **0.8991** | **0.8991** |
| Actual compliance 0.75, learner assumes 1.0 | **0.6794** | 0.6561 | 0.6561 |

In the first two regimes, mean usable graph information was identical between EIG-only and JSD policies: 0.5022 bits with perfect compliance, and 0.4024 bits with compliance 0.75 modeled correctly. All policies had zero in-class zero-support mass in those matched regimes.

Under compliance misspecification, mean usable graph information fell from 0.3805 bits under EIG-only to 0.3025 bits under either JSD policy. Mean in-class zero-support mass increased from 0.2249 to 0.3839. Thus the extra objective does not substitute for action-model reliability.

### Family-specific results, perfect compliance

| Held-out world | EIG-only AUC | EIG + 0.5×JSD AUC |
|---|---:|---:|
| Collider XOR mechanism | 0.5137 | 0.9516 |
| Collider XNOR mechanism | 0.8233 | 0.9149 |
| Collider NAND mechanism | 0.8265 | 0.9516 |
| Chain, second-edge mechanism inverted | 0.8265 | 0.8265 |
| Fork, both child mechanisms inverted | 0.9241 | 0.9241 |
| Reversed-chain topology | 0.9123 | 0.9852 |

The aggregate gain is real for this test grid, but family-specific: it comes mainly from collider variants and the reversed-chain case. It does not improve the two shown chain/fork mechanism-shift cases. A universal “OOD detector” conclusion would be unjustified.

## Reference-bank sensitivity and implementation hardening

The broad bank changes per-action JSD values, and the tests assert the full list of models contributes to the predictive mixture. Yet at this budget, coefficient and action set, broad- versus narrow-bank evaluation produced identical reported policy metrics across the grid. This indicates that simply adding reference models does not guarantee different selected actions or better detection; the action ranking and coefficient matter more than bank size alone in this fixture.

During the exploratory implementation, I caught an important generalization bug: an earlier helper paired a reference list with only two weights, which would silently truncate a larger bank through \`zip\`. The committed stress-test implementation instead validates matching lengths and assigns weights to every provided reference model. A regression test verifies that broad and narrow banks yield different predictive mixtures. The earlier Pareto pilot uses a two-model reference bank, so the bug did not alter its declared runs; this follow-up is where generalized bank size is exercised.

## Literature and industry comparison

- **CausalArena (September 2026)** calls for synthetic, semantic-operational and formula-grounded SCMs plus real-data checks, noting that method rankings shift across families. This six-world hand-built grid is substantially narrower. https://arxiv.org/abs/2609.11897
- **ACDB (September 2026)** offers a common observe/intervene/submit protocol and separates graph skeleton, directed structure and intervention efficiency. We have not run its official protocol here, and our AUC must not be compared with its graph-recovery scores. https://arxiv.org/abs/2609.31675
- **Causal Discovery under Off-Target Interventions (AISTATS 2024)** explicitly models interventions that act on a random subset of variables rather than assuming perfect targeting. The current compliance model is a much simpler single-target compliance variable and should eventually be compared against richer off-target models. https://proceedings.mlr.press/v238/choo24a.html
- **MetaCaDI (UAI 2026)** addresses unknown intervention targets across environments and few-shot adaptation. This highlights that intervention-target uncertainty is existing research territory; our contribution here is a small stress-test integration with active EIG/JSD policy evaluation, not the claim to have invented unknown-target discovery. https://proceedings.mlr.press/v337/ong26a.html
- **TCD-Arena (ICLR 2026)** studies time-series causal discovery under 33 assumption violations and reports results over roughly 30 million discovery attempts. Our evaluation is nowhere near that breadth. https://tcd-arena.github.io/
- **Since Faithfulness Fails (ICML 2025)** finds significant performance limitations under faithfulness violations. The next test suite must add cancellation/non-faithfulness and latent-confounding cases rather than focusing mostly on Boolean logic variants. https://proceedings.mlr.press/v267/olko25a.html

## Conclusions

1. EIG + a JSD reference objective can improve surprise-AUC in selected worlds where the reference models are informative and intervention semantics are correctly modeled.
2. The improvement is not uniform across topology/mechanism families.
3. A bigger reference bank changed predictive-divergence values but did not change the tested action outcomes at this coefficient; this itself is a negative result for the assumption that bank breadth automatically adds value.
4. Incorrect intervention compliance causes a clear trade-off regression: more zero-support mass and lower usable graph information. A robust policy must estimate, marginalize or constrain over action validity before optimizing OOD objectives.
5. No claim of SOTA, external benchmark equivalence, or real-world detection power is made.

## Next gates

- Replace hand-built reference banks with predeclared generated reference ensembles; hold out complete graph families and mechanism families, not only truth tables.
- Add unknown-target and multi-target/off-target interventions informed by the relevant literature.
- Add unfaithful mechanisms, latent confounding, measurement error and nonstationarity.
- Compare at equal data/action budgets with a verified ACDB baseline and assumption-appropriate PC/FCI/GES/IGSP variants.
- Report graph SHD, skeleton/directed precision-recall, OOD AUROC/AUPRC, false alarms at a calibrated 5% threshold, power at specified OOD severities, zero-support mass, claim coverage, and paired intervals over independently seeded worlds.
- Keep the primary claim falsifiable: require a predeclared improvement with paired uncertainty excluding zero, without degrading the intervention-validity endpoint.

## Reproduction and provenance

From the repository root:

    python3 validation/open_world_reference_stress.py --self-test
    python3 -m unittest -v validation/test_open_world_reference_stress.py
    python3 validation/open_world_reference_stress.py --source-git-blob fbb4e3ec45ec896e48bcdb90e2ac3ac3ce92b7fd --output /tmp/open-world-reference-stress.json

- Source blob: \`fbb4e3ec45ec896e48bcdb90e2ac3ac3ce92b7fd\`; SHA-256 \`aaf4f71b7594e4ed4f4bd557879b534101314058c7ac725fc5f5fe00e745d8b7\`; 12,492 bytes.
- Test blob: \`b58d7f4f177b5809e5a96cb188700da334906e35\`; SHA-256 \`a0b12e4d11ab191c2e843c8ed99339b6ae78854607b58222bc8994fdd8f09601\`; 1,664 bytes; 5 tests.
- Embedded self-test passed; 5/5 unit tests passed locally against these exact source/test bytes.
- Full 54-row artifact: 30,061 bytes; SHA-256 \`e6d71190cc92e173831cfcf974d5f0028723c223807c6989272efbeaf6e21d99\`; the artifact binds the source blob above.
- This is local validation; no GitHub Actions pass is inferred.

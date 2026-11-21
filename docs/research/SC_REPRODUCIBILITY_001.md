# SC-REPRODUCIBILITY-001 — specimen, batch and laboratory reproducibility

**Tracking issue:** [Symthaea #13078](https://github.com/Luminous-Dynamics/symthaea/issues/13078)  
**Initial implementation:** [research/sc-reproducibility-v0.1](https://github.com/Luminous-Dynamics/symthaea/tree/research/sc-reproducibility-v0.1)  
**Status:** synthetic split-design auditor; no real variance decomposition or material replication is claimed.

## Why this is a separate benchmark

Clark & Stewart, *The challenge of synthesis dependence in quantum materials*, Nature Physics 22, 1386–1393 (published 1 September 2026), describes synthesis dependence as a major source of sample-to-sample variation, unreproducible quantum materials and disagreement about measured properties:
https://www.nature.com/articles/s41567-026-03417-x

The 2026 PNAS programmatic perspective argues that the field must connect theoretical prediction, synthesis, materials engineering and experimental characterization in a continuous feedback loop:
https://doi.org/10.1073/pnas.2520324123

Symthaea already owns the protocol-to-phase-to-property representation in #12671 and condition-bound observation / synthesis records in #12618. This issue does not create another material or sample ontology. It adds the **experimental-design and evaluation layer** needed to distinguish:
- repeatability of one specimen on one instrument;
- instrument transfer on a matched specimen;
- variation among specimens from one batch;
- between-batch variation;
- between-campaign / process transfer;
- inter-laboratory reproducibility;
- protocol transfer;
- temporal / aging behaviour.

These are different estimands. They cannot be collapsed into one “reproducibility score.”

## Variance structure

A future statistical model may use a hierarchical representation such as:

`y = mu + u_batch + u_campaign + u_lab + u_instrument + u_protocol + u_specimen + interactions + residual`

The terms are not automatically identifiable. Campaign may be nested in batch, instrument may be nested in lab, and specimens may be nested in batches. If the experiment never crosses laboratories or instruments, the corresponding variance components may be confounded. The design must determine what can be estimated before a model is selected.

Do not use a small synthetic fixture to claim variance-component recovery. The first executable slice only verifies whether split definitions preserve the intended specimen / realized-state / batch / campaign / lab / instrument grouping.

## Realized state identity

Keep nominal material family and realized state separate. Independent specimens or batches of the same intended material can have different oxygen content, phase fraction, strain, texture, defect population or metastable history. Each observation therefore binds the target `material_id` plus a realized `material_state_id`. Same-specimen repeatability and paired instrument / inter-lab transfer require matching state IDs; between-batch and between-campaign generalization explicitly permit different realized states.

## Current executable contract

Schema:
- `data/schemas/sc_reproducibility_design_v0_1.schema.json`

Synthetic fixture:
- `tools/superconductivity-discovery-harness/fixtures/reproducibility_hierarchy_smoke_v0_1.json`

Auditor:
- `tools/superconductivity-discovery-harness/src/bin/reproducibility_hierarchy_audit.rs`

Supported tasks:
1. `same_specimen_repeatability`: same specimen, batch, campaign, lab, instrument, protocol and condition; distinct repeat index.
2. `instrument_transfer`: same specimen and matched batch / campaign / lab / protocol / condition, with disjoint instruments and an explicit paired-specimen declaration.
3. `within_batch_specimen_holdout`: different specimens from the same batch and campaign.
4. `between_batch_holdout`: disjoint specimens and batches.
5. `between_campaign_holdout`: disjoint specimens, batches and campaigns.
6. `inter_lab_transfer`: disjoint laboratories; either explicitly paired same-specimen transfer or unpaired specimen holdout.
7. `protocol_transfer`: disjoint protocol identities.
8. `time_forward`: all training timestamps precede every test timestamp, and specimens / batches are disjoint.

The auditor validates the declared schema, observation and task identities, split membership, group constraints and canonical UTC time ordering. It emits SHA-256 receipts for the exact schema and fixture.

## Important restrictions

- A same-specimen repeatability task is allowed to reuse the same specimen by design. That is not an acceptable generalization split.
- Instrument transfer may use the same specimen only when explicitly declared as paired and conditions / state / protocol are matched.
- Inter-lab paired transfer is distinct from generalization to new samples in an unseen lab.
- Time-forward tasks must not leak specimens or batches from the training window into the future test cohort.
- Repeated measurements from the same physical sample or re-publication of one sample's data are not independent replication.
- A property comparison is admissible only for the same declared property and unit. Full condition-specific scientific equivalence remains an upstream evidence / measurement concern and must be audited before statistical fitting.
- The current UTC timestamp validator accepts normalized `YYYY-MM-DDTHH:MM:SSZ` and validates calendar ranges; timezone offsets / sub-second forms should be normalized by a reviewed ingestion layer, not guessed here.

## Next statistical benchmark phase

After a rights-cleared cohort is admitted, compare at matched data and tuning budgets:
- pooled mean / standard deviation (descriptive baseline only);
- nested mixed-effects model;
- Bayesian hierarchical model with prior sensitivity and posterior predictive checks;
- robust heavy-tailed model where justified;
- state-aware model using measured stoichiometry / phase fractions / defects / strain / texture / oxygen content;
- protocol-to-phase-to-property model from #12671;
- a deliberately naive model that ignores specimen / batch / lab structure.

Report property-specific variance estimates and confidence / credible intervals, within- and between-lab agreement, predictive interval coverage / width, held-out batch / lab performance, false candidate promotion / exclusion, and decision value of additional replicates. Correlation alone is not agreement; methods require paired comparisons and limits-of-agreement analysis.

Do not report an ICC or variance ratio if the design is not crossed / replicated enough to identify it. Do not impute missing laboratory / batch levels. Unknown remains unknown.

## Prospective target ladder

After determining expected variance and the measurement-system capability, preregister one selected REBCO tape property profile or one bulk superconducting material family:
- repeatability from repeated same-specimen acquisitions;
- within-batch specimen variance;
- independent batch and campaign variance;
- inter-laboratory transfer using either paired sister specimens or explicitly transferred same specimens;
- a held-out batch / lab prediction assessment.

A minimum pilot with two batches and two labs can reveal design failures but is not enough to establish general reproducibility. Sample size and power must be based on the target variance components, effect sizes, clustering and resource budget.

## Ownership and interoperability

- #12618: canonical material / specimen / process / observation semantics.
- #12671: planned protocol -> executed process -> realized phase / microstructure -> property.
- #12677: pairing vs global phase coherence and same-specimen multimodal evidence.
- #12672: pressure-quench metastability and thermal-aging behaviour.
- #12966: interventions that change specimen state.
- #11736: REBCO conductor-to-coil physical engineering.
- Symtropy #6522 / #6550 / #6577: REBCO process twin, long-length spatial quality and direct-vs-proxy metrology qualification.
- Symtropy #2995 / #3121: measurement-system uncertainty and manufacturing VVUQ.
- Mycelix #6426: signed append-only event / review / independent replication attestations.

## Claim ceiling

The current executable checks **group-split and reproducibility-design semantics only**. It does not estimate variance components, identify causal synthesis effects, prove inter-laboratory agreement, reproduce a material, establish a superconducting transition, or claim manufacturing readiness.

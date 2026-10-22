# First-Aid Clinical Evidence Provenance Gate v0.1

**Status:** research/qualification contract. This is not a medical protocol, clinical decision aid, clinical validation, or deployment authorization.

## Why this tranche exists

A separate software interpreter can reveal implementation disagreement, but two programs can still agree on an incorrect medical requirement. The next boundary is evidence provenance: each future reference-evaluation claim must be traceable to a specific authoritative source and reviewed for the intended population, setting, and jurisdiction by people independent of the implementation.

This tranche registers source metadata only. It deliberately contains **zero clinical recommendation claims** and copies no guideline passages or treatment algorithms.

## Source registry at this revision

The initial catalog records the IFRC *International First Aid, Resuscitation and Education Guidelines 2025*, the European Resuscitation Council / Resuscitation Council UK 2025 first-aid guidance, the American Heart Association 2025 CPR/ECC guidelines, and a 2026 AED-specific notice published by the Resuscitation Council of Southern Africa (RCSA). These have different scopes and jurisdictions; they are not interchangeable:

- IFRC describes global evidence-based principles and explicitly expects adaptation to local cultural, environmental, legal, resource, and population context.
- ERC/RCUK is a regional first-aid guideline source.
- AHA's 2025 CPR/ECC publication is a resuscitation/cardiovascular-care source, not a universal source for every first-aid topic.

Official source pages are linked in `clinical-evidence-sources-v0.json`. URLs and edition metadata were checked on 2026-10-22. The RCSA notice is regional context for an AED/resuscitation subtopic, not a comprehensive local first-aid protocol. The registry does not copy or redistribute the source documents. Before ingesting guideline text or reproducing tables/algorithms, verify copyright, licensing and permission separately.

## Claim-admission contract

The registry must fail closed if its schema, source identifiers, edition metadata, canonical HTTPS URLs, or review policy are invalid. V0 also allowlists exact publisher identities and official URL host prefixes; a new source ID requires explicit code/provenance review, and unknown fields at any registry layer are rejected. A future claim intended for reference evaluation must have:

1. A stable claim ID and topic, intended population, setting, target jurisdiction, locale, and source edition.
2. An exact source locator (section/table/page/figure or stable heading), with no unsupported paraphrase silently promoted to a ground-truth label.
3. A statement of limitations, exclusions, relevant prerequisites, and what the source does **not** establish.
4. At least two independent reviewers with recorded qualifications and independent decisions, plus a separate qualified adjudicator for disagreements or final approval.
5. Explicit jurisdiction-localization review. A global or regional source alone does not establish that an action is appropriate in South Africa or any other target jurisdiction.
6. A review date and a source-update trigger. When source edition or localization changes, affected claims must become stale until re-reviewed.

The reviewer fields should use pseudonymous IDs or role attestations; do not put private reviewer contact details or patient data in the public repository.

For v0, each atomic claim must cite exactly one primary source ID and its registered edition; split cross-source synthesis into separately traceable atomic claims instead of hiding multiple editions behind one locator. The claim's source edition must exactly match the registry entry. The only admissible approval in v0 is `approved_for_reference_evaluation_only`. That status may make a case eligible for a software reference benchmark; it does **not** authorize instructions to real people, a clinical product, patient care, or clinical deployment. This gate is evidence bookkeeping, not a clinical reasoning engine.

## Fail-closed requirements

- Zero adjudicated claims means the receipt must say `BLOCKED_NO_ADJUDICATED_CLAIMS`.
- A malformed registry is rejected, not reported as a clean zero-case pass.
- A source being official or current does not imply that every claim drawn from it is reviewed.
- A passing metadata test is not a clinical-quality score, sensitivity estimate, or safety claim.
- Public fixtures are not blind holdouts. A credible blind evaluation requires an independent custodian, pre-committed scoring code, controlled access, and a versioned commit-then-reveal process.
- Clinical correctness, readability under stress, accessibility, local emergency access, and human outcomes require separate expert and empirical evaluation.

## Benchmark progression

| Stage | Evidence required | Claim ceiling |
|---|---|---|
| 0. Registry integrity | Schema-valid source metadata; stable digests | Source metadata is internally consistent |
| 1. Expert-adjudicated cases | Qualified independent reviewers; source locators; resolved disagreements | Agreement with reviewed reference cases |
| 2. Independent blind set | Held-out cases under independent custody; frozen score code | Generalization to that held-out distribution only |
| 3. Human-factors evaluation | Ethics-reviewed protocol as applicable; representative participants; comprehension/time/error measures | Measured performance in the tested setting |
| 4. Clinical/operational evaluation | Applicable clinical, legal, safety and regulatory governance | Only the specific evidence-supported claims authorized by the responsible parties |

Never combine these stages into one “accuracy” number. In particular, software-contract agreement must stay separate from medical correctness and real-world outcomes.

## Future performance targets (provisional gates, not current results)

- **100%** of admitted reference claims have versioned authoritative citations, exact locators, population/context/jurisdiction and limitation fields.
- **100%** of approved claims have two independent qualified reviews and the required adjudication record.
- **0** claims with stale, unknown-edition, or non-localized evidence can enter an evaluation set as approved.
- **0** invalid registries are silently scored as zero violations.
- Report reviewer agreement and adjudication rate; do not optimize agreement by suppressing hard disagreements.
- Evaluate sensitivity to guideline-version changes and localization conflicts with explicit counterfactual fixtures.

These are process-integrity acceptance gates. They do not imply a target medical error rate and must not be represented as patient-safety guarantees.

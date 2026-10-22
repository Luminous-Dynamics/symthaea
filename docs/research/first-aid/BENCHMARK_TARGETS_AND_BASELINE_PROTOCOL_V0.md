# First-Aid Workflow Benchmark: Targets and Fair-Baseline Protocol v0.1

**Status:** evaluation-design proposal only. This document contains no treatment instructions and does not authorize clinical use.

## 1. Why benchmark the work, not the answer?

The current clinical-AI benchmark landscape distinguishes exam questions from tasks performed across documents, software, time, context, and workflow. MedHELM organizes medical work into 5 categories, 22 subcategories, and 121 tasks, with 37 evaluations across the taxonomy. Its report shows materially different model performance across categories; the Stanford summary of the results reports lower win rates for administration/workflow than for several other categories. This makes task selection and evaluation protocol important; it does **not** make MedHELM scores directly comparable to our synthetic first-aid process checker.

Reference:
- Stanford HAI, *Measuring What Medical AI Can Actually Do*, published 2026-10-19: https://hai.stanford.edu/news/measuring-what-medical-ai-can-actually-do
- Nature Medicine, *Holistic evaluation of large language models for medical tasks with MedHELM*: https://www.nature.com/articles/s41591-025-04151-2

The 2025 IFRC First Aid, Resuscitation and Education Guidelines are an evidence source, not a universally self-executing protocol. The ERC/RCUK 2025 first-aid guidance, AHA 2025 CPR/ECC guidance, and RCSA's 2026 AED-specific notice have different scope and jurisdiction relevance. Their presence in a source registry is not adjudication, and a regional resuscitation notice does not stand in for comprehensive South African first-aid guidance.

Source registry links:
- IFRC 2025: https://www.ifrc.org/document/ifrc-international-first-aid-resuscitation-and-education-guidelines-2025
- ERC/RCUK First Aid 2025: https://www.resus.org.uk/professional-library/2025-resuscitation-guidelines/first-aid-guidelines
- AHA 2025 CPR/ECC: https://professional.heart.org/en/science-news/2025-aha-guidelines-for-cpr-and-ecc
- RCSA notice, 2025 ECC guideline impact on AEDs, 2026-04-10: https://resus.co.za/back-office/uploads/pdf/pdf_20260413_102704.pdf

## 2. Benchmark stages and separate scoreboards

Do not publish one blended "medical accuracy" or "safety" score. Report at least these five scoreboards separately:

1. **Registry integrity** — schema acceptance/rejection, source/edition binding, duplicate IDs, stale evidence, malformed records and correct fail-closed disposition.
2. **Software contract conformance** — ordered trace outcomes, state-transition violations, stale-bundle rejection, uncertainty preservation, consent/data-minimization process, handoff lifecycle.
3. **Expert-adjudicated reference agreement** — exact agreement with independently authored, edition- and jurisdiction-scoped cases. Not yet available.
4. **Human factors** — time to locate a relevant aid, comprehension, critical omission, conflicting-instruction rate, handoff completeness, accessibility, and recovery when the user is uncertain.
5. **Operational/clinical outcomes** — only under the governance, ethics, safety, legal and clinical oversight appropriate for that study and setting. These are not inferable from the four earlier scoreboards.

Keep technical-process correctness and clinical correctness distinct. A perfect checker over an erroneous requirement remains wrong; an expert-correct recommendation that is inaccessible or unusable under stress can still fail the intended task.

## 3. Provisional acceptance targets

These are engineering targets for future test design, **not achieved measurements, medical thresholds, or safety guarantees**.

| Measure | Proposed target | Interpretation |
|---|---|---|
| Malformed/unknown registry or trace input | 100% explicitly rejected in the declared mutation suite | Rejection is distinct from a clean zero-finding score |
| Unreviewed, stale, wrong-edition, or non-localized claim admission | 0 false admissions in the frozen suite | Metadata rule, not medical truth |
| Approved-case source traceability | 100% with exact primary-source locator, edition, population, setting, jurisdiction, limitations, review date and update trigger | All fields must be verifiable by reviewers |
| Independent approval separation | 100% of admitted claims have two distinct independent qualified reviewers plus a separate qualified adjudicator | Identity/role attestations must be independently auditable; JSON booleans alone are not sufficient proof |
| Synthetic workflow contract | 0 expected-vs-actual outcome mismatches on frozen tests and deterministic mutations | Agreement with software contract only |
| Critical unsafe output | 0 observed in the pre-registered adjudicated/holdout set before considering a human-facing trial | A zero count is not proof of zero risk; report denominator, sampling design, confidence bounds and limitations |
| Staleness after source update | 100% of claims affected by a declared edition/update trigger become ineligible until review | No automatic carry-forward across editions |
| Human task performance | Non-inferior to the existing reviewed aid on all predeclared critical-error measures; seek improvements in completion time and comprehension only after non-inferiority gates pass | Compare on same population, scenarios, language and access conditions |

For the clinical claim and human-factors rows, do not set final thresholds, calculate pass rates, or recruit participants until a qualified review group has defined the population, scenario taxonomy, harm categories, sample size, ethics/consent requirements, and statistical analysis plan. Rare high-severity failures need more than an average accuracy score.

## 4. Fair comparison with industry baselines

Compare systems on the **same frozen task set and evidence edition**, with the same input visibility, time budget, tool access, allowed resources and scoring rules. Register the candidate and baselines before unblinding. Report:

- task-stratified exact agreement and critical-error counts;
- abstention/reject rate, including false rejects and unsafe acceptances;
- time and resource use per completed task;
- uncertainty calibration and whether uncertainty leads to the expected safe disposition;
- performance broken down by language, accessibility needs, setting, connectivity, available equipment and relevant population;
- paired deltas with confidence intervals, not only aggregate averages;
- source-version sensitivity, contamination controls and performance on a held-out set.

Baselines should include the existing reviewed cognitive aid / current workflow and qualified human participants under an approved protocol where appropriate. LLM judges may assist with rubric scoring only after their agreement with qualified human raters is measured; no model should be its own sole ground-truth judge. Report reviewer disagreement and adjudication rate, rather than discarding hard cases.

MedHELM is a landscape reference for breadth and task-based evaluation, **not an apples-to-apples benchmark for this project**. The First-Aid process-conformance suite does not establish that Symthaea outperforms MedHELM models. A valid superiority claim requires a task-level matched experiment and an independently curated holdout.

## 5. Holdout and governance design

- Public development cases support debugging; they are not blind generalization evidence.
- An independent custodian maintains the held-out set, records its digest before testing, and does not contribute to the tested implementation.
- Freeze model/software revision, source versions, policy configuration, evaluation code, scoring rubric and time/resource budgets before the run.
- Commit the benchmark identity and scorer digest before test access; reveal labels only after candidate outputs are frozen.
- Preserve the full denominator, rejections, missing results and protocol deviations. Never silently drop a difficult case.
- Re-review claims when a source edition, jurisdictional interpretation, emergency-system assumption, localization or intended population changes.
- Do not publish patient-identifiable data or non-authorized source text in the public fixtures.

## 6. Targeted paradigm shifts

1. **Evidence-versioned workflow:** every claim is tied to an edition and exact source locator; updates invalidate affected claims.
2. **Abstention as a valid outcome:** a system earns credit for detecting uncertainty and handing off correctly, rather than being forced to produce an answer.
3. **Temporal correctness:** test transitions and source changes across an episode, not merely individual prompts.
4. **Independent oracle + blinded evaluation:** diversify implementation and separate evaluators from builders; independent ground truth still requires competent human adjudication.
5. **Outcome-aware evaluation:** measure actual task completion and comprehension only when authorized; never use synthetic agreement as a substitute.
6. **Local-first adaptation:** jurisdiction, language, accessibility, connectivity, training level and available equipment are explicit dimensions, not hidden assumptions.

## 7. Current state

The evidence registry currently contains source metadata only and zero admitted claims. The required disposition is `BLOCKED_NO_ADJUDICATED_CLAIMS`. The synthetic trace-oracle work is also separate and pending CI. No human factors, clinical correctness, patient benefit, emergency reachability, regulatory conformity, or deployment result has been measured by these PRs.

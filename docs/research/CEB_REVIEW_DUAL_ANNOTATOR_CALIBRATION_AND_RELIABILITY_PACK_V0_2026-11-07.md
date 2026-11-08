# CEB Review Dual-Annotator Calibration and Reliability Pack V0 — 2026-11-07

**Status:** Development-only reviewer operations pack. It contains no completed human annotations, no reliability estimate, no admitted benchmark tasks, no sealed holdout, and no candidate-performance result.  
**Owner:** Existing CEB-REVIEW / CEB-HUMAN / causal-diagnosis programme; #10723, #10367 and #10419.  
**Composes with:** [Causal-diagnosis preregistration](CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md), [counterfactual-control qualification](CEB_REVIEW_COUNTERFACTUAL_CONTROL_QUALIFICATION_V0_2026-11-07.md), [finding adjudication card](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md), and [human calibration pack](CEB_REVIEW_HUMAN_CALIBRATION_PACK_V0_2026-11-07.md).  
**Primary CEB-REVIEW measures:** unchanged — defect Recall@5, actionable precision, reviewer burden.  
**Execution/regrade boundaries:** #10064 / #10369. Task admission and custody remain under #10607 / #10609.

## 1. Purpose and precise gap

The causal-diagnosis preregistration already requires two independent blinded human reviewers and evidence-based resolution by a third expert. The adjudication card already captures the causal mechanism, source authority, freshness, utility and P/I/F/N/M/IRR control qualifications.

This pack makes that requirement operational. It specifies how to issue the same development cases independently to Reviewer 1 and Reviewer 2, preserve their first-pass labels, measure disagreement before adjudication, classify disagreement causes, and freeze the Phase A analysis plan without silently converting consensus into truth.

It is not another task schema, metric lane, grader, evidence-authority class, or execution permission. It is a human annotation procedure and a blank worksheet contract.

## 2. Development materials and their limits

Use the existing public [Human Calibration Pack V0](CEB_REVIEW_HUMAN_CALIBRATION_PACK_V0_2026-11-07.md):

- CAL-01: authorization-ordering case, variants X1–X6.
- CAL-02: untrusted-length allocation case, variants X1–X6.
- CAL-03: tenant-scoped authorization-cache case, variants X1–X6.
- Total: 18 public development variants spanning P/I/F/N/M/IRR.

The separately published [calibration key](CEB_REVIEW_HUMAN_CALIBRATION_KEY_V0_2026-11-07.md) must be withheld from both reviewers until their first-pass annotations are locked. **The key is public in the repository, so withholding it is an instruction-based exercise, not secure blinding.** Do not call this a sealed or contamination-resistant test.

The 18 variants are nested within only three source anchors. They are appropriate for rubric walkthrough, vocabulary calibration and discovery of ambiguous instructions. They are not 18 independent defect samples and must not be used to claim confirmatory inter-rater reliability, reviewer generalization, benchmark gold completeness, or model performance. No human has annotated these cases as part of this commit.

The 32 ERG scenarios remain public development examples. They may be used to rehearse the finding-adjudication workflow, but neither the ERG cases nor the CAL variants may later be relabeled as sealed holdout material.

## 3. Roles and separation

| Role | May see before first-pass lock | Must not see before first-pass lock |
|---|---|---|
| Coordinator | Full pack, key, mapping, rubric, both blank forms | Nothing; coordinator is not a blinded annotator |
| Reviewer 1 | Rubric, assigned case text and evidence, own blank form | Reviewer 2 labels, answer key, candidate/model identity, other arm outputs |
| Reviewer 2 | Same evidence and rubric, independently randomized assignment and own blank form | Reviewer 1 labels, answer key, candidate/model identity, other arm outputs |
| Third adjudicator | Both locked first-pass annotations, source evidence, rubric, then key | Must not alter the original annotations |
| Candidate/model operator | Frozen task view and candidate contract only | Human labels, key, adjudication notes, protected task custody |

The two reviewers should have relevant software-review experience. A security task requires a reviewer competent in the applicable security boundary and vulnerability class. Record role, relevant experience, conflicts, and any prior exposure to the case. Do not collect unnecessary personal information.

The coordinator creates independent anonymous item IDs and a different randomized order for each reviewer. The coordinator retains the item-to-CAL mapping. Reviewers must not discuss cases, share notes or inspect the other form until both first-pass files are locked.

## 4. Operating procedure

### Step 0 — Freeze the annotation packet

Before either reviewer starts, record:

- exact commit/tree or document version for every source artifact;
- SHA-256 digest of the pack, key, rubric, evidence bundle and blank form;
- rubric version and date;
- reviewer instructions and permitted references;
- anonymous item mapping and independent item order;
- whether the coordinator has withheld the public key;
- any known prior exposure or blinding limitation.

A change to a source artifact, rubric or case after annotation begins creates a new packet revision. Do not silently replace the material.

### Step 1 — Independent orientation

Provide both reviewers the six-state definitions and the causal-chain rubric below. Explain that a nearby line, plausible comment, matching symptom or correct file is not a correct mechanism diagnosis by itself.

Do not show the answer key. Do not teach the cases by revealing their labels. Reviewers may ask procedural questions; answer them identically and log any clarification that changes interpretation.

### Step 2 — Independent first pass

Each reviewer annotates all assigned variants separately. For every variant:

1. classify the source state or select unresolved;
2. decide whether the target mechanism is live, absent or indeterminate;
3. assess each causal link independently;
4. for CTG cases, add the topology worksheet in Section 6.1: typed nodes/edges, AND/OR gates, temporal constraints, sufficient paths and residual-path predictions;
5. distinguish the target from the nearest alternative mechanism;
6. assess the candidate's counterfactual prediction, where one is supplied;
7. identify evidence authority and freshness;
8. assign a disposition, confidence and elapsed annotation time;
9. cite the smallest source/evidence anchors that support the decision.

Reviewers must record uncertainty rather than infer missing evidence. A reviewer may use the existing evidence classes only: source_declared, syntax_derived, compiler_resolved, runtime_observed, intervention_observed, and inferred/unresolved. The first-pass exercise is source-only; it cannot create runtime or intervention observations.

### Step 3 — Lock and preserve

Each reviewer returns a complete form with a completion timestamp and SHA-256 digest. The coordinator confirms both forms are locked before revealing either form to the other reviewer or releasing the answer key.

Never overwrite a first-pass annotation. Corrections after lock are append-only amendments with the old value, new value, reason, author and timestamp retained.

### Step 4 — Compute disagreement before adjudication

Produce, at minimum:

- exact-state agreement and the full 6 × 6 state confusion matrix;
- target-mechanism present/absent/indeterminate agreement;
- agreement for causal-diagnosis disposition;
- disagreement counts for each causal-chain link;
- missing/uncertain rates and confidence distributions;
- per-case and per-variant disagreement listings;
- reviewer time and any protocol deviations.

Do not calculate reliability on adjudicated or consensus labels. Preserve the two original ratings as the only inputs to first-pass inter-rater statistics.

### Step 5 — Adjudicate disagreements

The third adjudicator reviews each disagreement and any high-severity or unresolved item. They must record the evidence and rule used to resolve it. Use these disagreement causes:

- **STATE_CONFUSION:** P/I/F/N/M/IRR semantics were misunderstood or confused.
- **CHAIN_LINK:** reviewers disagree on trigger, path/state, operation, invariant or consequence.
- **ALTERNATIVE_MECHANISM:** same location/symptom, but different mechanism identity.
- **CONTROL_VALIDITY:** the source variant does not qualify for the causal claim under Q1–Q6.
- **EVIDENCE_AUTHORITY:** reviewers assign different authority classes or overstate what evidence proves.
- **FRESHNESS_OR_BINDING:** source, configuration or observation is stale, superseded or not bound to the reviewed revision.
- **RUBRIC_AMBIGUITY:** the rule admits multiple reasonable interpretations.
- **KEY_OR_CASE_DEFECT:** the published expected answer or synthetic case is wrong or incomplete.
- **UNRESOLVED:** available evidence cannot decide the dispute.

Do not settle disagreements by majority vote or by asking the same model family that generated the candidate finding to adjudicate itself. If the key or case is defective, quarantine that item, document the correction and issue a new packet revision. Preserve a dissent when the evidence does not resolve it.

### Step 6 — Debrief and rubric revision

Only after first-pass ratings are preserved may reviewers compare interpretations and inspect the public key. Categorize every disagreement as a reviewer error, unclear rubric, invalid control, incomplete evidence, or key defect.

Clarify the rubric where necessary. If definitions change, increment the rubric version and repeat the affected calibration using newly randomized copies. Do not retroactively rewrite the original first-pass agreement results.

## 5. Six-state annotation rules

| State | Target mechanism | Required interpretation | Common false inference |
|---|---|---|---|
| P — Parent | Absent | The target has not yet been introduced | Treating later knowledge as proof it is present |
| I — Introducing | Present | The historical change makes the mechanism live | Matching the file without tracing the cause |
| F — Fixed | Removed by the qualified repair | The specific target is no longer live | Declaring the entire repaired PR/program defect-free |
| N — Neighboring mechanism | Target absent; another cause may exist | Similar symptom/location does not make it the same cause | Calling a neighboring defect a target hit |
| M — Minimal causal intervention | Necessary target cause disabled | The target should disappear while unrelated context is preserved | Assuming the edit is minimal merely because few lines changed |
| IRR — Irrelevant intervention | Target remains present | The intervention leaves the necessary cause intact | Predicting the target disappears because nearby code changed |

IRR is a positive persistence control: the same target mechanism should remain live. Do not group it with P/F/N/M. N is not automatically a clean negative; it may contain a different valid defect.

A historical commit is not automatically a qualified intervention. Complete the existing Q1–Q6 control-qualification worksheet before using a source state to support a causal claim. If the state is not qualified, annotate the control as limited or unresolved and narrow the claim.

### Causal-chain scoring

Record each link separately:

| Link | Question | Allowed first-pass label |
|---|---|---|
| Trigger / precondition | What must be true for the behavior to occur? | supported / partial / contradicted / unresolved / not applicable |
| Path / state transition | How does the triggering condition reach the operation? | supported / partial / contradicted / unresolved / not applicable |
| Operation | What exact operation or decision is incorrect/unsafe? | supported / partial / contradicted / unresolved / not applicable |
| Violated invariant | Which requirement, authorization rule or program invariant is broken? | supported / partial / contradicted / unresolved / not applicable |
| Consequence | What follows under the stated conditions, and with what scope? | supported / partial / contradicted / unresolved / not applicable |

A target diagnosis is not correct solely because the reviewer identifies the right file, function, line or vulnerability family. The causal identity and relevant conditions must be right. The consequence must not exceed the evidence.

## 6. Blank annotation schema

Create one row per anonymous variant, per reviewer. Use the following columns without changing their meaning between reviewers:

| Field | Required content |
|---|---|
| packet_id | Frozen packet identifier |
| reviewer_id | R1 or R2; use pseudonymous study code in retained data |
| anonymous_item_id | Coordinator-assigned randomized item ID |
| case_anchor_id | CAL anchor, kept from the reviewer if needed for clustering |
| variant_id | Coordinator-only mapping; omit from reviewer-facing export if it reveals state |
| source_state_label | P / I / F / N / M / IRR / unresolved |
| target_mechanism_state | present / absent / indeterminate |
| trigger_label | supported / partial / contradicted / unresolved / N/A |
| path_label | supported / partial / contradicted / unresolved / N/A |
| operation_label | supported / partial / contradicted / unresolved / N/A |
| invariant_label | supported / partial / contradicted / unresolved / N/A |
| consequence_label | supported / partial / contradicted / unresolved / N/A |
| alternative_mechanism | distinguished / conflated / absent / unresolved |
| causal_diagnosis | correct / wrong / incomplete / unresolved |
| counterfactual_prediction | consistent / inconsistent / incomplete / not testable / not supplied |
| evidence_authority | one of the six existing CEB evidence classes |
| evidence_freshness | current / stale / superseded / unbound / unresolved |
| disposition | confirm target / reject target / valid non-target / abstain / escalate |
| confidence | low / medium / high |
| rationale_and_anchors | concise explanation and exact source references |
| elapsed_minutes | active annotation time, excluding unrelated breaks |
| blinding_deviation | none / possible / confirmed, with explanation |

The coordinator-only mapping is not part of the reviewer-facing export. Retain missing values as missing; do not silently convert them to a negative label.


### 6.1 Causal-topology addendum (CTG cases only)

Use this addendum only for the six cases in [Causal-Topology Development Challenge V0](CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_V0_2026-11-07.md). It extends the annotation of the mechanism; it does not create a new task schema, receipt, evaluator or primary metric.

| Field | Reviewer 1 / Reviewer 2 entry |
|---|---|
| Target outcome and frozen invariant | |
| Node inventory (ID, kind, semantic statement, source anchor) | |
| Edge inventory (from, to, type, condition, source anchor, authority class) | |
| AND gates and jointly required conditions | |
| OR gates and alternative sufficient paths | |
| Temporal partial order / concurrency / unknown ordering | |
| Minimal sufficient path or condition sets | |
| Nearest alternative mechanism | |
| Qualified intervention set(s) | |
| Residual target paths after each intervention | |
| Declared context boundary and unobserved remainder | |
| Topology disposition | correct / wrong / incomplete / unresolved |
| Main topology disagreement cause | missing path / wrong gate / wrong order / unsupported edge / residual path / scope / other |

Do not require the same node names or an identical drawing. Reviewers must identify semantic equivalence, not reproduce a canonical diagram. A call-graph or shared-file edge is not causal evidence by itself. If the source package does not bound all relevant callers/configurations, mark the open-world remainder unknown.

For CTG calibration, compare first-pass graphs before either reviewer sees the other or the key. Preserve the independent graphs and edge-level rationale; never replace them with a consensus graph. The third adjudicator records the accepted semantic path/gate set and the evidence that resolves each disagreement.


### 6.2 Causal-identifiability addendum (CID cases only)

Use this addendum for the six cases in [Causal-Identifiability Development Challenge V0](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_CHALLENGE_V0_2026-11-07.md). Each reviewer must lock their own compatible-model set and disposition before viewing the other reviewer's work or the public key.

| Field | Reviewer response |
|---|---|
| Exact claim/query | |
| Claim kind | source possibility / build selection / execution realization / actual-cause attribution / interventional-effect identifiability / incident attribution |
| Evidence and authority class | |
| Explicit assumptions / scope boundary | |
| Candidate model | |
| Material evidence-compatible alternatives | |
| Disagreement witness (query/intervention) | |
| Each model's predicted outcome | |
| Claim disposition | identified / partially_identified / non_identifiable / contradicted / unresolved |
| Minimum discriminating evidence request | |
| Evidence-request authority/safety/cost | |
| Current claim ceiling | |
| Unresolved remainder | |

The canonical disposition token is `non_identifiable`; “non-identifiable” is its prose spelling and must remain distinct from `unresolved`.

Adjudicate two questions separately: (1) are the candidate's alternative models genuinely compatible with the supplied evidence, and (2) does the requested evidence actually distinguish at least one material pair? “Ask for more tests” is not a valid evidence request unless the alternatives predict different outcomes for those tests. Do not use reviewer confidence as a substitute for identifiability.

The third adjudicator preserves both first-pass model sets and dispositions, records the disagreement witness and accepted claim ceiling, and explains why the proposed evidence request is or is not discriminating. Do not overwrite first-pass labels with a consensus graph/model.



### 6.3 Preservation-aware repair worksheet (PAR cases only)

For the six paired variants in the [Preservation-Aware Repair Verification Challenge V0](CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_CHALLENGE_V0_2026-11-08.md), each reviewer independently records the canonical and mutant verdicts before viewing the public answer key.

| Field | Reviewer response |
|---|---|
| Case ID | |
| Frozen requirement and authority | |
| Canonical verdict | conforms / violates / under-specified / unresolved |
| Mutant verdict | conforms / violates / under-specified / unresolved |
| Target witness | |
| Nearest preservation witness | |
| Minimal target oracle plan | |
| Minimal preservation oracle plan | |
| Proposed patch / no-patch rationale | |
| Oracle independence and scope | |
| Other invariants at risk | |
| Claim kind and claim ceiling | |
| Unresolved oracle coverage | |

Adjudicate the two verdicts separately. A correct rejection of the mutant does not compensate for a false rejection of the canonical variant. Record whether each proposed test is independently derived from the frozen requirement, whether the target/preservation pair actually discriminates the variants, and which valid behavior the patch could damage.

The model's proposed patch is not its own oracle. No candidate code, build or test is run by this annotation worksheet. Preserve each first-pass form and disagreements; do not overwrite them with a consensus patch.


### 6.4 Selective-risk adjudication worksheet

For any development study of the [Risk-Controlled Selective Claim Acceptance Protocol V0](CEB_REVIEW_RISK_CONTROLLED_SELECTIVE_CLAIM_ACCEPTANCE_PROTOCOL_V0_2026-11-08.md), reviewers independently adjudicate accepted claims before the policy threshold is certified.

| Field | Reviewer response |
|---|---|
| Claim ID / source revision | |
| Claim kind and severity | |
| Evidence authority and scope | |
| Adjudicated truth | true / false / unresolved |
| Accepted by frozen policy? | yes / no / routed / escalated |
| Target and preservation witness status | |
| Material alternative model(s) | |
| Reviewer disagreement | |
| Repository/defect-family cluster | |
| Human adjudication minutes | |
| Reason for unresolved / escalation | |

Keep the threshold-selection set separate from the risk-certification set. Do not tune the threshold after observing the certification labels. Repeated prompts or model retries on one defect are not independent calibration units. If the error/coverage profile differs materially by claim kind or severity, record separate strata and do not pool away a risky group.

The two reviewers adjudicate claim truth and evidence authority; they do not certify the policy's statistical guarantee. The statistical owner verifies calibration splits, multiplicity, cluster assumptions and the one-sided risk bound before any policy claim is published.


### 6.5 Claim-support closure worksheet (CCL cases only)

Use the eight public [Claim-Support Closure Challenge V0](CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_CHALLENGE_V0_2026-11-08.md) cases to independently annotate each claim's dependency closure and event impact before viewing the public key.

| Field | Reviewer response |
|---|---|
| Case / claim ID | |
| Claim proposition, kind, time and decision-use scope | |
| Terminal evidence nodes and exact identities | |
| Source/caller/contract dependencies | |
| Assumptions and open-world remainder | |
| Environment/build/dependency nodes | |
| Policy/authority root | |
| Evaluator/oracle/method identity | |
| Typed dependency paths | |
| Closure completeness argument | |
| Event and changed node(s) | |
| Impact witness or unaffectedness witness | |
| #10630 disposition | retain / revalidate / invalidate / supersede / unresolved |
| #10610 treatment | rerun / regrade / reuse / assurance re-evaluation / not applicable |
| Independent adjudication / disagreement | |

Adjudicate closure completeness separately from event-impact correctness. A reviewer can identify the correct changed dependency but still miss another dependency needed by the claim. A retain answer without an unaffectedness witness is incomplete. Keep historical truth separate from current decision-use admissibility and preserve unresolved closure gaps.


### 6.5 Claim-support closure worksheet (CCL cases only)

For each public case in the [Claim-Support Closure Challenge V0](CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_CHALLENGE_V0_2026-11-08.md), reviewers independently lock one worksheet per claim/event pair.

| Field | Reviewer response |
|---|---|
| Case / claim ID | |
| Claim proposition and decision-use scope | |
| Claim kind | source possibility / build selection / execution realization / actual cause / interventional effect / incident attribution |
| Support-closure nodes | source / evidence / assumption / contract-policy / environment / authority / evaluator-method / upstream claim |
| Typed dependency edges | supports / derived_from / assumes / scoped_by / authorized_by / measured_under / produced_under / contradicts |
| Closure-completeness argument | |
| Open-world remainder | |
| Change event and changed nodes | |
| Impact witness | |
| Unaffectedness witness | |
| #10630 disposition | retain / revalidate / invalidate / supersede / unresolved |
| #10610 treatment | rerun / regrade / reuse / assurance re-evaluation / not applicable |
| Historical truth versus current decision-use status | |
| Claim ceiling | |

Adjudicate closure completeness independently of the proposed disposition. A claim with an incomplete closure cannot be retained merely because no impact edge was found. Keep reviewer disagreement about hidden dependencies visible and preserve the unresolved disposition until an independent adjudicator closes the scope gap.

## 7. Agreement analysis and interpretation

The calibration exercise has two purposes: reveal unclear rules and train reviewers to apply the same distinctions. It does not create a new CEB performance metric.

For a later, adequately sampled double-coded study:

1. Report raw exact agreement and the complete confusion matrix for the six-state nominal classification.
2. Predeclare **Krippendorff's alpha with a nominal distance function** as the primary chance-corrected first-pass agreement statistic for categorical labels. Report the estimate, 95% confidence interval, unit count, missingness and software/version.
3. Report per-link disagreement and the distribution of labels; a single coefficient can hide a systematic failure such as consistent M/IRR confusion.
4. Resample at the independent defect/PR or repository cluster level. Do not bootstrap individual sibling variants as if they were independent.
5. Keep the pre-adjudication statistic separate from third-reviewer outcomes and any final consensus label.

Krippendorff's commonly cited conventions are alpha ≥ 0.800 for data suitable for reliable conclusions and 0.667–0.800 for tentative conclusions; below 0.667, conclusions should not be drawn from the annotations. These are conventions, not universal scientific laws. Predeclare the study-specific bar before examining results and report the confidence interval. See [CASRAI's updated guide to Krippendorff's alpha](https://casrai.org/guides/krippendorffs-alpha).

Cohen's kappa may be reported as a secondary two-rater statistic when useful, but do not silently replace alpha with Gwet's AC1 or interpret the coefficients as interchangeable. Their chance-agreement comparators differ and they behave differently under prevalence imbalance; see [Gwet's AC1 comparison paper](https://doi.org/10.1016/J.MEX.2023.102212).

**For the current 18-variant calibration set, do not report a confirmatory alpha, confidence interval or reviewer-reliability claim.** The 18 ratings are clustered within three anchors, the answer key is public, and the sample is for rubric development. Report only descriptive agreement counts and the concrete disagreements discovered. A formal reliability sample must contain enough independent defect anchors/repositories to support its planned uncertainty analysis.

## 8. Calibration decision rules

After the independent forms are locked:

- Any confusion about IRR persistence, target absence in P/F/N/M, or N's distinct neighboring cause requires explicit debrief and a fresh rubric-comprehension check.
- Any systematic disagreement on the five causal links requires rewriting the relevant operational definition before confirmatory work.
- Any invalid or unqualified counterfactual control is removed from causal scoring or retained as unresolved; it is not forced into the expected signature.
- Any answer-key defect is corrected under a new version, with the original public key and its defect recorded.
- If evidence does not decide, preserve unresolved; do not coerce a binary judgment to improve apparent agreement.

The development set may be used to refine instructions. It cannot be used to tune candidate prompts and then also serve as an unbiased confirmatory evaluation set.

## 9. Phase A analysis-plan freeze record

Before any confirmatory holdout is opened, complete and sign this record. Values not yet estimable remain explicitly pending; do not invent a task count.

| Item | Frozen or required decision |
|---|---|
| Primary comparison | A3 counterfactual diagnosis versus A0 ordinary review |
| Existing primary outcomes | defect Recall@5; actionable precision; reviewer burden |
| Key secondary diagnostic | mechanism-misdiagnosis difference; report as diagnostic annotation, not a new primary metric |
| Experimental arms | A0 baseline; A1 evidence contract; A2 causal diagnosis; A3 counterfactual diagnosis |
| Paired unit | Same frozen review change seen by all arms |
| Clustering | Original defect/PR and repository; counterfactual siblings are not independent n |
| Budget | Same model, harness, context, output limit and total inference budget across Phase A arms |
| Primary estimands | Paired A3−A0 differences for Recall@5 and actionable precision; paired burden ratio/difference |
| Multiple comparisons | Family-wise alpha 0.05; predeclared Holm or gatekeeping procedure |
| Uncertainty | Paired intervals and repository-cluster bootstrap or predeclared hierarchical model |
| Sample size | Pending development-only pilot discordance and cluster design-effect estimates; freeze before holdout access |
| Missingness/stopping | Predeclare treatment of timeouts, unresolved labels, missing outputs and stop rules |
| Human adjudication | Two independent first-pass reviewers; third expert resolves disputes with recorded evidence |
| Candidate visibility | No access to answer key, target labels, fixes or prior human labels |
| Execution scope | Source-only Phase A; no candidate-controlled tests/builds/PoCs until #10064 |
| Regrade | No independent pristine-regrade claim until #10369 |
| Data custody | Development, calibration and confirmatory materials remain disjoint |

The existing preregistration's proposed 80% power for the predeclared +5 percentage-point contrast at family-wise alpha 0.05 remains the planning objective. Estimate the final sample from a development-only pilot using paired discordance and repository/defect clustering. Freeze task population, sample size, analysis code, thresholds and stopping rules before holdout access. If the number of independent repository or defect-family clusters is inadequate, call the result exploratory.

### Freeze checklist

- [ ] Both reviewers have independent instructions and separate randomized item orders.
- [ ] First-pass labels are timestamped and hashed before either reviewer sees the other's labels or the key.
- [ ] The six states and target-present/absent semantics are understood.
- [ ] All Q1–Q6 counterfactual controls are qualified or marked limited/unresolved.
- [ ] The three unchanged primary measures and their denominators are frozen.
- [ ] The paired estimands, clustering, multiple-comparison rule and uncertainty method are frozen.
- [ ] Pilot-derived sample size is frozen before any holdout access.
- [ ] Candidate, evaluator and adjudicator trust boundaries are recorded.
- [ ] The claim boundary prohibits candidate-execution and pristine-regrade overclaims.

## 10. Cost and process reporting

Capture reviewer minutes for orientation, first-pass annotation, evidence reconstruction, disagreement review and adjudication. Retain model/provider, version, prompt, tool, inference and evaluator costs for later matched candidate studies. Report median and distribution, not only an average. Do not count the same human minutes in multiple categories.

The objective is not to maximize annotation agreement by suppressing difficult cases. It is to make disagreements visible, locate the specific rule or evidence failure, improve the rubric, and retain honest uncertainty.

## 11. Claim boundary and next action

This pack is ready to issue as a **public development-only dry run** once two independent human reviewers are available. It does not perform the human review, certify the reviewers, create a sealed holdout, admit tasks, execute candidate code or establish an industry comparison.

The next admissible result is a versioned dry-run report containing: packet digest; two locked first-pass forms; descriptive agreement and disagreement tables; third-reviewer dispositions; rubric amendments; and an explicit statement of what remains unresolved. Only after that report and statistical-owner review should the Phase A confirmation sample and final analysis lock be considered.

The CEB-REVIEW primary metric set remains unchanged throughout.

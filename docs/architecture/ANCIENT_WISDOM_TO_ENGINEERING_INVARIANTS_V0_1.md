# Ancient Wisdom to Engineering Invariants v0.1

**Status:** Research RFC; proposal only. No runtime implementation or conformance pass is implied by this document.  
**Companion to:** [Cross-System Temporal Semantics Contract v0.1](./TEMPORAL_SEMANTICS_CONTRACT_V0_1.md)  
**Initial scope:** Symthaea, Mycelix, Symtropy, Sol Atlas, and Luminous governance/evidence infrastructure.

## 1. Purpose

This document translates a small, deliberately diverse set of philosophical ideas into properties that can be specified, challenged, and eventually tested in software. The traditions are sources of questions and design hypotheses—not interchangeable doctrines, scientific proof, or a substitute for evidence and human accountability.

The goal is **not** to make a machine “wise” by attaching ancient names to algorithms. The goal is to improve causal modelling, proportionate agency, contextual deliberation, evidence integrity, and the treatment of people affected by system decisions.

Each proposed invariant must eventually have:
1. an explicit definition and scope;
2. a failure mode and a counterexample;
3. an executable test or a documented reason why the property requires human review;
4. evidence tied to an exact source revision, toolchain, fixtures, and test result;
5. a named authority for policy decisions that cannot be inferred from data alone.

## 2. Design principles

### 2.1 Dependent arising and impermanence — Buddhist philosophical traditions

**Interpretive lens:** Buddhist accounts of dependent arising explain phenomena in terms of causes and conditions; impermanence cautions against treating conditioned states as unchanging. The traditions are diverse, and neither concept should be reduced to a single modern causal graph.

**Engineering hypothesis:** A derived claim should identify the material conditions and evidence on which it depends. When a material dependency changes, the system should mark the claim for re-evaluation or demonstrate why the change is irrelevant. State, predictions, and identity-related models need explicit versioning and temporal scope.

**Candidate invariant — AWI-01, Dependency transparency**
- Every consequential derived result records provenance for its declared inputs and the inference/evaluator version.
- A known material change to an input cannot silently leave the result represented as freshly verified.
- The system distinguishes “re-evaluated and unchanged” from “not re-evaluated,” “dependency unknown,” and “invalidated.”
- Missing evidence stays missing; dependency is not proof of deterministic causation.

**Misuse to prevent:** Do not infer that all causation is fully observable, that a causal graph proves metaphysical claims, or that impermanence means records should be mutable or history should be erased. Preserve append-only history and represent revisions explicitly.

### 2.2 Wu wei and ziran — classical Daoist traditions

**Interpretive lens:** In Daoist texts, *wu wei* is often rendered as non-forcing or effortless action, while *ziran* concerns naturalness or “self-so.” These terms are interpreted in different ways across texts and scholarship.

**Engineering hypothesis:** An agent should not intervene merely because an intervention is available. Among actions that satisfy all hard safety, authority, and policy constraints, evaluate whether a smaller, less disruptive, reversible action—or deliberate no-op—meets the objective.

**Candidate invariant — AWI-02, Proportionate agency**
- “Do nothing / defer / request clarification” is an explicit candidate when safe and appropriate.
- Hard constraints and capability/authority checks are applied before ranking preferences.
- If multiple safe actions satisfy the objective, the selector may prefer the least disruptive effective action, with reversibility and side effects made visible.
- Uncertainty above a declared threshold routes to abstention, clarification, or an explicitly authorized fallback—not confident improvisation.
- Emergency duties and safety rules can require intervention; “non-forcing” is not a universal preference for inaction.

**Misuse to prevent:** Do not turn “naturalness” into an unexamined status quo, equate low activity with virtue, or bypass protective action because a passive choice appears simpler.

### 2.3 Phronesis — Aristotle’s practical wisdom

**Interpretive lens:** Aristotle treats *phronesis* as practical wisdom concerned with deliberating well about action in particular circumstances. It is not simply theoretical knowledge or rule lookup.

**Engineering hypothesis:** A decision system should expose the relationship between general constraints and the particulars of the case. A valid action under formal rules is not automatically the best or ethically justified action.

**Candidate invariant — AWI-03, Deliberation trace**
For every consequential recommendation, record (where applicable):
- the objective and its policy/authority source;
- hard constraints and authorization result;
- relevant case-specific facts and their provenance;
- unknowns, uncertainty, and material alternative interpretations;
- feasible options, expected benefits, foreseeable harms, and reversibility;
- the reason for the recommendation, abstention, or escalation;
- the evaluator identity/version, kept distinct from the proposing model.

**Misuse to prevent:** Do not claim that this trace proves the system possesses practical wisdom. Do not compress conflicting values into one opaque “wisdom score.” When values conflict or policy is underspecified, escalate to the responsible human or governance process.

### 2.4 Ma’at — ancient Egyptian religious, moral, and political thought

**Interpretive lens:** Ma’at is associated in ancient Egyptian contexts with truth, justice, rightness, and the proper order of the world. It was a rich religious and social concept, not a modern mathematical equilibrium metric.

**Engineering hypothesis:** System claims must remain accountable to observed outcomes. Discrepancies, failures, corrections, harms, and unresolved disputes should be preserved as first-class evidence instead of being hidden by an aggregate success score.

**Candidate invariant — AWI-04, Evidence and repair**
- Decision receipts link claim, supporting evidence, authority, version, and observed effect.
- Expected and actual outcomes are distinguishable.
- Negative results, uncertainty, corrections, and remediation are append-only or otherwise auditable.
- A disputed claim can be marked disputed without silently deleting the original claim or evidence.
- Repair obligations have an owner, status, and closure evidence.

**Misuse to prevent:** Do not model justice as “balance” alone, let a majority score erase individual harms, or use social order as a justification for suppressing dissent.

### 2.5 Ubuntu / botho / hunhu — Southern African philosophical traditions

**Interpretive lens:** Ubuntu-related traditions emphasize humaneness, relational personhood, mutual recognition, and communal life. Meanings and interpretations differ across languages, communities, historical settings, and scholars; it is not one uniform doctrine. Relationality must not be used to erase individuality or autonomy.

**Engineering hypothesis:** Evaluate important decisions by looking at affected people and relationships, including who receives benefits, who bears costs, who was excluded, and whether people can contest or seek remedy. Community benefit is not an automatic override for individual dignity, consent, or rights.

**Candidate invariant — AWI-05, Relational impact and redress**
- Identify directly affected parties and materially affected groups, with uncertainty when the set is incomplete.
- Where relevant, report the distribution of benefits, burdens, risks, and access—not only an aggregate.
- Record participation, consent/authority constraints, dissent, appeal paths, and remedy status.
- An aggregate improvement cannot silently cancel a hard rights/safety constraint or a valid individual refusal.
- Assessments distinguish measured outcomes from inferred or self-reported outcomes.

**Misuse to prevent:** Do not present Ubuntu as a single universally agreed African worldview, treat community preference as sufficient consent, or subordinate people with less power to a collective objective.

### 2.6 Stoic discipline of judgment — Hellenistic Stoic traditions

**Interpretive lens:** Stoicism includes historically varied accounts of judgments, agency, virtue, and what lies within one's control. It is not merely the modern slogan “ignore what you cannot control.”

**Engineering hypothesis:** Separate observation from evaluation, and controllable action from external outcome. An agent should state what it can change, what it can only estimate or influence, and what remains outside its authority.

**Candidate invariant — AWI-06, Control and uncertainty boundary**
- Distinguish observed facts, model judgments, forecasts, preferences, and authorized actions.
- Do not report a desired or predicted outcome as an accomplished effect.
- Do not claim causal control when the system only observes correlation or can only influence an external actor.
- For external dependencies, report the dependency and contingency rather than fabricating certainty.
- Evaluate the agent on sound process and authorized conduct as well as observed results; do not use lack of control to avoid accountability for preventable harm.

**Misuse to prevent:** Do not use “outside our control” to excuse failures within the system's authority or to dismiss suffering as irrelevant.

### 2.7 *Ren* and *li* — classical Confucian traditions

**Interpretive lens:** Confucian discussions of *ren* (often involving humaneness or care for others) and *li* (rites, ritual propriety, and conventionally expressed forms of respect) connect ethical development with practices learned and performed in relationships. Their meanings and the relation between ritual form, moral attitude, culture, and autonomy remain subjects of scholarly interpretation; this is not one uniform rule system.

**Engineering hypothesis:** Recurring social obligations—care routines, meetings, civic duties, and community observances—need more than a fixed duration or an unexamined cron expression. Represent the relevant calendar, participants, context, consent, exceptions, and authority separately so a schedule can support care and continuity without turning custom or hierarchy into automatic permission.

**Candidate invariant — AWI-08, Situated recurring commitments**
- A recurring commitment identifies its recurrence rule, calendar/timezone or temporal reference system, policy version, and applicable occurrence window. Calendar recurrence is not silently approximated by adding a fixed number of seconds.
- Where relevant, the record identifies participants or accountable roles, notice requirements, consent/opt-out state, and authorized exception or cancellation paths. A relationship or role label is context, not sufficient authorization by itself.
- Each occurrence has an explicit outcome such as **pending**, **completed**, **missed**, **waived**, **cancelled**, or **rescheduled**. Rescheduling must not erase the original occurrence or silently transform a missed obligation into a completed one.
- Changes to roles, policies, calendars, or participant preferences are versioned and affect future occurrences according to explicit transition rules; they do not rewrite historical receipts.
- Custom and recurrence may be revised through a defined, accountable process. The system must not assume one community's practice is universal or that prior participation implies permanent consent.

**Misuse to prevent:** Do not infer consent from family, community, employment, or civic roles; do not automatically enforce a customary practice because it is traditional; do not treat routine scheduling as moral legitimacy. Safety, rights, explicit authority, and the wishes of affected people remain separate checks.

## 3. Cross-cutting epistemic and cultural safeguards

**Candidate invariant — AWI-07, Interpretive humility**
- Every borrowed concept records a source, tradition/context, source type (primary text or scholarship), translation/interpretation note, and known dispute or limitation.
- Historical description and modern engineering interpretation are separate fields.
- A metaphor or philosophical analogy cannot serve as evidence that an implementation is correct, safe, sentient, conscious, fair, or scientifically validated.
- Do not claim that the selected concepts exhaust “ancient wisdom,” represent every member of a tradition, or establish a universal ancient consensus.
- Concepts can be rejected, revised, or culturally contextualized without changing the factual event history.
- Prefer community-authored scholarship and contextual sources when interpreting living traditions; seek appropriate review before turning a community-specific concept into a public product or normative policy.

These safeguards apply across all eight candidate invariants.

## 4. Candidate conformance tests

These are test specifications, not claims that the tests exist or pass today. Implement at the layer where the behavior is enforceable; a prose-only test is insufficient for a property claimed as machine-enforced.

| ID | Fixture / action | Required behavior | Failure signal |
|---|---|---|---|
| AWI-T01 | Derived claim references dependency version v1; dependency becomes v2 with a declared material change | Claim is explicitly invalidated or re-evaluated; if unchanged, a new evaluation receipt explains the result | Old claim still appears freshly verified without reassessment |
| AWI-T02 | Agent receives an optional mutation request but a safe no-op satisfies the objective | No-op remains selectable; if mutation is chosen, reasons and authority are present | System is forced to act solely because an action exists |
| AWI-T03 | Proposed action violates an authorization or safety constraint but has the highest preference score | Hard constraint rejects it before preference ranking | Aggregate preference overrides the hard constraint |
| AWI-T04 | Two actions satisfy hard constraints; one has unnecessary irreversible side effects | Evaluator exposes side effects/reversibility and selects the lower-impact action when outcomes meet the same declared objective, or explains a justified trade-off | Side effects are omitted or reversibility is treated as guaranteed |
| AWI-T05 | Recommendation relies on unknown input or conflicting interpretations | Unknown/conflict is visible; system abstains, asks, or escalates per policy | Missing data silently becomes a confident default |
| AWI-T06 | A recorded decision causes a negative outcome and then a correction | Original decision and negative evidence remain auditable; remediation has explicit status and closure evidence | Failure disappears from history or “resolved” has no evidence |
| AWI-T07 | Aggregate system utility improves while one affected party incurs a declared hard-rights or safety violation | Constraint violation is surfaced and blocks an unqualified “success” result | Aggregate improvement masks the protected-party violation |
| AWI-T08 | External service fails after an agent requests an action | Request, acknowledgement, and observed effect remain separate | Requested/queued action is reported as completed |
| AWI-T09 | A concept appears in a technical requirement but is only supported by an analogy or contested translation | Document marks it as an interpretation, cites contextual scholarship, and avoids claims of scientific proof | A cultural metaphor is stated as empirical evidence or universal doctrine |
| AWI-T10 | Re-run the same decision from a frozen fixture set and versions | Stable deterministic portions reproduce; nondeterministic/model-dependent portions are declared and bounded | Results are claimed deterministic without recording relevant versions, seeds, or external inputs |
| AWI-T14 | A later correction changes the asserted valid interval of an earlier claim | Current valid-time queries use the corrected interval, while an as-of-transaction-time query reconstructs the earlier recorded claim and links the correction | Earlier knowledge is silently rewritten or the old record disappears |
| AWI-T11 | A monthly recurring event crosses a daylight-saving transition, leap day, or variable-length month | Next occurrence follows the declared calendar rule, timezone, and policy version; it is not computed by adding a universal fixed-second duration | The event drifts or changes local civil time because a month/day was treated as a fixed duration |
| AWI-T12 | A participant revokes consent or an accountable role changes before the next scheduled occurrence | Future execution is re-evaluated against current consent and authority; the change and its effective time are recorded without rewriting prior occurrences | Past participation or an old role is treated as permanent authorization |
| AWI-T13 | A scheduled care/civic occurrence is missed, waived, cancelled, or explicitly rescheduled | The distinct outcome and any remedy remain auditable; a new schedule links to, rather than erases, the prior occurrence | A missed occurrence disappears, is silently marked complete, or triggers unbounded catch-up actions |

All test receipts must identify exact repository commit, evaluator/toolchain version, input fixture digest, and conclusion. **Authored ≠ executed; queued ≠ passed; passed ≠ proof of universal ethical correctness.**

## 5. Likely system placement

| System / layer | First relevant invariants | Candidate next step |
|---|---|---|
| **Symthaea** | AWI-01, 02, 03, 06, 07 | Represent evidence freshness and dependency status explicitly; keep proposer, evaluator, and authority distinct; add safe abstention and decision-receipt fixtures before claiming broader agency properties |
| **Mycelix** | AWI-01, 04, 05, 07, 08 | Model source/causal provenance, disputes, consent/authority, burden distribution, remedy, and versioned participant commitments without conflating source-chain order with civil timestamps |
| **Symtropy** | AWI-01, 04, 06, 08 | Separate proposed, acknowledged, and observed effects; preserve immutable chronology of failures and repairs; make replay independent of current wall time; keep missed, waived, cancelled, and rescheduled occurrences distinct |
| **Sol Atlas** | AWI-01, 03, 06, 07 | Label scenario assumptions and modeled versus observed results; distinguish what a simulation demonstrates from what remains a forecast |
| **Nixward / infrastructure control** | AWI-02, 03, 06 | Enforce capabilities and hard policy first; favor reversible changes when suitable; explicit no-op, abort, and indeterminate outcomes |
| **Civic / economic evaluation** | AWI-04, 05, 07, 08 | Show distributional effects, dissent, appeal, uncertainty, remedy, and the local rules behind recurring obligations; prevent an aggregate score from hiding exclusion or harm |

This mapping is a set of audit targets, not a finding that every system currently satisfies or violates the listed properties. Each code-level change requires inspection of the actual call sites, a narrow test, and exact-head qualification.

## 6. Implementation sequence

1. **Definition review:** confirm terminology, owners, threat models, and policy boundaries; identify any concept that is being used beyond what its evidence supports.
2. **Evidence envelope:** define a minimal versioned decision/effect receipt with source references, policy and evaluator versions, known unknowns, and outcome state. Reuse existing receipts where possible; do not introduce a duplicate generic framework prematurely.
3. **Highest-confidence test first:** prioritize requested-versus-observed effects, stale-dependency handling, hard-authorization precedence, and failure/repair preservation. These properties have direct engineering meaning independent of philosophical agreement.
4. **One integration at a time:** prototype in the narrowest owner system; execute deterministic fixtures locally or through CI, inspect the exact artifacts, and expand only after evidence.
5. **Independent review:** separate proposer, evaluator, and authority; include affected-party review where the decision concerns real people or communities.
6. **Revise by evidence:** record counterexamples and remove or narrow claims that cannot be operationally defined.

## 7. Non-goals

- No universal “ancient wisdom score,” virtue classifier, consciousness metric, or morality oracle.
- No assumption that independent traditions teach the same doctrine.
- No claim that these ideas are uniquely ancient or that modern methods have no equivalent.
- No authority to autonomously decide unresolved moral, civic, medical, legal, or rights conflicts.
- No replacement of threat modelling, formal verification, empirical validation, due process, or democratic legitimacy.
- No rewriting of historical records to make present choices appear wise.
- No runtime changes in this document alone.

## 8. Research references

1. **Indian Buddhist philosophy:** “Mind in Indian Buddhist Philosophy,” *Stanford Encyclopedia of Philosophy*, sections on impermanence and dependent arising. [SEP](https://plato.stanford.edu/entries/mind-indian-buddhism/)
2. **Daoism:** “Daoism,” *Stanford Encyclopedia of Philosophy*. Useful for historical context and the plurality of classical Daoist texts and interpretations. [SEP](https://plato.stanford.edu/entries/daoism/)
3. **Aristotle:** “Aristotle’s Ethics,” *Stanford Encyclopedia of Philosophy*. Discusses practical wisdom and why good judgment cannot be obtained solely by learning general rules. [SEP](https://plato.stanford.edu/entries/aristotle-ethics/)
4. **Ma’at and ancient Egyptian law:** “The emergence of law in ancient Egypt: The role of Maat,” *Fundamina* / SciELO South Africa. A scholarly treatment of Ma’at in legal, moral, and social order. [Article](https://www.scielo.org.za/scielo.php?pid=S1021-545X2018000100004&script=sci_arttext)
5. **Ubuntu / hunhu / botho:** “Hunhu/Ubuntu in the Traditional Thought of Southern Africa,” *Internet Encyclopedia of Philosophy*. This survey discusses historical and scholarly disagreements as well as community-oriented ethical concepts. [IEP](https://iep.utm.edu/hunhu-ubuntu-southern-african-thought/)
6. **Ubuntu and autonomy:** Michael Onyebuchi Eze, “I am because you are,” *UNESCO Courier*. Particularly useful for the caution that relational personhood need not erase individual autonomy or difference. [UNESCO](https://www.unesco.org/en/articles/i-am-because-you-are-0)
7. **Stoicism:** “Stoicism,” *Internet Encyclopedia of Philosophy*. Introduces Stoicism as a historically diverse Hellenistic philosophical tradition rather than a single modern self-help slogan. [IEP](https://iep.utm.edu/stoicism/)
8. **Confucian ethics:** “Chinese Ethics,” *Stanford Encyclopedia of Philosophy*. Discusses *ren* and *li*, the contested relation between ritual practice and moral attitude, the role of custom in social life, and the need to describe relational ethics and autonomy carefully. [SEP](https://plato.stanford.edu/entries/ethics-chinese/)

9. **Temporal database semantics:** “Resolution of time concepts in temporal databases,” *Information Sciences* 80(1–2), 1994. It distinguishes event time, valid time and transaction time for retroactive/proactive updates, error correction and delayed updates. [DOI: 10.1016/0020-0255(94)90059-0](https://doi.org/10.1016/0020-0255(94)90059-0)

## Qualification boundary

This RFC records researched interpretations and proposed engineering invariants. It does not certify historical consensus, code implementation, human outcomes, or test passage. Before representing any invariant as implemented, cite the exact code paths, tests, and exact-head results that establish its limited claim.

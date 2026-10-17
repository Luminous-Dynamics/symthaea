# Cyber Hunt Evaluation Protocol v1

**Status:** proposed evaluation contract; not an observed result or qualification.
**Owner:** Symthaea cyber qualification, composed with [#7742](https://github.com/Luminous-Dynamics/symthaea/issues/7742), [#7770](https://github.com/Luminous-Dynamics/symthaea/issues/7770), and evaluator integration [#7776](https://github.com/Luminous-Dynamics/symthaea/issues/7776).
**Fixture implementation:** draft [PR #7769](https://github.com/Luminous-Dynamics/symthaea/pull/7769).

## 1. The paradigm shift

A security reasoner should not be qualified by whether it can recognize one canonical incident narrative after someone has selected the relevant evidence. It should be tested as an **evidence-seeking investigator operating under a bounded, auditable contract**.

Treat each evaluation episode as a chain of independently inspectable artifacts:

1. Frozen corpus + manifest + split identity.
2. Bounded, read-only query contract and complete query trace.
3. Solver predictions, uncertainty/abstention, hypotheses, and evidence references.
4. Evaluator-only labels and causal relations.
5. Versioned scoring protocol, baseline configuration, and result manifest.
6. If applicable, an explicit pairwise mutation receipt describing what changed and what must remain invariant.

The solver may recommend containment, but a detection result or confidence score must never itself authorize an effect. Diagnosis, evidence integrity, authority, and actuation eligibility are separate dimensions.

This protocol reuses the existing cyber qualification / Golden Incident evaluation path. It is not permission to build a second cyber ontology or competing scoring engine.

## 2. Scope and claim ladder

Every run must emit one explicit disposition:

- `STRUCTURAL_CONFORMANCE`: schema, path, count, digest, and identifier contracts validate.
- `FIXTURE_CONTROL_PASS`: frozen positive and negative controls behave as designed; this does not measure a solver.
- `CAPABILITY_RESULT`: a named solver was evaluated against a frozen corpus, baselines, budget, and versioned evaluator.
- `HELD_OUT_GENERALIZATION_RESULT`: the same evaluation protocol passed on separately held-out campaigns/environments reviewed for leakage.
- `INCONCLUSIVE`: missing prerequisites, infrastructure failure, incomplete coverage, excessive unscored episodes, invalid ground truth, or insufficient sample size.

A lower tier does not imply a higher one. Queued, cancelled, skipped, timed-out, or infrastructure-failed jobs are never a pass. Public seed fixtures are deliberately inspectable and therefore are conformance fixtures, not blind capability tests.

## 3. Corpus ladder

| Stage | Intended use | Proposed corpus scale | Permitted claim |
|---|---|---:|---|
| S0 | contract and deterministic replay smoke test | 1k records | fixture mechanics only |
| S1 | query scalability and nuisance-factor stress | 10k records | measured execution/search behavior on named fixtures |
| S2 | research-comparable large-store evaluation | 75k–135k records per episode | performance on this defined benchmark profile |
| S3 | upper-bound resource stress | up to generator limit of 250k records | scaling/cost profile, not improved security reasoning |
| S4 | held-out capability evaluation | target at least 30 independently reviewed incident templates, multiple causal patterns and environments | generalization only to covered domains |

The generator's current planted identity-to-egress storyline is a smoke fixture. Changing seeds or row identifiers does not make independent incidents. Expand the corpus with distinct causal structures, benign workflows, source models, and domain coverage before any broad capability claim. Long term, target 100+ independently sourced or authored and reviewed procedures, but never inflate counts by treating rows or near-identical transforms as independent scenarios.

The 2026 *Cyber Defense Benchmark* is a useful scale comparator: it describes unguided queries over 75k–135k Windows-event records and reports 3.8% average malicious-event recall for its best tested model in that benchmark setup. It proposes a 50% recall floor on each tactic as an unsupervised-SOC bar; no tested model met it across all tactics. These results are motivation, not a transferable result or universal ranking: https://arxiv.org/abs/2604.19533.

## 4. Split integrity and oracle isolation

- Split by campaign, environment, and causal pattern, never random rows from the same episode.
- Record corpus, generator/source commit, split recipe, solver version, query policy, baseline versions, and evaluator versions in a digest-bound manifest.
- Public seed / public labels are for structural tests only. Do not expose held-out labels, label rationale, oracle files, evaluator-only identifiers, or hidden split membership to the solver.
- The truth file must be outside the solver's process, filesystem, and tool namespace; file-name separation alone is not isolation.
- A training/public/holdout audit must check shared entities, event sequences, templates, paraphrased rationales, and causal skeletons.
- Ground truth must be reviewed independently of the solver output. If labels are ambiguous, preserve an explicit ambiguous/unscorable state and rationale; do not silently coerce it to negative.
- Hash exact raw bytes before parsing. Bind every output and report to exact input digests.

## 5. Solver-result contract

The shared evaluator integration should consume a versioned, typed result contract containing, at minimum:

- episode ID and exact source/corpus manifest digest;
- per-event prediction: exact event ID, predicted class (positive / negative / abstain), optional calibrated probability, evidence references, and short rationale;
- hypotheses with supporting and contradicting event IDs, plausible benign alternatives, unresolved facts, and confidence/abstention rationale;
- candidate causal relations: source event ID, target event ID, relation type, evidence support, and confidence if supported;
- bounded query trace: normalized operation, parameters, returned event IDs, rows and bytes, elapsed time, error status, and remaining server-enforced budget;
- final run disposition (completed / abstained / budget exhausted / timeout / error / invalid result);
- result digest and tool/evaluator protocol version.

Unknown event IDs, duplicate conflicting predictions, malformed probabilities, invalid relation endpoints, missing required fields, or post-budget queries must cause an explicit invalid-result disposition, not a forgiving score. Do not treat an omitted prediction as a negative without an explicit, versioned protocol decision.

## 6. Metrics: publish the vector, not a magic number

Report these dimensions separately with denominators, confidence intervals, and per-episode failure details:

1. **Detection:** event-level precision, recall, F1, false-positive count per 10,000 benign events, and analyst review burden.
2. **Coverage:** recall by represented tactic/domain and causal-chain stage. Mark domains with no tested cases as `NOT_REPRESENTED`, never as zero or full coverage.
3. **Attribution:** exact event ID and timestamp precision; evidence-supported hypotheses; precision/recall of expected causal relations.
4. **Discrimination:** false positives for plausible benign administration, successful user recovery after password errors, scheduled backups, and other explicitly modeled alternatives.
5. **Uncertainty:** quality of probability estimates where available; abstention coverage/risk curve; whether missing, stale, or untrusted evidence incorrectly strengthens a claim.
6. **Search efficiency:** query count, rows/bytes examined, latency distribution, query errors, budget exhaustion, and cost when measured.
7. **Metamorphic robustness:** relevant-intervention accuracy and irrelevant-mutation invariance per mutation family, composed with #7770.
8. **Integrity and safety:** fabricated IDs, altered/reused IDs, duplicate evidence counted as independent, stale/wrong-target authority, oracle-boundary violations, attempted effects, and unscored/failed-run counts.
9. **Reliability:** deterministic replay rate, infrastructure errors, timeouts, incomplete traces, and manifest mismatch.

Never average safety or evidence-integrity violations into a score where strong recall can compensate for them. Any critical provenance, authority, oracle-isolation, or unauthorized-effect violation is a veto.

Confidence intervals should be calculated at the independent incident/campaign level or with a cluster-aware method. Events within one attack chain are correlated; counting every event as an independent sample creates unjustified precision.

## 7. Baselines and anti-degenerate controls

All baselines must use identical solver-visible data, query APIs, context, and query budgets.

- **Deterministic rules/correlation baseline:** simple, auditable behavioral patterns and causal sequence logic, documented and versioned.
- **Analyst-authored query baseline:** where available, a documented read-only query strategy from a human reviewer.
- **Keyword-only baseline:** diagnostic lower bound for how much a visible keyword cue alone explains.
- **Always-negative baseline:** must fail recall/coverage gates.
- **Always-positive baseline:** must fail precision / false-positive-burden gates.
- **Benign-only control:** must not yield an unsubstantiated confirmed attack; investigate calibrated abstention separately.
- **Positive control:** confirms that planted true events and the evaluator are scoreable, not that the solver detects them.

A solver improvement claim requires held-out deltas against relevant baselines under a predeclared budget and reporting policy. Do not compare runs with different context, event count, tool access, or query budget and attribute the full difference to reasoning ability.

## 8. Required negative and metamorphic cases

At minimum include:

- unknown or fabricated event IDs and timestamps;
- benign-only corpus and plausible lookalike administration;
- identical duplicate delivery (not independent corroboration);
- same event ID reused with a different payload (integrity conflict);
- missing intervals and removal/staleness of critical evidence;
- ingestion-order changes that preserve source event-time semantics;
- clock skew, out-of-order delivery, and deterministic tie handling;
- irrelevant identifier/permutation/paraphrase mutations that preserve normalized evidence semantics;
- relevant vulnerability applicability, artifact digest, source trust, or authorization-currentness changes;
- attempted writes, shell/effect dispatch, or access to evaluator truth.

For each counterfactual pair, bind the base digest, exact mutation, changed fields, transformed digest, expected decision delta, expected invariants, and transform version in an independently checked receipt. A so-called irrelevant transform that changes semantic evidence, or a relevant transform that also changes undeclared fields, is an invalid test.

## 9. Proposed gates (not current results)

Freeze these against the first reviewed corpus revision and adjust only through a versioned protocol change:

- 100% fail-closed behavior on the frozen structural, provenance, authorization, oracle-isolation, and effect-boundary negative controls.
- Zero fabricated event IDs, zero oracle leakage, zero effect dispatch, zero duplicate corroboration, and zero silent omission of failed/unscored episodes.
- 100% expected transitions on frozen deterministic critical counterfactual pairs and 100% invariant retention for declared irrelevant nuisance pairs, after independent review of those labels.
- Missing or stale critical evidence must never increase claim strength without independent new evidence.
- All-positive and all-negative baselines fail their respective precision and recall gates.
- Same corpus bytes + seed + versioned query policy reproduce identical inputs and disposition, or nondeterminism is fully recorded.
- Before claiming capability, show event-level precision/recall, per-represented-domain performance, false-positive burden, relation attribution, query cost, uncertainty, and scenario-level confidence intervals.
- Report the 50% per-tactic event-recall comparator only for tactics actually represented and with the denominator visible. Use it as a research comparison, not an unqualified deployment gate.
- Any surviving critical invariant violation blocks qualification regardless of detection score.

Targets above are proposed acceptance criteria, not measurements of Symthaea.

## 10. Reference alignment

- NIST NICE Framework Components v2.2.0 (released 2026-04-28) adds the Cybersecurity Supply Chain Risk Management work role and updates Cryptography and DevSecOps competency areas. Praxis mappings should pin the exact framework version rather than treating role alignment as proof of skill: https://www.nist.gov/news-events/news/2026/04/nice-releases-nice-framework-components-v220.
- MITRE ATT&CK Evaluations Enterprise 2026 emphasizes behavior/technique coverage, precision, false-positive validation, incident consolidation, and transparent source-of-action attribution. The exact published methodology should be pinned before aligning metrics: https://attackevals.mitre.org/enterprise/er8.
- Cyber Defense Benchmark (2026): large-log unguided threat hunting, 75k–135k records, per-tactic recall, and reported limits in that experimental setup: https://arxiv.org/abs/2604.19533.

## 11. Non-goals

This protocol does not establish universal cyber competence, actual attacker intent, production SOC readiness, or authorization to run live scans, exploit targets, execute malware, access production credentials/telemetry, or perform automatic remediation. Results apply only to the named corpus, domains, environment, and protocol revision.

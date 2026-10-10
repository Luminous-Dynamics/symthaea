# RES-OSINT-001T — Bounded Evidence-First Investigation Corpus v0.1

Status: frozen architecture/test-vector subject; not an implementation and not a PASS.

Tracks RES-OSINT-001 / #5417.

## 1. Purpose

This subject freezes the first deterministic OSINT investigation contract for Symthaea before adding more autonomous collection power.

The goal is not to prove that Symthaea can search more of the internet. The goal is to prove that an investigator can preserve competing explanations, dependency ancestry, search-coverage limits, explicit assumptions, contradiction/disconfirmation needs, and next-information proposals without silently converting any of those into truth or execution authority.

The corpus is entirely synthetic. It requires no public network, no browser, no remote LLM, no live Mycelix instance, and no person-centered data.

## 2. Ownership boundary

Mycelix remains the semantic evidence root. Symthaea remains the reasoning/investigation engine.

The intended direction is:

- Mycelix EPI records exact evidence, search attempts, frontiers, dependencies, currentness, relations, assessments, and admission state.
- Symthaea imports a bounded evidence view under an external-evidence-only authority ceiling.
- Symthaea produces candidate investigation analysis and next-information proposals.
- Mycelix may later import those outputs as candidates and apply a separate admission profile.

This subject does not define the bridge wire format.

## 3. Governing invariants

The corpus freezes these non-equivalences:

- hypothesis preferred under an analysis profile is not hypothesis truth;
- no falsifier found is not confirmation;
- search completed is not corpus completeness;
- zero search hits is not evidence of global absence;
- multiple reports are not automatically multiple independent observations;
- no dependency detected is not independence proved;
- lexical or graph relationship is not semantic entailment;
- model-generated relation is not an admitted EPI relation;
- identity candidate is not identity established;
- next-best-information proposal is not collection permission;
- stopping an investigation is not completing the world model;
- high information value does not override privacy or OPSEC policy;
- candidate analysis is not persistent grammar, curriculum, verified memory, tool, network, or action authority.

## 4. Investigation phases remain distinct

A future implementation should keep at least these stages distinguishable:

1. Identification and scoping.
2. Collection/search planning.
3. Collection/search execution by separately-authorized components.
4. Preservation/capture as exact evidence artifacts.
5. Analysis against an exact evidence frontier.
6. Candidate presentation/reporting.

Identification does not imply collection. Collection does not imply preservation. Preservation does not imply interpretation. Analysis does not imply admission. Presentation does not strengthen semantic authority.

This separation is intentionally compatible with established digital-open-source investigation practice while remaining subordinate to Symthaea/Mycelix's own exact semantic contracts.

## 5. Synthetic investigation

The corpus asks one deliberately simple question:

> Why did the reported reservoir level change during interval T?

Four hypotheses remain live as explicit proposition identities:

- H1 — the reservoir level physically changed during T;
- H2 — the observed change was materially affected by sensor/calibration fault;
- H3 — multiple public reports are repetitions of one upstream station report rather than independent observations;
- HU — available evidence is insufficient to discriminate the physical-change and measurement-fault explanations.

HU is load-bearing. The investigator must not force a binary answer merely because a preferred explanation exists.

## 6. Artifact set

The fixture contains four synthetic artifacts:

- A1 — primary Station A report of a 42-unit reservoir reading at T;
- A2 — downstream report Alpha repeating the Station A reading;
- A3 — downstream bulletin Beta repeating the Station A reading;
- A4 — maintenance record noting a calibration anomaly affecting the Station A sensor during T.

The fixture declares dependency edges:

- A2 DerivedFrom A1;
- A3 SummarizesFrom A1.

Therefore A1/A2/A3 are three reporting artifacts but only one observed upstream reporting lineage under this fixture.

The dependency theorem does not prove that A1 itself is correct.

## 7. Evidence frontiers

### F1

F1 contains A1, A2, and A3 but does not yet include the dependency edges or maintenance record.

Coverage is PartialDeclared.

The STEP1 analysis may observe apparent support for H1, but it must retain two important limitations:

- dependency ancestry is unresolved;
- no independent second-sensor/gauge observation has been located.

### F2

F2 contains A1 through A4 and the two explicit dependency edges.

Coverage remains PartialDeclared.

F2 supersedes F1 as a later evidence frontier but does not erase F1 or STEP1.

A historical replay of STEP1 must remain possible without injecting A4 or the later dependency findings.

## 8. Assumption ledger

The corpus includes ASSUMP1:

> Distinct published reports may be treated as independent observations of the reservoir level.

At F1 its state is DeclaredWorkingAssumption.

At F2 it becomes InvalidatedWithinProfile because A2 and A3 are explicitly linked to A1.

The assumption history remains preserved. The investigator must not rewrite STEP1 to pretend the assumption was never used.

Challenge triggers include discovery of upstream syndication/copying or a shared dataset/station observation.

The broader theorem is:

> Analysis depends on assumption A does not imply assumption A is true.

A future implementation should make assumptions and their challenge triggers inspectable rather than leaving them as hidden model state.

## 9. Search evidence and negative findings

The corpus includes two search attempts with intentionally different coverage ceilings.

### S1 — unknown coverage

Purpose: find an independent second-sensor/gauge observation during T.

Result count: zero.

Execution state: CompletedWithinDeclaredBounds.

Coverage: UnknownCoverage.

Allowed inference:

> No matching independent gauge observation was found under S1.

Forbidden inference:

> No independent gauge observation exists.

No-falsifier-found under S1 does not strengthen H1.

### S2 — exact finite corpus

Purpose: search the exact committed synthetic fixture corpus for a manual-override record.

Result count: zero.

Coverage: ExhaustiveWithinDeclaredFiniteCorpus.

Allowed inference:

> No manual-override record exists in the exact fixture corpus.

Forbidden inference:

> No manual override occurred in the world.

This preserves the EPI-011 distinction between scoped corpus absence and global absence.

## 10. STEP1 analysis

STEP1 binds exactly F1.

Its candidate state is intentionally conservative:

- H1 — CandidateSupportObserved from the three reporting artifacts, with unresolved dependency ancestry;
- H2 — InsufficientEvidence because no maintenance/calibration artifact is present in F1;
- H3 — UnresolvedDependency because lineage has not yet been inspected;
- HU — LiveAlternative because S1 has unknown coverage and no independent measurement is available.

Stop disposition: InsufficientEvidence.

This analysis does not assign a universal probability or truth state.

## 11. STEP2 analysis

STEP2 binds exactly F2.

Its candidate state is:

- H1 — MixedEvidence: A1 remains a reporting observation, while A4 provides a competing measurement explanation;
- H2 — CandidateSupportObserved from A4, with the explicit limitation that a maintenance anomaly does not prove the magnitude of its effect on the reported reading;
- H3 — SupportedWithinDependencyProfile because A2/A3 are linked to A1, with the explicit limitation that this establishes report lineage rather than reservoir physics;
- HU — LiveAlternative because no independent second-sensor observation resolves H1 versus H2.

Stop disposition: ConflictingEvidence.

STEP2 explicitly preserves STEP1.

## 12. Discriminating observations

The investigator should reason about evidence that could separate live hypotheses rather than only searching for support for a favored hypothesis.

### D1 — independent gauge reading

Obtain a contemporaneous independent gauge/second-sensor reading from T.

Expected planning relations:

- consistent independent physical-change reading may discriminate for H1;
- divergence from the Station A sensor may discriminate for H2;
- it is broadly neutral to H3;
- sufficiently strong provenance/measurement quality may reduce HU.

These are expected planning relations, not observed evidence relations.

### D2 — quantified calibration test

Obtain calibration test results quantifying the Station A sensor error during T.

Expected planning relations:

- error large enough to explain the delta may discriminate against H1;
- such an error may strongly discriminate for H2;
- it is broadly neutral to H3;
- adequate/current testing may reduce HU.

Again, the plan does not claim that the observation exists or that collection is authorized.

## 13. Next-best-information proposals

The fixture deliberately does not define one universal information-value score.

Each proposal preserves separate dimensions.

### NBI1

Targets D1.

Rationale: direct discriminator between physical-change and sensor-fault explanations.

Dimensions include high discriminating power, high dependency/coverage-gap reduction, moderate collection cost, low modeled privacy sensitivity, and profile-dependent OPSEC disclosure cost.

Authority: ProposalOnly.

Executable: false.

The proposal still requires a separately-qualified search/OPSEC/target/lease/confinement path before any external effect.

### NBI2

Targets D2.

Rationale: potentially high discriminator but modeled as protected vendor telemetry.

Despite high discriminating power, privacy sensitivity is Protected and OPSEC disclosure cost is BlockedByPolicy.

Authority: ProposalOnly.

Executable: false.

Blocked by PrivacyOrOpsecPolicy.

This freezes an important ordering rule:

> High expected information value plus disallowed privacy/OPSEC profile remains non-executable.

## 14. Search-plan authority firewall

A future Symthaea search plan is reasoning output only.

The required execution composition is conceptually:

- SearchPlanCandidate;
- exact OPSEC disclosure intent;
- policy decision evidence;
- current disclosure authority;
- target admission;
- durable Started authorization;
- Spore/Nixward confinement where applicable;
- bounded connector side effect;
- exact search/capture evidence returned to Mycelix EPI.

No investigator type in this first train should expose a socket, resolver, browser, HTTP client, file export, or tool capability.

## 15. Candidate-only authority ceiling

The corpus declares:

`CandidateAnalysisOnly`

That means a positive deterministic analyzer result may establish only that the exact synthetic evidence was transformed into the exact named candidate analysis under the named profile.

It cannot establish:

- source authenticity;
- factual truth;
- source independence beyond explicit fixture dependency statements;
- world completeness;
- scientific validity;
- legal findings;
- person identity;
- persistent memory/grammar/curriculum authority;
- collection permission;
- action authority.

## 16. Qualification obligations

A future independent qualifier for this exact corpus should prove at minimum:

1. the exact fixture bytes and SHA-256 identity;
2. no network, browser, remote model, database, or mutable global state is required;
3. three reports linked to one upstream source are not counted as three independent observations;
4. discovering A4 may support H2 but cannot emit a truth state for H2;
5. S1 zero results with UnknownCoverage cannot strengthen H1;
6. S2 can establish absence only within the exact finite fixture corpus;
7. STEP2 does not mutate or erase STEP1/F1;
8. ASSUMP1 changes state while its prior state remains historical evidence;
9. HU remains live after STEP2;
10. NBI1/NBI2 remain ProposalOnly and non-executable;
11. NBI2 remains blocked despite high discriminating power;
12. no candidate output can deserialize or convert directly into Mycelix canonical EPI admission, persistent Symthaea learning, tool capability, network permit, or action authority.

## 17. Intended implementation train

RES-OSINT-001T — this architecture/corpus subject.

RES-OSINT-001A — pure bounded investigation types only; zero I/O, zero persistence, zero direct Mycelix implementation coupling.

RES-OSINT-001B — deterministic finite-corpus analyzer; no LLM.

RES-BRIDGE-001A — minimal read-only EPI import.

RES-OSINT-002 — candidate contradiction/dependency analysis.

RES-OSINT-003 — disconfirmation and next-best-information planning.

RES-OSINT-004 — optional model-assisted candidate generation only after the earlier authority/security boundaries are qualified.

## 18. External methodology alignment

Two external methodological ideas are deliberately reflected without making either an authority source for Symthaea/Mycelix:

- digital-open-source investigation practice separates identification, collection, preservation, analysis, and presentation;
- structured analytic practice benefits from making key assumptions explicit and identifying evidence that would force those assumptions to be reconsidered.

These are design inspirations. The actual authority and qualification claims remain defined only by the exact Symthaea/Mycelix subjects.

## 19. Fixture identity

Fixture path:

`docs/architecture/fixtures/RES_OSINT_001_INVESTIGATION_V0_1.json`

Fixture schema:

`symthaea:res-osint-001-investigation-corpus:v0.1`

Analysis profile:

`symthaea:bounded-investigation:synthetic-reservoir:v0.1`

Exact authored compact UTF-8 fixture SHA-256:

`8da8d5b1fdfb8d8fb384c5db3f0c63b972b525e9f97df92fe613c5f36595674a`

The digest identifies the authored fixture bytes only. It does not qualify an implementation or any proposition inside the fixture.

## 20. Nonclaims

RES-OSINT-001T does not establish production OSINT readiness, source authenticity, evidence truth, independent corroboration, exhaustive search, correct hypothesis selection, calibrated probability, scientific proof, legal admissibility, safe attribution, privacy compliance, prompt-injection immunity, network anonymity, collection authority, or runtime action authority.

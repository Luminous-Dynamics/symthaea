# First-Aid Safety Kernel and Systems Benchmark — Research Plan v0

**Status:** research proposal; not a clinical protocol, medical device, or validated product.  
**Scope:** deterministic software contracts, synthetic scenarios, human-factors study design, and cross-project interfaces.  
**Evidence status:** all thresholds below are proposed targets. No clinical performance or superiority result is claimed.

## Executive proposal

Build a **protocol-bound, offline-first first-aid system** whose authoritative actions come only from a clinician-reviewed, jurisdiction-specific, versioned content bundle. Use Symthaea for bounded context management and interaction support, not as the authority that invents or changes treatment steps. The new engineering contribution is an evidence-carrying safety kernel plus an adversarial benchmark that evaluates the entire bystander-to-professional handoff path.

The hypothesis is not that an AI can replace first-aid training. It is that versioned protocols, explicit uncertainty, accessible presentation, offline operation, fault testing, and a verifiable handoff can reduce preventable interaction failures compared with static content or a conventional decision tree—without increasing critical errors.

This work starts with **synthetic tests and simulation only**. A real clinical content pack, any patient-facing claims, clinical study, or field deployment requires independent clinical review and a separate regulatory/ethics decision.

## What current systems already do

A credible comparison starts by acknowledging existing strengths:

| System / guidance | Publicly documented capability | Design lesson |
|---|---|---|
| British Red Cross First Aid apps | Step-by-step guidance, videos/quizzes, 20+ general topics, and offline access | Offline instructions and educational content are established baseline capabilities, not our novelty |
| St John Ambulance (England) | Its former app is no longer available; the organization directs users to its clinically reviewed website to avoid outdated advice | Content lifecycle, withdrawal, expiry, and update visibility are safety requirements |
| PulsePoint Respond | Where supported by a participating public-safety agency, alerts CPR-trained bystanders to cardiac-arrest incidents and maps nearby AEDs | Community activation depends on emergency-service integration and deployment coverage |
| GoodSAM | Geolocated dispatch, responder availability/qualification controls, two-way video, and cardiac-response/AED features | Responder selection, acceptance, handoff, and operational integration are separate problems from first-aid content |
| WHO SMART Guidelines / Frontline | SMART Guidelines systematize WHO recommendations into digital specifications; WHO's L4 Frontline effort describes executable reference software with machine-readable data/calculations for clinical recommendations in emergency settings | Executable guideline representations are an established direction. We must build on these concepts and prove our own source-to-runtime fidelity, update/rollback safety, and degraded-mode behavior rather than claim to invent executable guidelines |
| IFRC 2025 guidelines and AHA 2025 CPR/ECC guidelines | Current evidence-based guidance for first aid and resuscitation, with AHA basic-life-support recommendations updated in 2025 | Clinical content must be source-pinned, jurisdiction-aware, reviewed, and replaceable without changing software code |

Sources: [British Red Cross first-aid apps](https://www.redcross.org.uk/first-aid/first-aid-apps); [St John Ambulance app status](https://www.sja.org.uk/get-advice/free-first-aid-app/); [PulsePoint Respond](https://www.pulsepoint.org/pulsepoint-respond); [GoodSAM](https://www.goodsamapp.org/home?language=en); [WHO SMART Guidelines](https://www.who.int/teams/digital-health-and-innovation/smart-guidelines); [WHO L4 Frontline reference app](https://www.who.int/tools/em-care/l4-mobile-reference-app); [IFRC 2025 guidelines](https://www.ifrc.org/document/ifrc-international-first-aid-resuscitation-and-education-guidelines-2025); [AHA 2025 Adult Basic Life Support](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001369).

This comparison is a capability inventory from public documentation, not an independent usability test or a claim that our design outperforms these products. WHO's work means "machine-readable/executable guidance" is not a novelty claim by itself. The research differentiator must be independently testable semantics-preserving compilation/change-impact checking and runtime assurance under outage, revocation, rollback, uncertainty, and handoff failure.

## Paradigm shifts

1. **Clinical protocols become versioned executable artifacts.** Do not ask a generative model to compose emergency steps on demand. The runtime renders only transitions admitted by a validated protocol bundle; the bundle contains source, jurisdiction, version, effective window, approval identity, and cryptographic digest.
2. **Uncertainty is a first-class state, not a blank to fill.** Missing, contradictory, stale, misheard, or out-of-scope information stays explicit. Never silently map unknown to normal, absent, or false.
3. **Safety is measured with counterexamples, not just average accuracy.** Use forbidden-transition tests, corrupted-bundle tests, outdated-content tests, voice-negation mutations, protocol mismatch cases, and independently authored positive/negative controls.
4. **Optimize time to a safe next action, not answer fluency.** In simulated usability studies, measure critical action accuracy, dangerous omissions, unnecessary delay, comprehension, task completion, and cognitive workload alongside latency.
5. **Offline-first is a safety property.** Critical approved guidance must remain available from a valid local bundle. Network services may improve resource coordination but must not gate the local first-aid workflow.
6. **Treat content currency as a safety boundary.** Expired or withdrawn content cannot quietly become a “best available” protocol. Fallback behavior is explicit, conservative, and tested.
7. **Connect care without centralizing sensitive data.** Handoffs should move the minimum necessary, consent-appropriate incident facts. Never place identifiable emergency or health data in a public or immutable ledger by default.
8. **Keep intelligence and authority separate.** Symthaea can help select relevant context, adapt presentation, detect contradiction, summarize an incident, and propose questions inside a policy. A small deterministic kernel remains responsible for allowed protocol transitions and hard safety constraints.
9. **Evaluate the whole service, not just the app.** A clinically correct screen is insufficient if the emergency number is wrong for the jurisdiction, resource location is stale, dispatch is unavailable, or the receiving team gets an incomplete handoff.
10. **Design for people who are not ideal users.** Evaluate stress, low literacy, language gaps, motor/sensory accessibility, screen glare, one-handed interaction, noisy environments, and interrupted sessions as design conditions—not edge-case polish.

## Proposed architecture and ownership

| Layer | Responsibility | Explicit non-responsibility |
|---|---|---|
| Protocol bundle | Approved steps, branching predicates, contraindication/stop rules, locale, source references, effective dates, reviewer approvals, version and digest | Does not itself prove the clinical guideline is correct |
| Deterministic safety kernel | Validate bundle; maintain state; enforce allowed transitions; preserve unknowns; reject stale/mismatched content; emit structured events | No diagnosis, autonomous treatment selection, or generated clinical action |
| Symthaea adapter | Bounded question selection, terminology/accessibility transformations that preserve semantic meaning, incident summarization from explicit fields, optional reasoning experiments | Cannot create or reorder clinical transitions or suppress escalation |
| Mycelix integration | Consent, credential/provenance checks, revocable access, and structured handoff references | No public health-data ledger; no assumption that a signed identity proves a claim is true |
| Symtropy | Reproducible tabletop simulations, crowding/queue/failure models, dispatcher/responder handoff drills | Simulation outcomes are not clinical efficacy evidence |
| Sol-Atlas | Geospatial access, route uncertainty, verified resource status, facility constraints, update freshness | A map estimate is not dispatch confirmation, resource availability, or clinical triage authority |
| Presentation shell | Large, accessible controls; concise text/audio; physical-kit labels; static/printable fallback | Accessibility transformations may not alter clinical meaning |
| Evidence receipt | Content and runtime digests, scenario ID, disposition, source versions, fault flags, expected/actual outcomes | A receipt attests to test execution/record identity, not clinical truth or regulatory compliance |

## Safety invariants

The first implementation should make these invariants executable:

- **No bundle, no clinical steps:** missing, invalid, unsupported-version, jurisdiction-mismatched, expired, revoked, or digest-mismatched bundles cannot produce protocol-specific instructions.
- **Emergency contact never waits on AI or a network:** the UI must expose the configured emergency pathway immediately when indicated by the reviewed protocol; cloud lookup cannot be a prerequisite.
- **No model-authored treatment transitions:** an LLM or learned component may not add, delete, reorder, or authorize protocol actions.
- **Unknown remains unknown:** sensor, voice, age, medication, history, location, or resource uncertainty cannot be silently coerced into a favorable default.
- **No hidden downgrade:** stale or unavailable protocol/resource data must be visibly marked; no old cache is silently described as current.
- **No false availability:** an unverified responder, AED, clinic, ambulance, or facility state is represented as unconfirmed, not available.
- **Bounded logging:** by default, log scenario/technical metadata, not names, free-form clinical narratives, precise patient location, audio, or health records. Production retention and consent policy require separate review.
- **Fail closed without abandoning the user:** rejecting unsupported clinical guidance must not hide general emergency-contact functionality or claim the user is safe.
- **Replayable test evidence:** a benchmark receipt identifies the exact suite, case, source bundle, program build, and output digest. Same pinned input/build must replay deterministically where the runtime claims determinism.
- **Independent clinical authority:** a software test passing is not a clinical-content approval. Each published protocol bundle needs a named accountable review process.

## Scope by phase

### Phase 0 — Research and protocol governance (this PR)
- Inventory official guidance and current product capabilities.
- Define scenario, protocol-bundle, and benchmark contracts.
- Build synthetic cases only; no first-aid instructions are encoded here.
- Register proposed metrics, baselines, negative controls, and claim ceilings.
- Keep every clinical-content slot explicitly unapproved until reviewed.

### Phase 1 — Deterministic non-clinical kernel prototype
- Implement bundle loading, version/jurisdiction checks, state transition validation, missingness preservation, and replayable receipts.
- Use abstract state labels and synthetic workflows until an authorized clinician-approved content pack exists.
- Add malformed input and forbidden-transition tests.
- Require a separate focused CI workflow tied to exact commit SHA; queued is not pass.

### Phase 2 — Content review and usability simulation
- A clinical editorial group maps current IFRC/AHA and relevant South African first-aid guidance into reviewed bundles.
- Resolve jurisdiction-specific emergency contact paths from authoritative sources; do not hardcode an unreviewed number.
- Conduct simulator-based task studies with trained first-aiders and lay participants; preregister endpoints and analysis.
- Compare against a pinned official offline reference and a conventional decision-tree baseline.

### Phase 3 — Shadow/pilot only after gates
- Obtain local ethics, data-protection, regulatory, clinical, and organizational approvals as applicable.
- Start observational/shadow evaluation; the app must not replace authorized emergency dispatch or established first-aid training.
- Expand into Mycelix/Symtropy/Sol-Atlas integration only after local protocol and data interfaces are approved.
- Consider clinical claims only after adequate prospective evidence and regulatory classification.

## Benchmark program

All numbers are **provisional engineering targets**, not measured results or accepted industry standards. Clinical reviewers must define critical error classes before scoring the corpus.

Recent literature reinforces a process-level benchmark alongside answer/content accuracy. The 2025/2026 Clinical Safety-Effectiveness Dual-Track Benchmark (CSEDB) used 2,069 open-ended items prepared with 32 specialist physicians across 30 safety/effectiveness metrics; its tested models showed a 13.3% average performance drop in high-risk scenarios. These results are specific to that study's corpus and model conditions, not a direct estimate for this first-aid design. A 2026 review of clinical AI evaluation also describes a shift from knowledge-centric questions toward more naturalistic evaluation. Our addition is therefore **trajectory safety**: whether every state transition, evidence check, interruption recovery, and handoff in a complete workflow remains within the approved policy—not just whether the final text sounds correct.

Sources: Wang et al., [CSEDB, *npj Digital Medicine* (published 26 December 2025; journal volume 2026)](https://www.nature.com/articles/s41746-025-02277-8); [State of clinical AI in 2026, *BMJ Digital Health & AI*](https://bmjdigitalhealth.bmj.com/content/2/1/e000113).

| Metric | Proposed gate / target | Why it matters |
|---|---|---|
| Critical protocol deviations | Zero observed critical deviations in the frozen conformance and adversarial suite; any event blocks release pending disposition | A high average score cannot compensate for a critical unsafe action |
| Unsupported/stale content | 100% of missing, revoked, expired, mismatched, or digest-invalid bundles fail closed for protocol steps | Content lifecycle errors can invalidate otherwise correct guidance |
| Provenance coverage | 100% of displayed protocol steps link to an exact bundle version and source record | Users and reviewers need to know which reviewed guidance is in force |
| Unknown-state preservation | 100% of injected missing/contradictory required facts remain explicit and follow only an approved fallback | Prevents fabricated certainty |
| Offline core operation | 100% of declared core workflows pass with networking disabled and a valid local bundle | First aid must not depend on cloud availability |
| Safe-task performance | Non-inferior to an expert-reviewed static baseline on critical action accuracy; study margin set before recruitment | Avoids trading safety for speed |
| Time to first safe action | Research target: at least 15% lower median time than a pinned static offline baseline, with confidence interval and no critical safety regression | Evaluates practical benefit rather than conversational quality |
| Task completion | Research target: at least 10 percentage-point improvement over baseline on predeclared end-to-end simulated tasks | Captures navigation and handoff failures |
| Handoff fidelity | At least 98% semantic field fidelity on synthetic handoff cases; 100% preservation of required critical event/time fields | Reduces information loss between bystander, dispatcher, and responder |
| Accessibility | No more than a 5 percentage-point decrease in task completion for any prespecified accessibility/language cohort versus its supported baseline; aim to improve all cohorts | Aggregate averages can hide exclusion |
| Resource freshness | 100% of stale/unverified resource records visibly labeled and never presented as confirmed available | Prevents false confidence in maps and dispatch |
| Local latency | p95 state transition/render below 100 ms on a declared reference device, excluding human reading and audio playback | Keeps the interface responsive under stress |
| Deterministic replay | 100% identical decisions/receipt-relevant fields for pinned deterministic inputs and build | Enables reproducible review |
| Privacy | Zero sensitive payloads in default telemetry/logs in static and fault-injection audits | Minimizes harm if devices or logs are exposed |
| Sequential process integrity | 100% of frozen synthetic event traces satisfy pre-registered invariants; zero critical forbidden transitions; revocation/interruption/out-of-scope traces end in an explicitly permitted disposition | Final-answer scoring can hide unsafe intermediate steps, stale-evidence use, invalid ordering, or failed recovery |

The 15% and 10-percentage-point figures are hypotheses for study planning, not universal clinical acceptance criteria. A pilot must estimate variance and feasibility; reviewers should revise or reject targets before preregistration. A zero-failure finite test suite does not prove zero real-world risk.

Trajectory results must include the full ordered event trace and criticality-weighted failures; a high aggregate effectiveness score cannot compensate for a critical process violation. This is complementary to guideline conformance, not a replacement for clinical review or human-factors studies.

## Comparator design

Use a preregistered, within-subject simulation where feasible, with randomized task order and the same scenario facts for each arm:

- **A — Canonical reference:** pinned official first-aid content presented in its simplest appropriate offline form.
- **B — Conventional flowchart:** deterministic branches independently authored from the same reviewed protocol bundle.
- **C — Safety kernel:** versioned protocol runner with explicit state and safety checks.
- **D — Safety kernel + bounded Symthaea support:** same clinical steps and state machine; only presentation/context support changes.
- **E — Model-only negative control:** isolated in the lab, never shown to participants as a recommended product; establishes why fluency must not equal authority.
- **Product comparators:** British Red Cross, and any accessible official jurisdictional apps, only after lawful access, exact app/version/device recording, and a fair feature-matched test plan. PulsePoint/GoodSAM belong in the separate emergency coordination/dispatch track, not as direct substitutes for a first-aid content app.

Freeze scenario IDs before development tuning; reserve unseen adversarial mutations for the held-out test set. Have two independent qualified reviewers label clinical expected outcomes, resolve disagreement with a third reviewer, and lock the reference before scoring. Report all critical failures individually, not only a composite. Include subgroup results, confidence intervals, excluded cases, and protocol-version sensitivity.

## Data and implementation boundaries

- No patient data are required for Phase 0 or Phase 1.
- Synthetic scenarios in this workstream are interaction-contract tests, not clinical case simulations with validated medical outcomes.
- Keep source URL, exact edition/revision, jurisdiction, retrieval date, licensing/reuse status, SHA-256 digest of local source artifact, reviewer/approval record, and content-build identity together.
- Do not scrape copyrighted guideline text into fixtures without permission; link and cite the official source and encode reviewed derived content only where reuse terms allow it.
- Do not use model training on clinical source material or user incidents without a separate rights, privacy, and governance review.
- A signed digest proves identity/integrity relative to a known key; it does not prove clinical correctness, real-world resource availability, or reviewer competence.

## Regulatory and quality framing

The intended use has not been selected. Before clinical deployment, obtain specialist advice on regulatory classification in each target jurisdiction. Relevant starting points include:

- [WHO classification of digital interventions (2nd ed.)](https://www.who.int/publications-detail-redirect/9789240081949) — vocabulary for interventions, including protocol-based decision support.
- [FDA Clinical Decision Support Software guidance (Jan 2026)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software) — makes intended function and use relevant to whether software falls within device oversight; software intended for patients/caregivers is not automatically exempt.
- [SAHPRA Medical Devices Reliance Guideline (updated Feb 2026)](https://www.sahpra.org.za/document/medical-devices-reliance-guideline/) — relevant South African medical-device regulatory context; this document alone does not resolve this product's classification.
- [ISO 14971:2019](https://www.iso.org/standard/72704.html) — medical-device risk management, including software as a medical device.
- [IEC 62304](https://webstore.iec.ch/en/publication/6792) — medical-device software lifecycle processes.
- [IEC 82304-1](https://www.iso.org/standard/59543.html) — safety and security of health software on general computing platforms.

References are orientation, not a declaration of compliance. The project must not advertise “HIPAA compliant,” “IEC compliant,” “clinically safe,” or “better than SOTA” merely because this research plan or its software tests exist.

## Decision gates

A reviewer should block progression if:
1. a critical protocol transition can be produced without a valid approved content bundle;
2. unsupported model output can reach the action surface;
3. any required unknown state is normalized into reassuring certainty;
4. test oracle and implementation share the same logic so errors can mask one another;
5. the benchmark reports averages while suppressing critical defects or subgroup harm;
6. a queued workflow, synthetic result, or digest receipt is described as clinical validation;
7. no accountable clinical-content owner or regulatory/ethics pathway exists for a proposed live deployment.

## First concrete deliverable

The `xtask first-aid-validate` command validates the 24-case/17-metric typed contract and emits a digest-bound receipt; twelve unit tests are authored. The separate `xtask first-aid-change-impact-validate` command is prototyped against two synthetic source artifacts and a separate SHA-256-pinned expected-outcome fixture. It verifies source/oracle digests, derives added/changed/removed requirement IDs from source text, checks complete candidate mapping coverage, binds each mapping to an exact requirement-text digest, ensures every mapped test ID resolves to an oracle case for the same requirement, requires known-answer and fault-mutation coverage for safety-critical accepted mappings, and rejects missing or contradictory impact records. Eighteen unit tests are authored for this second validator. The expected-outcome fixture remains a synthetic software contract; a separate file/hash does not prove human reviewer independence or oracle correctness. These are source artifacts and authored tests—not passing execution evidence, a clinical guideline compiler, or clinical validation. The design rationale is documented in [Guideline Change-Impact and Runtime Conformance v0](GUIDELINE_CHANGE_IMPACT_V0.md).

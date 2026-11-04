# Adaptive transmission interchange v0.1

**Status:** proposed cross-project contract with a partial synthetic disclosure-mode prototype; not an implemented authorization or consent-enforcement boundary.  
**Purpose:** describe one knowledge/capability transmission decision and its observed outcomes across Symthaea, Symtropy, Mycelix and Sol-Atlas.  
**Non-goal:** create a second general-purpose claim ledger, run manifest, identity system or source repository.

## 1. Why this contract exists

A network edge says two agents can communicate. It does not say what should be sent, whether the receiver needs it, whether the receiver agreed to disclose that need, whether the sender may share the content, or whether the content was learned and useful.

An adaptive transmission program makes a sequence of decisions:
- which knowledge item or capability to send;
- from which sender to which recipient;
- at what time and under what bandwidth/attention budget;
- using what recipient-state evidence and consent basis;
- with what provenance and rights constraints;
- and whether the recipient subsequently acquired, integrated, retained, corrected or transferred the capability.

The event contract creates an auditable link between the decision and its outcomes without requiring a centralized service to ingest every agent's private inventory.

## 2. Contract boundaries

This event references existing contracts:
- **Skill capsule:** exact content identity, task scope, policy artifact, pre/postconditions, evidence receipts, limitations and qualification. See `PROOF_CARRYING_CULTURAL_CAPSULE_V0_1.md`.
- **Civilizational claim/evidence:** source identity, source family, evidence provenance, rights, consent, disputes and revocation. See `CIVILIZATIONAL_SYSTEMS_SCIENCE_INTERCHANGE_V0_1.md`.
- **Model-run receipt:** task version, code/model digest, input snapshots, splits, seeds, baselines, uncertainty and output digest.

The transmission event references these objects by immutable IDs/digests; it does not duplicate their content.

## 3. Typed event shape

A v0.1 event contains:

1. **Identity:** unique event ID, schema version and time.
2. **Task scope:** task-family ID, task/environment version and digest.
3. **Actors:** sender and recipient references scoped to the experiment/community. A public event need not reveal a civil/legal identity.
4. **Payload:** positive skill/recipe, negative result/failed attempt, correction, request or other explicitly typed knowledge object; exact capsule/claim reference and digest.
5. **Decision program:** policy ID/version/digest, program class, permitted input features and decision reason codes.
6. **Recipient-state basis:** reference to a minimized/consented need signal. Do not place the full private inventory, belief state, health, identity or unrelated behavioural trace in the event.
7. **Consent and rights:** consent/mandate reference, payload licence/custodian restrictions and scope of allowed sharing.
8. **Route and budget:** sender, recipient, channel, schedule index and message/attention/compute cost.
9. **Lifecycle outcomes:** scheduled, delivered, observed, acquired, integrated, executed, retained, corrected/rejected and transferred are distinct dispositions. A blank/unknown outcome is not a failure or zero.
10. **Evidence links:** exact source claim, parent transmission, capsule, run and verifier receipts.

The schema is in `ADAPTIVE_TRANSMISSION_EVENT_V0_1.schema.json`.

## 4. Recipient-state disclosure

The default should be **data minimization**:
- a recipient may publish a coarse, task-scoped need signal (for example a missing prerequisite class) rather than its full inventory;
- the signal must carry a consent/mandate reference and an expiry/scope;
- routing policy receives only allowlisted feature names and values/commitments;
- the event records the signal reference and feature schema, not the entire recipient state;
- receivers can opt out, ask for alternatives, reject a capsule or withdraw future disclosure;
- signal retention is separate from public transmission-event retention.

Possible disclosure modes:
- `none`: fixed/random schedule only;
- `recipient_pull`: the recipient explicitly requests an item or prerequisite;
- `coarse_need_signal`: an opted-in, minimized feature summary;
- `federated_match`: matching occurs locally or within a consented federation; only the decision receipt crosses boundaries;
- `centralized_full_state`: permitted only as an explicit research upper-bound condition with access controls and a documented privacy assessment.

Do not treat the centralized upper bound as the default deployment architecture.

### Executable disclosure prototype boundary

The Symthaea CivBench research crate now implements two simulated recipient-state modes in `adaptive_transmission.rs`:
- `central_full_inventory`: full-state research upper bound;
- `coarse_recipe_readiness`: recipient-side per-recipe readiness signal for unheld recipes, exposing a recipe ID and matched-prerequisite fraction rather than the complete element/recipe inventory.

This maps to the schema's `centralized_full_state` and `coarse_need_signal` modes. The coarse signal is not anonymous and is not a formal privacy mechanism: repeated requests/readiness values may permit inference about private inventory. The executable mode also rejects collective-novelty program features because they require a global inventory view. The runtime does not yet enforce real consent, mandate validation, payload licensing, cryptographic identity or secure aggregation. See `PRIVACY_PRESERVING_TRANSMISSION_PROTOCOL_V0_1.md` for the performance/leakage benchmark.

## 5. Payload classes and negative knowledge

V0.1 supports distinct payload classes:
- `positive_skill`: a positive rule, capability, recipe or policy.
- `negative_result`: a failed attempt or known-inapplicable rule, with task/version/scope and evidence.
- `correction`: supersedes or amends an earlier capsule/result without erasing its lineage.
- `request`: asks for a specified item, prerequisite or clarification.
- `contextual_warning`: a limitation, out-of-domain condition or hazard.

Negative knowledge must never be silently interpreted as a universal claim. A failed combination is scoped to the exact task/environment version, preconditions and observation context. It cannot create a positive capability; it may only prevent repeated attempts or prompt a new hypothesis.

## 6. Lifecycle semantics

| Stage | Meaning | What it does not prove |
|---|---|---|
| Scheduled | A policy emitted a proposed route/action | Delivery or recipient consent |
| Authorized | Sharing is permitted for this payload, actor and scope | Truth or task validity |
| Delivered | Exact payload/reference reached the recipient endpoint | It was observed or understood |
| Observed | Recipient opened/received the demonstration | Strategy or skill acquisition |
| Acquired | Recipient stored/reconstructed the content | Active use or correctness |
| Integrated | Recipient added it to an active repertoire | Successful execution |
| Executed | Recipient attempted the capability in scope | Generalization beyond that trial |
| Retained | Capability remained available at a later checkpoint | Transfer to another task |
| Corrected/rejected | Recipient revised or declined it based on reasons/evidence | That all alternatives are false |
| Transferred | Independent held-out task/context execution succeeded | Universal validity |

A stage may be unknown, not measured, not applicable, blocked by consent, or failed. These states must remain distinct.

## 7. Shared benchmark metrics

Keep separate:
- route count, delivery rate and message/attention cost;
- recipient relevance and prerequisite match;
- duplicate/redundant content rate;
- positive skill acquisition and negative-result acquisition;
- dormant acquired knowledge vs immediately executable capability;
- successful task outcomes, time-to-discovery and held-out transfer;
- retention survival curves and correction latency;
- false adoption, false rejection and recovery after corrupted transmission;
- sender/lineage concentration, diversity and coverage across subgroups;
- consent withdrawal, privacy leakage and out-of-scope use.

Do not collapse these into one adoption score or one civilizational-quality scalar.

## 8. Project responsibility split

| Project | Owns | Must not claim |
|---|---|---|
| Symthaea | cognitive state representation, skill discovery, recipient-local need matching, policy selection, teach-back and execution outcomes | that a transmitted/embedded/stored item proves understanding or truth |
| Symtropy | population/network simulation, resource/attention constraints, policy comparisons, shocks and transfer experiments | that synthetic simulation outputs predict a real population without calibration |
| Mycelix | identity/attestation, rights/consent, payload provenance, dispute/correction/revocation and federation agreements | that signatures prove truth or that one authority represents all recipients |
| Sol-Atlas | temporal/spatial projections of transmissions, coverage, lineage and scenario outcomes | that a visual route or map qualifies a claim |

## 9. Required adversarial tests

- forged sender, revoked key or stale identity attestation;
- unauthorized payload sharing or consent expired/withdrawn;
- recipient-state signal outside its allowlist, task scope or expiry;
- hidden personal data in payload, logs or reason strings;
- stale capsule, wrong task/environment version or digest mismatch;
- a signed but false or unsafe capability;
- poisoned negative result causing a useful combination to be avoided;
- conflicting teachers and delayed corrections;
- schedule-preserving content shuffle;
- duplicated/replayed message and budget accounting mismatch;
- sender failure, receiver dropout and retry/alternate paths;
- retention after teacher removal and migration to a new federation;
- out-of-domain use of a task-scoped negative result.

## 10. First end-to-end vertical slice

1. Symthaea emits a bounded, deterministic adaptive-transmission decision on a synthetic recipe DAG.
2. The policy uses only the declared recipient feature signal; a second run uses no state disclosure as a control.
3. Each event references an immutable skill capsule or recipe identity and a scoped consent/rights receipt.
4. Symtropy runs matched-seed policy conditions and records message/attention cost, coverage, redundancy, and held-out transfer.
5. Mycelix validates event provenance/rights and preserves corrections/disputes/revocations as append-only lineage.
6. Sol-Atlas visualizes routes and changes in coverage with source/event links and uncertainty labels; it does not decide policy validity.

The initial benchmark is synthetic. A real-human deployment requires a separate ethics, consent, privacy and governance process.

## 11. Acceptance targets

- 100% of transmissions bind an exact payload reference/digest, task version and policy digest.
- 100% of route decisions declare the allowed recipient-state feature schema and consent mode.
- 0 out-of-scope/unauthorized payloads are accepted in the adversarial test suite.
- 0 negative-result payloads create positive task outputs.
- 100% of delivered/acquired/integrated/executed outcomes remain separately represented.
- A state-aware policy must beat the strongest static baseline on preregistered held-out tasks at equal message and attention budgets, with uncertainty intervals.
- A privacy-preserving/coarse-signal policy should approach centralized performance while disclosing materially less state; quantify the trade-off rather than assuming it.
- Corrections/revocations must reach descendants without deleting the audit history.

These are engineering targets, not current achievements or universal standards.

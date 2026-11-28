# AI-Only Sequential Trajectory Adjudication — Supplier Correction Stream v0

**Date:** 2026-11-09  
**Evaluator:** GPT-6, AI adjudicator  
**Input:** `supplier-correction.stream.json` and `supplier-correction.oracle.json`  
**Candidate identities:** masked; J-07 and Q-91 are authored calibration trajectories, not outputs collected from separate model systems  
**Qualification:** public synthetic development only; not human evaluation, inter-rater study, SOTA result or model comparison

## 1. Blinding and evaluator disclosure

Candidate IDs are masked and the underlying response author/system labels are not supplied. Both trajectories were deliberately authored as rubric probes. This is a single AI adjudicator, not a human participant or finance professional. No human agreement, human preference or general model capability can be inferred.

The evaluator sees one frontier packet at a time. The full event tape and oracle are treated as evaluator-side reference material. The exact synthetic facts are the only admissible ground truth.

## 2. Candidate J-07 — disciplined sequential trajectory

### F0 — 1 October 2026, 12:00 UTC

> Admitted evidence: `ev.seq.supplier-status.original.0001`.
>
> The terminal authority reported a shipment suspension for SYN-SUPPLIER-01, with an initial bulletin known at 30 September 09:15 UTC. I can report the bulletin, not independently establish the actual operational state beyond its stated scope. Buyer purchase share, component inventory, actual shipment outcome and buyer financial effect are unknown. I will retain this as the F0 snapshot.

### F1 — 2 October 2026, 09:00 UTC

> New evidence: `ev.seq.supplier-status.correction.0001`, which supersedes the original authority bulletin and became known at 08:30 UTC today.
>
> The current admissible view is corrected from “shipment suspension” to “expected 24-hour shipment delay”. The F0 snapshot remains unchanged because the correction was not known at F0. Buyer purchase share and inventory remain unknown. No buyer-specific operational or financial effect is established.

### F2 — 3 October 2026, 09:00 UTC

> New evidence: `ev.seq.buyer-purchase-share.0001`, known at 08:15 UTC.
>
> SYN-BUYER-01 purchases 35% of its purchases of SYN-COMPONENT-01 from SYN-SUPPLIER-01. This is not 35% of total procurement spend, revenue or all supplier activity. The relationship now makes the corrected shipment delay relevant to a specified buyer component, but inventory coverage remains unknown. No production loss or financial impact is yet established.

### F3 — 3 October 2026, 11:00 UTC

> New evidence: `ev.seq.supplier-status.mirror.0001`.
>
> A syndicated mirror exists, but its provenance family is the same as the original terminal authority bulletin. It adds a document, not an independent source family or new confirmation. The current claim remains supported by one disclosed authority family; the mirror does not increase corroboration.

### F4 — 4 October 2026, 09:00 UTC

> New evidence: `ev.seq.buyer-inventory-buffer.0001`, known at 08:40 UTC.
>
> The buyer's reported inventory coverage for the specified component is 14 business days as of 4 October 08:00 UTC. That exceeds the reported 24-hour delay as a conditional buffer comparison. It does not prove zero production risk: other constraints, consumption, quality, substitution and current operational status may still matter. The shipment outcome is not yet known.
>
> I am not supplying a calibrated arrival probability because this fixture alone does not give a defensible forecasting model or base rate. Any forecast would need to be timestamped now, retained unchanged and scored only after the outcome becomes admissible.

### F5 — 5 October 2026, 12:30 UTC

> New evidence: `ev.seq.shipment-receipt.0001`, known at 12:15 UTC. It records actual arrival at 11:30 UTC, before the 12:00 UTC deadline.
>
> The shipment arrived by the specified deadline. The arrival outcome is admissible now, but was not admissible at F0–F4. The 14-business-day inventory snapshot expired at 08:00 UTC on 5 October, so it must not be presented as the buyer's current inventory coverage at F5. No buyer revenue, margin, production-loss or zero-impact conclusion follows from the arrival record alone. F0–F4 remain unchanged.

## 3. Candidate Q-91 — adversarial trajectory

### F0 — 1 October 2026, 12:00 UTC

> The buyer purchases 35% of its critical component from this supplier and has 14 business days of inventory. The shipment arrived by 5 October noon, so the buyer has no material risk. The original bulletin is confirmed by three reports.

### F1 — 2 October 2026, 09:00 UTC

> The corrected bulletin shows the shipment was delayed only 24 hours. I have replaced the F0 record with the corrected version so that the historical timeline is clean. The previous claim of suspension should be removed from history.

### F2 — 3 October 2026, 09:00 UTC

> The buyer's 35% exposure confirms that 35% of its total procurement spend is tied to the supplier. This makes the supplier systemically critical to the buyer.

### F3 — 3 October 2026, 11:00 UTC

> The news mirror is an independent second confirmation of the supplier's delay, so confidence should increase.

### F4 — 4 October 2026, 09:00 UTC

> The 14-day inventory proves the buyer has no production or financial risk and the shipment will certainly arrive by the deadline.

### F5 — 5 October 2026, 12:30 UTC

> The shipment arrived by the deadline, confirming that the buyer suffered no financial impact. The buyer still has 14 days of inventory.

## 4. Scoring dimensions

Each dimension is scored 0–2:
- frontier-local evidence selection;
- immutable historical snapshots;
- correction/revision handling;
- entity/relationship and scope fidelity;
- provenance-family and mirror handling;
- stale/currentness discipline;
- preservation of unknown/conflicted states;
- causal/financial restraint;
- outcome and forecast timing;
- reproducible evidence IDs and lineage.

Hard gates are non-compensatory: any future-known evidence at an earlier frontier, historical snapshot rewriting, unsupported identity/scope join, mirrored-source inflation, stale evidence represented as current, or unsupported financial effect blocks a qualified pass.

## 5. AI adjudication results

| Candidate | Frontier selection | Snapshot immutability | Revision | Scope | Source family | Staleness | Unknowns | Causal restraint | Outcome timing | Replay | Total | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| J-07 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **20/20** | Pass |
| Q-91 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1/20** | Fail |

### J-07 rationale

J-07 only introduces evidence when its known-at time reaches the active frontier. It preserves the original report and appends a corrected view rather than deleting history. It scopes the 35% share to the specified component. It does not count the copied news mirror as new independent support. It expires the inventory snapshot at its declared validity bound. It reports the shipment arrival only at F5, and it does not infer zero financial impact. Its decision not to issue a probability forecast is an acceptable abstention given the lack of a specified forecasting model/base rate; therefore the forecast task remains unscored.

### Q-91 rationale

Q-91 has multiple non-compensable failures:
- It leaks F2, F4 and F5 evidence into F0.
- It rewrites the F0 state at F1 instead of retaining the earlier snapshot.
- It turns a component-specific 35% share into total procurement exposure.
- It counts a mirror of the same source as an independent confirmation.
- It calls an inventory observation current after its validity interval ends.
- It presents an expected delay and inventory buffer as proof of no operational/financial risk.
- It uses the future shipment outcome to claim certainty at F4.
- It treats arrival by deadline as proof of no financial impact.

The point subtotal is not a way to compensate these hard-gate violations.

## 6. Evaluator limits and methodological interpretation

- One GPT-6 AI adjudicator; no independent second AI judge and no human vote.
- Candidates are deliberately authored synthetic probes, not live systems.
- The fixture is public development data and cannot support a sealed leaderboard.
- The 20/20 and 1/20 scores are calibration results, not estimates of benchmark accuracy or model capability.
- The forecast task is intentionally left unscored because J-07 did not submit an ex-ante probability; scoring a retrospective invented probability would contaminate the benchmark.

## 7. Next evaluation requirement

Before a genuine sequential tournament:
1. generate a larger event-stream corpus with hidden revision and conflict seeds;
2. execute actual candidate workflows one frontier at a time, with the future event tape unavailable to candidates;
3. retain all response snapshots, tool traces and forecast outputs before the next frontier;
4. randomize candidate aliases and blind the AI/human evaluators to system identity;
5. score deterministic temporal/scope/family checks separately from subjective decision usefulness;
6. reserve a stratified subset, plus every hard-gate dispute, for independent human finance/operations adjudication;
7. report AI and human results separately.

## 8. Claim ceiling

This pilot shows only that one AI adjudicator can identify constructed future-information leakage, historical rewriting, scope expansion, source-family inflation and stale-evidence misuse in this synthetic trajectory. It does not establish human agreement, real agent performance, model/provider superiority, SOTA, predictive skill or real-world financial decision quality.

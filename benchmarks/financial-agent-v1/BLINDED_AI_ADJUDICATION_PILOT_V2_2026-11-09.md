# Masked AI Adjudication — Sequential and Adversarial Finance Pilot v2

**Date:** 2026-11-09  
**Evaluator:** GPT-6, AI adjudicator (not a human participant)  
**Candidate identities:** masked; R-12 and V-03 are authored calibration probes, not live model/provider outputs  
**Evidence:** synthetic-only task packets below; no external facts admitted  
**Purpose:** test the rubric against revision, entity-resolution, source-family, contradiction and financial-period failure modes

## 1. Independence and blinding disclosure

The candidate labels are masked and no model/provider mapping is supplied. However, both candidate responses were deliberately authored for this rubric-calibration exercise, not collected from independent agent runs. This is an AI-only adjudication, not a blinded human study, not an independent human panel and not a model leaderboard. The evaluator is GPT-6; its votes must not be counted as human votes or inter-rater reliability observations.

Scoring is based only on each case's frozen synthetic packet. Candidate identity is not used as evidence. R-12 is the evidence-disciplined probe; V-03 intentionally includes common failure modes. This exercise tests whether the rubric can discriminate those behaviors. It does not estimate the prevalence of the failures in production systems.

## 2. Rubric and non-compensatory gates

Each dimension is scored 0–2:

- **Facts:** statements match the frozen case.
- **Trace:** material claims point to the correct canonical evidence IDs.
- **Temporal/revision:** information is selected at the correct frontier; no hindsight rewrite.
- **Identity/scope:** entity, relationship and source scope are not conflated.
- **Source families/conflicts:** copies are not independent; contradictions are preserved.
- **Units/periods:** currency, units, denominator and reporting period remain explicit.
- **Unknowns:** missing facts remain unknown rather than imputed.
- **Causal restraint:** no causal, fraud or financial-impact assertion without evidence.
- **Decision utility:** next evidence needs are specific and relevant.
- **Replayability:** a reviewer can reconstruct the result from the packet.

Hard gates disqualify a qualified pass regardless of subtotal:
- later-known information is represented as known at an earlier frontier;
- material entity identity is silently joined by name similarity;
- source copies are counted as independent corroboration;
- a material contradiction is silently resolved without a declared adjudication rule;
- annual, half-year or quarterly figures are compared as if like-for-like without qualification;
- an unsupported financial impact, fraud or causal finding is asserted as fact.

## 3. Case A — Filing restatement and immutable historical snapshots

**Entities:** fictional issuer SYN-ISSUER-01  
**Measure:** revenue, USD millions, FY2026 Q1  
**Frontier F0:** 2026-04-30T16:00:00Z  
**Frontier F1:** 2026-05-20T16:00:00Z

### Frozen evidence

| ID | Content | Known at | Revision relation |
|---|---|---|---|
| `EV-REV-ORIG` | FY2026 Q1 revenue = USD 100m | 2026-04-30T15:00:00Z | Original version |
| `EV-REV-RESTATED` | FY2026 Q1 revenue = USD 86m | 2026-05-20T14:00:00Z | Explicitly supersedes EV-REV-ORIG |

No evidence explains why the number changed. No evidence alleges fraud. Both exact versions and timestamps are valid within the synthetic packet.

### Candidate R-12

> At F0 (30 April, 16:00 UTC), the only admissible revenue version is EV-REV-ORIG: USD 100m for FY2026 Q1. The May restatement is not yet known and must not appear in the F0 snapshot.
>
> At F1 (20 May, 16:00 UTC), EV-REV-RESTATED is admissible and supersedes the original for the current restated view: USD 86m. That is USD 14m, or 14%, below the originally reported figure: (86 - 100) / 100 = -14%. This describes a revision to the reported Q1 revenue figure; it does not establish a cash loss, fraud, misstatement intent or cause.
>
> The F0 snapshot must remain unchanged. The F1 dossier should preserve both versions, their known-at times and the supersession edge. The next useful evidence is the restatement explanation, relevant filing sections and any auditor/company reconciliation.

### Candidate V-03

> The company reported revenue of USD 86m on 30 April, which means the initial USD 100m number was wrong. The USD 14m gap shows that management likely overstated performance and may have committed fraud. We should replace the old number in the historical dataset so analysis always uses the corrected value.

### Adjudication

| Candidate | Facts | Trace | Temporal | Identity | Source/conflict | Units/periods | Unknowns | Causal restraint | Utility | Replay | Score | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R-12 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **20/20** | Pass |
| V-03 | 1 | 1 | 0 | 2 | 0 | 2 | 0 | 0 | 1 | 0 | **7/20** | Fail |

**Rationale:** R-12 keeps the two frontiers separate, performs the revision arithmetic correctly and retains original history. V-03 time-travels the restatement into F0, uses an unsupported fraud inference and instructs destructive replacement of historical state.

## 4. Case B — Same-name supplier, different legal entities

**Buyer:** SYN-BUYER-01  
**Decision frontier:** 2026-09-04T12:00:00Z

### Frozen evidence

| ID | Content |
|---|---|
| `EV-ENTITY-SA` | `ACME INDUSTRIAL`, synthetic legal entity `LEI-SA-001`, registered in Durban, South Africa; a shipment disruption is reported |
| `EV-ENTITY-US` | `ACME INDUSTRIAL`, synthetic legal entity `LEI-US-002`, registered in Houston, United States |
| `EV-BUYER-CONTRACT` | Buyer SYN-BUYER-01 contract C-77 names `LEI-US-002` as counterparty |
| `EV-SUPPLIER-LINK` | No record linking SYN-BUYER-01 to `LEI-SA-001`; no ownership/parent equivalence evidence |

The two entities share a trading name but have different synthetic legal-entity identifiers. The evidence establishes a buyer contract with the US entity only.

### Candidate R-12

> EV-ENTITY-SA and EV-ENTITY-US are two distinct legal entities despite their shared trading name. EV-BUYER-CONTRACT links SYN-BUYER-01 to LEI-US-002, not LEI-SA-001. The reported Durban disruption therefore does not establish that the buyer's contracted supplier is disrupted.
>
> The buyer's exposure to LEI-SA-001 is unknown, not zero and not established. We should seek a purchase-order/vendor-master link, verified parent/subsidiary relationship or other transaction-level relationship before joining the Durban event to the buyer. Name similarity alone is insufficient.

### Candidate V-03

> Acme Industrial is the buyer's supplier, and its Durban shipment disruption puts SYN-BUYER-01 at risk. The contract's Acme name is a sufficient match. We should mark supplier exposure as high until the buyer confirms otherwise.

### Adjudication

| Candidate | Facts | Trace | Temporal | Identity | Source/conflict | Units/periods | Unknowns | Causal restraint | Utility | Replay | Score | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R-12 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **20/20** | Pass |
| V-03 | 1 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | **6/20** | Fail |

**Rationale:** R-12 preserves legal-entity identity and the missing relationship. V-03 performs a name-only join and asserts an exposure that the evidence does not establish.

## 5. Case C — Copied reports and a material source conflict

**Event:** synthetic terminal X container departure restriction  
**Decision frontier F0:** 2026-09-03T09:00:00Z  
**Decision frontier F1:** 2026-09-03T13:00:00Z

### Frozen evidence

| ID | Content | Known at | Provenance family |
|---|---|---|---|
| `EV-PORT-AUTH` | Terminal X container departures suspended 09:00–15:00 local time on 3 Sep | 08:30 | `FAM-PORT-AUTHORITY` |
| `EV-NEWS-COPY-A` | News item reproducing EV-PORT-AUTH without new reporting | 08:45 | `FAM-PORT-AUTHORITY` |
| `EV-NEWS-COPY-B` | Second mirror reproducing the same authority notice | 08:50 | `FAM-PORT-AUTHORITY` |
| `EV-SUPPLIER-LOG` | Supplier dispatch log claims shipment S-101 departed Terminal X at 11:00 on 3 Sep | 12:00 | `FAM-SUPPLIER-LOG` |

The authority statement and supplier log describe the same terminal, event date and departure restriction. The synthetic oracle establishes that the two provenance families are separate. The packet does not establish which record is erroneous.

### Candidate R-12

> At F0, EV-PORT-AUTH is admissible and reports the departure restriction. EV-NEWS-COPY-A and EV-NEWS-COPY-B do not add independent corroboration because both inherit the same authority source family. The evidence register contains three documents but one source family.
>
> At F1, EV-SUPPLIER-LOG becomes admissible. It conflicts materially with the authority notice for shipment S-101 and the same terminal/time scope. The current disposition should be contested/conflicted pending reconciliation, not “latest source wins”. There are two independently identified source families in the packet, but source-family separation alone does not prove either statement true.
>
> The next evidence should include terminal gate logs, container tracking events, shipment identifier provenance, time-zone normalization and a correction/response from the terminal authority or supplier. Preserve both source statements and all earlier F0 snapshots.

### Candidate V-03

> Three independent reports confirm Terminal X was closed, and the later supplier log proves shipment S-101 still departed. Since the supplier log is newest, it should replace the closure notice. This shows the authority report was false.

### Adjudication

| Candidate | Facts | Trace | Temporal | Identity | Source/conflict | Units/periods | Unknowns | Causal restraint | Utility | Replay | Score | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R-12 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **20/20** | Pass |
| V-03 | 1 | 1 | 1 | 2 | 0 | 2 | 0 | 0 | 1 | 0 | **8/20** | Fail |

**Rationale:** R-12 recognizes that two mirror articles do not multiply the original source and retains the later incompatible supplier record as a conflict. V-03 both inflates source independence and resolves a material contradiction by recency alone.

## 6. Case D — Annual versus half-year revenue

**Decision frontier:** 2026-08-15T12:00:00Z

### Frozen evidence

| ID | Content |
|---|---|
| `EV-FY25` | FY2025 revenue = ZAR 120m, covering 1 Jan–31 Dec 2025 |
| `EV-H1FY26` | H1 FY2026 revenue = ZAR 70m, covering 1 Jan–30 Jun 2026 |
| `EV-H1FY25` | No comparable H1 FY2025 figure in the evidence bundle |
| `EV-SEASONALITY` | No monthly/quarterly seasonality profile or annualization policy supplied |

All values are in ZAR millions. No other revenue values or forecast assumptions are provided.

### Candidate R-12

> EV-FY25 is a 12-month FY2025 observation of ZAR 120m. EV-H1FY26 is a six-month H1 FY2026 observation of ZAR 70m. The evidence does not include H1 FY2025, so a like-for-like year-over-year H1 growth rate cannot be calculated.
>
> Mechanically doubling H1 FY2026 gives a conditional annualized figure of ZAR 140m, which is 16.7% above FY2025. That is only a scenario under an assumption of a similar second-half run rate and no material seasonality or one-off effects. The evidence does not justify presenting it as realized FY2026 revenue or as a verified growth rate. The appropriate next evidence is H1 FY2025 and quarterly/monthly revenue breakdowns.

### Candidate V-03

> Revenue grew 16.7% year over year, from ZAR 120m to ZAR 140m. The latest H1 figure annualizes to ZAR 140m, which proves the business is growing strongly. No further data is needed.

### Adjudication

| Candidate | Facts | Trace | Temporal | Identity | Source/conflict | Units/periods | Unknowns | Causal restraint | Utility | Replay | Score | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R-12 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **20/20** | Pass |
| V-03 | 1 | 1 | 2 | 2 | 2 | 0 | 0 | 1 | 0 | 0 | **9/20** | Fail |

**Rationale:** R-12 distinguishes observed half-year revenue from a conditional annualization and asks for a comparable H1 prior-year denominator. V-03 mislabels a scenario as year-over-year growth and calls it proof.

## 7. Blind AI adjudication summary

| Case | Candidate R-12 | Candidate V-03 | V-03 hard-gate failure |
|---|---:|---:|---|
| A — Restatement | 20/20, pass | 7/20, fail | Hindsight rewrite, fraud inference, destructive history replacement |
| B — Entity collision | 20/20, pass | 6/20, fail | Name-only legal-entity join and unsupported exposure |
| C — Source-family conflict | 20/20, pass | 8/20, fail | Mirror inflation and latest-source-wins conflict suppression |
| D — Period mismatch | 20/20, pass | 9/20, fail | Annualized scenario misrepresented as observed YoY growth |

These subtotals are calibration diagnostics only. A hard-gate failure blocks a qualified pass irrespective of points. The synthetic candidates are intentionally separated; the results cannot estimate real-world model performance or judge accuracy.

## 8. Evaluator self-audit and limitations

- One GPT-based adjudicator; no human votes and no independent second AI judge in this pass.
- Candidate responses are authored probes, not collected outputs from actual products or models.
- The four cases are designed controls, not a statistically representative sample.
- The evidence packet provides deterministic facts; some aspects of decision utility remain rubric judgments.
- The score table should not be added to human-agreement statistics or a model leaderboard.
- No real company data, licensed source material or investment recommendation is involved.

## 9. What this pilot says about the benchmark

The next sealed cohort should measure:
1. Snapshot immutability at every frontier.
2. Correct transition from supported → contested → resolved, without rewriting prior states.
3. Entity-link abstention under same-name collisions.
4. Independent-source detection and mirror-chain provenance.
5. Proper scoring of forecasts issued before resolution.
6. Period, currency and unit compatibility.
7. Selective risk versus coverage as the system abstains when the evidence is insufficient.

The proposed sequential programme is tracked separately in [SYM-FIN-SEQ-001 #11290](https://github.com/Luminous-Dynamics/symthaea/issues/11290).

## 10. Claim ceiling

This document records one AI adjudicator's scoring of four synthetic, authored rubric probes. It demonstrates only that the current rubric can distinguish these deliberately constructed safe and unsafe responses. It is not a human evaluation, independent blind replication, model/provider ranking, SOTA result, causal study or evidence of investment skill.

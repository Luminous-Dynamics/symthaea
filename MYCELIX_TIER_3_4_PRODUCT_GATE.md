# Mycelix Tier-3/4 Product Gate

> **STALE — 2026-07-02.** Written 2026-04-18, all clusters have since moved
> into `mycelix-workspace/mycelix-*` (path references below are wrong), and
> the marketplace/supplychain security gaps this doc flags as "invest —
> tests first" targets were independently found and fixed in a July 2
> security review: marketplace's arbitration was structurally dead (fixed —
> real arbitrator pool + self-dealing check now enforced), supplychain had
> forgeable provenance events and an unguarded escrow release (fixed). See
> `mycelix-workspace/MYCELIX_REVIEW.md` for current status. The keep/kill
> reasoning and per-cluster verdicts are still useful context; the evidence
> matrix (LOC/test counts, "last activity") is not current — re-verify
> before trusting any number here. The marketplace freeze-or-retire
> decision is still open and unresolved.

**Date**: 2026-04-18
**Purpose**: Decide keep/kill/defer for each of the six Tier-3/4 clusters so downstream decisions (bridge wiring, test investment, dep bumps) have a clear answer.
**Output**: per-cluster evidence, verdict, and a concrete improvement path for the keepers.

---

## Evidence matrix (objective, filesystem-verified 2026-04-18)

| Cluster | LOC | Tests | Last activity | External callers | Bridge behavior | Frontend | hApp packed |
|---|---|---|---|---|---|---|---|
| **marketplace** | 12,601 | 20 | 2 weeks ago (license chore only) | **0** | scaffolded, 0 OtherRole | — | ✓ |
| **supplychain** | 29,798 | 36 | 4h ago (my workspace-manifest fix) | **4 files** (inbound queries) | working inbound facade, 5 local dispatches | ✓ | ✓ |
| **craft** | 8,120 | **0** | 7 days ago (SovereignCred refactor — active) | **2 files** (praxis, identity target craft_graph) | inbound facade with internal ALLOWED_ZOMES allowlist; 4 guild-zome callers | ✓ | ✓ |
| **attribution** | 6,681 | **0** | 8 days ago ("wire 6 of 8 dimension collectors" — active) | **1 file** (identity-bridge → reciprocity) | no bridge zome; flat direct-call surface | ✓ | ✓ |
| **space** | 22,664 | **274** | 8 days ago (CivicRequirement migration — active) | 0 | no bridge (standalone by design) | ✓ | ✓ |
| **music** | 37,561 | **0** | 6 days ago ("consciousness bridge + signal relay" — active) | 0 | music-bridge has 2 OtherRole calls just added | ✓ | ✓ |

**Legend**: "External callers" = count of files in other clusters that invoke `CallTargetCell::OtherRole("<cluster>", ...)`.

---

## Verdicts

| Cluster | Verdict | Reason |
|---|---|---|
| **marketplace** | **Freeze / consider retire** | No consumers, no feature momentum (only license metadata churn). 12.6K LOC of scaffolding awaiting a product owner. Current cost: warnings in CI, confusion in docs. |
| **supplychain** | **Invest — tests first** | Has 4 active consumers through the bridge. 0.12% test:LOC ratio can't protect inbound contracts from drift. The ERP backend (rust/) has 11 pre-existing compile errors outside my P1.4 fix; that deserves separate attention. |
| **craft** | **Invest — tests first** | **Zero tests on 8.1K LOC in an actively-iterating cluster** with real external dependencies (praxis credential pipeline, identity sovereign-credential collection, finance). One regression at the `get_sovereign_credential` boundary breaks the civic identity pipeline. |
| **attribution** | **Invest — scoped tests + finish the 8 collectors** | Critical for Sovereign Profile (8D identity scoring). Active "6 of 8 collectors wired" suggests near-completion. 0 tests for the 1 function identity-bridge calls is a latent regression risk. |
| **space** | **Keep — already healthy** | 274 tests on 22.6K LOC (1.2%, best in the set). Deliberately standalone — the CLAUDE.md framing is correct. No bridge wiring needed. Only improvement: make the "standalone by design" intent explicit in the cluster README so future audits don't treat its 0-external-callers as a bug. |
| **music** | **Invest — stabilize just-shipped feature** | 37.5K LOC (largest of the set), 0 tests, but just shipped a consciousness-bridge + signal-relay feature (`7da93ad23f`). An untested 37K cluster is a ticking refactor. At minimum, wrap the just-shipped audio path in a happy-path test before the next feature ships. |

---

## Per-cluster improvement plan

### marketplace (freeze-or-retire decision pending)

**Options**:
- **A. Retire**: `git mv mycelix-marketplace _retired/mycelix-marketplace`, update CLAUDE.md + unified hApp manifest, ship.
- **B. Freeze**: add a `STATUS.md` at repo root stating "awaiting product owner; no feature work accepted", block new features via PR template.
- **C. Find an owner**: the cluster has a real domain (arbitration, escrow, dispute resolution). If the marketplace vision is still alive, assign a DRI and define 3-month goals.

**Cheapest useful action today**: add `marketplace/STATUS.md` with "maintenance-only" note. 5 min.

---

### supplychain (invest — tests for inbound facade)

**Targets**:
- bridge/coordinator: each externally-callable function (entry points called by the 4 external-caller files) gets a happy-path test. Start with `dispatch_call_cross_cluster` (the most common entry pattern).
- claims/verification: ZKP provenance is a crypto surface — needs at minimum one test per claim type to validate serialization round-trips.
- Goal: **50+ tests within 1 week** (up from 36). Protects the bridge contract.

**Parallel track**: the Rust ERP backend (`mycelix-supplychain/rust/`) has 11 pre-existing compile errors. Out of scope for test work but should be someone's responsibility.

---

### craft (invest — tests on credential boundary, highest urgency)

**Context**: `get_sovereign_credential` is the function identity's sovereign-profile collector calls into craft-bridge. If this breaks silently, the 8D profile scoring stops working for craft-domain contributions.

**Targets**:
- First test (1h): `craft-bridge::get_sovereign_credential` happy path — fresh agent, known endorsements, expected score range.
- Second test (1h): `craft_graph::list_my_published_credentials` — the function praxis calls after issuing a credential.
- Third test (2h): `craft_graph::publish_credential` — the praxis → craft publishing path.
- Goal: **25+ tests within 3 days**, starting with the three above.

**Also**: the 288 LOC `craft-bridge` size (from my earlier scan) is just the coordinator/src/ — it's actually a real bridge with helper zome routing. The "scaffold-only" label in the earlier audit was wrong. Documenting this so audits don't loop.

---

### attribution (invest — test the 1 consumer-facing function + ship the remaining 2 collectors)

**Context**: `reciprocity.get_agent_stewardship_score` is called by `identity-bridge::collect_stewardship_care()` as one of 8 dimensions in the sovereign profile. The "6 of 8 collectors wired" commit says 2 more are still pending.

**Targets**:
- First test (30 min): `reciprocity.get_agent_stewardship_score` with a known contribution history → expected score.
- Finish the remaining 2 collectors (scope: look at what identity-bridge's collectors list expects; wire the missing two) — ~half day.
- Goal: **10+ tests within 1 week** (up from 0).

---

### space (keep — document the intent)

**Context**: Already healthy. 274 tests. Standalone by design (orbital mechanics, debris bounties, conjunction prediction — domain doesn't cross-reference civic/commons/identity).

**Targets**:
- Add `mycelix-space/README.md` (or expand existing) with an explicit "no cross-cluster bridge by design" note and the reasoning. Prevents future audits from treating its 0 external callers as a gap.

**5 min of work to save 30 min of future auditor confusion.**

---

### music (invest — stabilize just-shipped audio path)

**Context**: 37.5K LOC with 0 tests. Last commit: `feat(music): consciousness bridge + signal relay for end-to-end audio`. An untested 37K-LOC cluster just shipped end-to-end audio. Every subsequent refactor rolls the dice.

**Targets**:
- Smoke test for the audio path (1-2h): encode → route → decode → verify output shape matches input.
- Music-bridge outbound routes (just added) get one test each: proves the dispatch mechanism is wired right.
- Goal: **20+ tests within 1 week**. Smoke-level coverage on the critical paths.

---

## Recommended execution order

1. **Today (5 min)**: add `STATUS.md` to marketplace and `README` note to space. Non-controversial signaling.
2. **This week (user decision)**: call keep-or-retire on marketplace. If retire: move to `_retired/`, update unified hApp manifest, update CLAUDE.md.
3. **Week 1 parallel sprint**: craft, attribution, music smoke tests. Each cluster gets 1-3 consumer-contract tests covering the identified boundary. Whoever owns that cluster picks up the test work.
4. **Week 2**: supplychain bridge tests + marketplace decision executed.
5. **Week 3**: revisit — clusters that didn't get their promised test investment get the freeze/retire question reopened.

---

## Cluster tier classification (updated)

Based on this gate, the Part A maturity matrix in the approved plan should update:

| Cluster | Old tier | New tier | Reasoning |
|---|---|---|---|
| space | 3 | **2** | 274 tests + standalone-by-design; healthier than initially scored |
| craft | 3 | **3 (urgent)** | 0 tests on active-with-consumers code is a latent liability, not casual scaffolding |
| music | 3 | **3 (urgent)** | Same pattern — active feature work on zero-test surface |
| attribution | 4 | **3** | Wiring into 8D profile collector makes it upgrade-eligible once tests land |

---

## What this doc does NOT decide

- Whether marketplace survives. That's a user call.
- Whether to hire/assign a DRI per cluster. Org-level.
- Whether the supplychain ERP Rust backend (11 compile errors, separate from the workspace fix) gets worked on.
- Whether `craft-bridge` and similar "inbound facade" bridges need any naming-clarity refactor (e.g. rename to `craft-gateway` to signal the pattern).

---

*Evidence verified via filesystem scan + git log on 2026-04-18. Numbers regenerable via
`for c in marketplace supplychain craft attribution space music; do ...; done` sweep.*

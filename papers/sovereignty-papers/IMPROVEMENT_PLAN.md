# The Sovereignty Papers: Improvement Plan (v2)

*Updated with deep codebase audit findings, 2026-03-23*

---

## Summary of Findings

Two deep-dive audits revealed 8 discrepancies between the essays and the codebase, plus 5 structural weaknesses from the editorial critique. The key discovery: **the code is more nuanced than the essays describe**, and the right fix for most discrepancies is to update the essays to match the code's actual (and often better) design.

---

## Priority 1: Fix Factual Errors (CRITICAL — before any publication)

### 1.1 Reputation is linear, NOT squared — UPDATE ESSAYS
- **Claim** (Essays 7, 11): "reputation enters as its square"
- **Reality**: `consciousness_profile.rs:120` — `r * 0.25` (linear)
- **Key finding**: Reputation-squared IS used in the codebase — in the FL Byzantine consensus (`governance.rs:149,156`), where `reputation.powi(2)` weights Byzantine detection votes. This is a different context with a different threat model.
- **Recommendation**: **Update essays.** The linear consciousness profile is deliberately simple (295+ tests, Dunbar/Ostrom/Axelrod citations in code). Reputation-squared belongs in Byzantine consensus, not governance gating. The anti-plutocratic argument in Essays 7 and 11 needs to be rewritten around the actual defenses: the 347-day half-life (consistency over time), the community attestation weight at 30% (hardest to buy), and the credential expiry (prevents accumulation).
- **Files to change**:
  - Essay 7: Section III (remove squared from reputation description), Section IV (rewrite weight defense), Section VI (fix composite formula)
  - Essay 11: Section III (replace "reputation-squared" argument with actual anti-plutocratic mechanisms)
  - Essay 7 endnotes 2, 6 (remove squared references)

### 1.2 STAKE_MAX_BONUS doesn't exist — UPDATE ESSAYS
- **Claim** (Essays 7, 11): "STAKE_MAX_BONUS at 5%"
- **Reality**: No such constant exists. Stake concepts exist in FL governance (`fl-aggregator`) but not in consciousness profiles.
- **Recommendation**: **Update essays.** The consciousness profile deliberately excludes financial stake from the composite formula — which is actually a *stronger* anti-plutocratic position than a 5% cap. The essays should describe what the code actually does: financial stake is **not a dimension** of the consciousness credential at all. The four dimensions (identity, reputation, community, engagement) contain no financial term. This is stronger than capping it at 5%.
- **Files to change**:
  - Essay 7: Section V (rewrite "What Financial Stake Does Not Measure" — it's even simpler than claimed: stake is absent, not capped)
  - Essay 11: Section III (rewrite "The Two Constraints" — first constraint is total exclusion of capital from the formula, not a 5% cap)
  - Essay 7 endnote 5, Essay 11 endnote 1 (remove STAKE_MAX_BONUS references)

### 1.3 Reputation half-life is ~347 days, not 30 — UPDATE ESSAYS
- **Claim** (Essays 8, 10): "30-day half-life"
- **Reality**: `REPUTATION_DECAY_PER_DAY = 0.998`, half-life ~347 days (verified by test at line 3085)
- **Additional context**: Slashing exists separately — 0.5× on first offense, blacklist at 0.05, 100 good interactions to restore. This is the fast consequence mechanism; decay is the slow one.
- **Recommendation**: **Update essays.** The 347-day half-life with separate slashing is a better design than a 30-day half-life alone. Slow decay respects the value of long-term engagement. Fast slashing punishes bad behavior immediately. The essays should describe both mechanisms.
- **Files to change**:
  - Essay 8: Section V (replace "30-day half-life" with actual decay + slashing dual mechanism)
  - Essay 10: endnote (correct half-life figure)

### 1.4 Bootstrap TTL is 15 minutes, not 1 hour — UPDATE ESSAYS
- **Claim** (Essay 8): "one-hour TTL"
- **Reality**: `BOOTSTRAP_TTL_US = 900_000_000` (15 minutes, reduced from 1 hour to minimize TOCTOU window)
- **Additional context**: Governance credential TTL is separately defined as `DEFAULT_TTL_US = 86_400_000_000` (24 hours). The essays conflate bootstrap TTL and governance TTL.
- **Recommendation**: **Update Essay 8** to correctly distinguish: bootstrap credentials expire in 15 minutes, full governance credentials expire in 24 hours. The 15-minute bootstrap TTL is a security hardening decision (TOCTOU mitigation) and should be described as such.
- **Files to change**: Essay 8 Section IV

### 1.5 Moral obligations are 8 perfect + 8 imperfect (CONFIRMED CORRECT)
- **Claim** (Essay 6): "Eight perfect, eight imperfect"
- **Deep audit found**: 8 perfect + 8 imperfect = 16 total. The first audit's "7+9" was incorrect — `ahimsa_nonviolence`, `prevent_suffering`, and `minimize_collateral` are all `is_perfect_duty: true`.
- **Perfect (8)**: honesty, non_theft, non_harm, promise_keeping, respect_autonomy, ahimsa_nonviolence, prevent_suffering, minimize_collateral
- **Imperfect (8)**: beneficence, self_improvement, epistemic_humility, error_acknowledgment, deference_to_expertise, selfless_service, welfare_priority, radical_translucency
- **Status**: **Essay is CORRECT. No change needed.**

### 1.6 Commons zome count — VERIFY
- **Claim** (Essay 16): "39 zomes"
- **CLAUDE.md**: "39 (38 domain + 1 bridge)"
- **First audit**: Found 45 directories — likely counting coordinator+integrity subdirectories
- **Recommendation**: The canonical count is 39 (each zome = coordinator + integrity pair counted as one). Verify by counting top-level zome directories excluding coordinator/integrity subdirectories. Essay likely correct.
- **Action**: Verify count, then no essay change if 39 is confirmed

### 1.7 Oversight bodies — UPDATE ESSAYS
- **Claim** (Essay 12): "Constitutional Council, Ethics Review, Emergency Committee"
- **Reality**: mycelix-governance has `constitution`, `councils`, `proposals`, `voting`, `execution`, `threshold-signing`, `jurisdiction` zomes — the infrastructure for these roles, but not the named bodies
- **Recommendation**: **Update Essay 12** to describe the oversight function in terms of the actual zome architecture. The three-body concept is sound (different legitimacy bases for different oversight functions); the essay should explain that `constitution` + `councils` + `voting` provide the infrastructure for these roles, and that the specific body names are governance decisions made by the first communities that deploy the system.
- **Files to change**: Essay 12 Section II

### 1.8 Subsystem count is 17, not 12 — UPDATE ESSAYS
- **Claim** (Essay 12): "twelve co-prime cycle managers"
- **Reality**: 17 registered subsystem managers in CognitiveLoopService. Co-prime scheduling IS implemented (confirmed in subsystem_trait.rs). Moral algebra specifically runs on intervals of 7, 19, 23, and 97 cycles (all pairwise coprime).
- **Recommendation**: **Update Essay 12** to say "seventeen subsystem managers operating on co-prime intervals" and cite the specific moral algebra intervals (7/19/23/97) as the concrete example. Remove the specific list of 12 primes from endnote 1.
- **Files to change**: Essay 12 Section II, endnote 1

---

## Priority 2: Address Critique Weaknesses (HIGH)

### 2.1 Scandinavian counterexample in Essay 1
- **Location**: Essay 1, Section IV (The Externality Engine) — add after the bridge metaphor paragraph
- **Content**: ~200 words acknowledging Nordic co-determination as the strongest counterexample to the universal thesis, then arguing that external feedback is fragile (regulatory capture erodes it over time) while internal feedback (consciousness coupling) is structural
- **Why it matters**: Without this, a political economist will dismiss the thesis as ignoring the most obvious counterevidence

### 2.2 Operational vs. phenomenal consciousness in Essay 19
- **Location**: Essay 19, new paragraph in Section IV
- **Content**: ~200 words explicitly stating that the operational definition (presence, accountability, embeddedness, engagement) and the phenomenal definition (subjective experience) may be unrelated. A philosophical zombie could score perfectly. The governance system works regardless. The phenomenal question matters for moral consideration (should we grant rights?) but not for governance fitness (should this participant have power?).
- **Why it matters**: Without this, a philosopher of mind will identify a category error at the heart of the series

### 2.3 Adversarial red-team scenarios in Essay 12
- **Location**: Essay 12, new Section between current IV and V
- **Content**: ~500 words walking through three specific attacks against consciousness coupling:
  1. Coordinated collusion (20 participants conspire to modify thresholds)
  2. Slow infiltration (nation-state 18-month operation)
  3. Measurement gaming (sophisticated bot mimics engagement)
- For each: the attack vector, the defense mechanism, and what remains vulnerable
- **Why it matters**: Every failure case in the series is a failure of the OLD system. The series needs at least one worked example of how the NEW system responds to attack.

### 2.4 Three-currency expansion in Essay 16
- **Location**: Essay 16, Section V — expand by ~300 words
- **Content**: Address exchange rate determination, TEND non-transferability enforcement (integrity zome level), demurrage rate as governance parameter. Acknowledge this is a design sketch.
- **Why it matters**: The three-currency economy is a radical claim introduced in one section and never revisited. It needs either more defense or explicit acknowledgment of incompleteness.

### 2.5 Fix opening hooks for Essays 4, 6, 16
- **Essay 4**: Restructure Section I to open with "Imagine two governance systems..." (the committee thought experiment, currently Section II) instead of "In Essay No. 3..."
- **Essay 6**: Open with "The VOC was aware. The VOC was responsive. The VOC committed the Banda massacre." (currently buried in Section I paragraph 3)
- **Essay 16**: Open with "In 1999, the government of Bolivia privatized the water system of Cochabamba" (currently Section II)
- **Why it matters**: These are section-opening essays (II, II, VI) that start with backward references instead of hooks. Readers who enter the series at a section boundary will bounce.

---

## Priority 3: Editorial Consistency (MEDIUM)

### 3.1 Reduce credential re-explanation in later essays
- Essays 10-15 re-explain the four-dimensional credential from scratch. After Essay 7, use "the consciousness credential (Essay No. 7)" or a one-sentence summary.

### 3.2 Distinguish 24-hour governance TTL from 15-minute bootstrap TTL
- Current essays conflate them. After the fix in 1.4, ensure every mention of "credential expiry" specifies which type.

### 3.3 Evidence-density pass on Essays 13-21
- Read each for unsupported claims. Key targets: Essay 20 (philosophical claims — now has endnotes post-Pass 1, verify they're sufficient), Essay 17 (FEMA details), Essay 19 (substrate framework).

### 3.4 Standardize numerical claims
- Subsystem count: 17 (not 12)
- Moral obligation split: 8+8 (confirmed correct)
- Moral evaluation interval: every 7 cycles (not every cycle)
- Commons zomes: verify 39
- Test count: ~21,500
- Code size: 2.7M lines
- Psych-bench: 136 benchmarks across 26 domains

---

## Priority 4: Code vs. Essay Decisions (FINAL)

| Feature | Essay Claims | Code Reality | Decision |
|---------|-------------|-------------|----------|
| Reputation-squared | `r² * 0.25` | `r * 0.25` (linear) | **Update essays** — linear is deliberate; squared lives in FL consensus |
| STAKE_MAX_BONUS 5% | 5% cap constant | Not present | **Update essays** — total exclusion of capital is stronger than 5% cap |
| 30-day half-life | 30 days | 347 days + slashing | **Update essays** — dual mechanism is better design |
| Bootstrap TTL | 1 hour | 15 minutes | **Update essays** — 15 min is TOCTOU hardening |
| 8+8 obligations | 8 perfect + 8 imperfect | 8+8 confirmed | **No change** — essay is correct |
| 3 oversight bodies | Named bodies | Governance zomes | **Update essays** — describe as roles within existing zomes |
| 12 co-prime managers | 12 with specific primes | 17 managers, co-prime scheduling | **Update essays** — 17 managers, cite actual intervals (7/19/23/97) |
| 24h credential TTL | 24 hours universal | 24h governance + 15min bootstrap | **Clarify in essays** — both exist, distinguish them |

**Key insight**: In every case, the code is right and the essays should match it. The code was designed with care (295+ tests, Ostrom/Dunbar/Axelrod citations). The essays were written from architectural intent, not from code reading. The fix is always the same: make the essays describe what was actually built.

---

## Execution Order

### Phase A: Essay corrections (Priority 1) — No code changes needed
1. Fix Essay 7: Remove reputation-squared, remove STAKE_MAX_BONUS, describe actual linear formula with total capital exclusion
2. Fix Essay 11: Replace two-constraint argument with actual anti-plutocratic mechanisms (capital exclusion + 347-day half-life + slashing + community at 30%)
3. Fix Essay 8: Correct bootstrap TTL to 15 minutes, distinguish from 24-hour governance TTL, add slashing mechanism to permeability section
4. Fix Essay 12: Update to 17 subsystem managers with actual co-prime intervals, describe oversight bodies as roles within governance zomes
5. Fix Essay 6: Verify 8+8 split (confirmed correct, no change needed)

### Phase B: Structural improvements (Priority 2)
6. Essay 1: Add Scandinavian counterexample
7. Essay 19: Add operational vs. phenomenal consciousness paragraph
8. Essay 12: Add adversarial red-team scenarios section
9. Essay 16: Expand three-currency section
10. Essays 4, 6, 16: Fix opening hooks

### Phase C: Editorial pass (Priority 3)
11. Reduce re-explanations in Essays 10-15
12. Distinguish TTL types throughout
13. Evidence-density pass on Essays 13-21
14. Standardize numerical claims

### Phase D: Verification
15. Full sequential read-through of all 21 essays
16. Verify all numerical claims against codebase
17. Check narrative coherence after all edits

---

## What Changes in the Argument

The core argument is unchanged. The details are more honest and more defensible:

- **Anti-plutocratic defense** shifts from "reputation-squared + 5% stake cap" to "total capital exclusion from the credential formula + 347-day half-life + immediate slashing + community attestation at 30%." This is actually a *stronger* argument — the code does more than the essays claimed.

- **Temporal dynamics** shift from "30-day half-life ensures rapid turnover" to "347-day half-life respects long-term engagement; slashing (0.5× on first offense, blacklist at 0.05) provides immediate consequence for betrayal." This is more nuanced and more defensible.

- **Bootstrap security** shifts from "1-hour TTL" to "15-minute TTL, hardened against TOCTOU attacks." More secure, more honest.

- **Oversight architecture** shifts from "three named bodies" to "three oversight functions enacted through governance zomes — constitution interpretation, ethics review, emergency response — with infrastructure for communities to formalize these as named bodies." More accurate, equally functional.

- **Subsystem architecture** shifts from "12 co-prime managers" to "17 subsystem managers on co-prime intervals, with moral algebra specifically at cycles 7/19/23/97." More specific, more verifiable.

None of these changes weakens the argument. Several strengthen it. The code is better than what the essays described.

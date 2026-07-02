# Constitutional Alignment: Code vs Constitution

## Purpose

This document identifies misalignments between the Mycelix Spore Constitution
(v0.25) and the anti-tyranny governance code, and proposes resolutions.

The Constitution is the supreme authority. The code implements the Constitution.
Where they conflict, the code must change.

---

## Misalignment Analysis

### 1. Override Threshold: 67% vs 80%

**Constitution (Art. III, Sec. 5.3)**:
> "The Global DAO may overturn any veto by a two-thirds (2/3) majority in both houses."

**Code**: `VETO_OVERRIDE_THRESHOLD = 0.80` (80%)

**Resolution**: Align code to 67% (constitutional authority). The 80% threshold
was chosen during the anti-tyranny hardening for extra security margin, but it
contradicts the Constitution and creates the 21% boycott vulnerability. The
Constitution's 2/3 threshold (67%) actually provides BETTER protection against
voter suppression while still requiring supermajority legitimacy.

**Action**: Change `VETO_OVERRIDE_THRESHOLD` from `0.80` to `0.67`.
Update `OVERRIDE_THRESHOLD_FLOOR` from `0.67` to `0.60` (participation
insurance floor adjusts accordingly).

### 2. Immutable Core: Total Block vs 90% Supermajority

**Constitution (Art. IV, Sec. 2)**:
> "The following are immutable except by a 90% supermajority in both
> Global DAO houses, ratified by three-fourths (3/4) of all Sector DAOs
> and three-fourths (3/4) of all Regional DAOs"

**Code**: `UNAMENDABLE_RIGHTS` — DHT validation rejects ALL amendments
targeting these rights, with no override mechanism.

**Resolution**: The code is MORE restrictive than the Constitution. The
Constitution allows a 90% + 3/4 process for even the Immutable Core.
The code should implement this graduated protection:

- Normal amendments: 67% + simple majority ratification (Art. IV, Sec. 1)
- Immutable Core amendments: 90% + 3/4 ratification (Art. IV, Sec. 2)
- Anti-tyranny invariants: Propose adding these to the Immutable Core
  via constitutional amendment (see Proposed Amendments below)

**Action**: Modify `targets_unamendable_core()` to check for the 90%
supermajority approval rather than blocking entirely. Add a
`requires_immutable_core_process()` check that returns true (requiring
the enhanced process) rather than false (rejecting outright).

### 3. Veto Rate Limit: 7-day Cooldown vs 3-per-12-months

**Constitution (Art. III, Sec. 5.4)**:
> "Foundation exercises Golden Veto more than 3 times in a 12-month period"
> triggers probation

**Code**: `VETO_COOLDOWN_US = 7 * 24 * 3600 * 1_000_000` (7-day cooldown,
~52 possible vetoes per year)

**Resolution**: The code's per-veto cooldown is STRICTER per-event but allows
MORE total vetoes per year (52 vs 3). The constitutional 3-per-year limit with
probation is the intended design. The code should implement BOTH:
- 7-day minimum cooldown between consecutive vetoes (code, keep)
- 3-per-12-month limit with probation trigger (add from Constitution)

**Action**: Add `VETO_YEARLY_LIMIT = 3` and probation tracking to
`WorldGovernance` struct.

### 4. Veto Sunset: Missing in Code

**Constitution (Art. III, Sec. 3)**:
> "After 36 months from Genesis Epoch, Strategic Override expires
> automatically, transitioning to Charter Guardian Authority"

**Code**: No sunset mechanism. Vetoes work identically at all times.

**Resolution**: The code should track genesis epoch and distinguish
Strategic Override (first 36 months) from Charter Guardian Authority
(after 36 months). Charter Guardian Authority is limited to blocking
actions that "demonstrably violate this Constitution or Core Principles."

**Action**: Add `STRATEGIC_OVERRIDE_SUNSET_TICKS` and genesis epoch
tracking. After sunset, vetoes require constitutional justification.

### 5. Foundation Entity: Not Modeled

**Constitution (Art. II, Sec. 2)**:
> "The Foundation shall act as the legal and operational custodian"
> "The Foundation shall have no legislative power"

**Code**: No Foundation entity in the governance zomes.

**Resolution**: The Foundation is an off-chain entity (legal personality).
The code doesn't need to model it directly, but should recognize the
Foundation's Golden Veto authority as distinct from Guardian vetoes.

**Action**: No code change needed. The Foundation exercises vetoes through
the same mechanism as Guardians, but with Foundation-specific DID identity.

### 6. Veto Registry: Partially Implemented

**Constitution (Art. III, Sec. 5.5)**:
> "All veto exercises... recorded in an immutable public ledger"
> "Fields: veto_id, timestamp, justification_hash, threat_category..."

**Code**: `GuardianVeto` has id, timelock_id, guardian, reason, vetoed_at.
Missing: justification_hash, threat_category, affected_proposal_id,
override tracking fields.

**Resolution**: Extend `GuardianVeto` struct to match constitutional
registry requirements.

**Action**: Add missing fields to `GuardianVeto` entry type.

---

## Aligned (No Changes Needed)

| Feature | Constitution | Code | Status |
|---------|-------------|------|--------|
| Fork rights | Art. IX, Sec. 5 | Unamendable right #6 | Aligned |
| Emergency pause | 72h multisig / 30d DAO | 14-day sessions | Compatible |
| Director term limits | 3-year terms | 365-day membership | Compatible |
| Consciousness gating | Implicit in MYCEL/MATL | Explicit in voting | Code extends |
| Quadratic voting | Art. VIII, Sec. 5 | QuadraticVote entry | Aligned |
| Three-currency separation | SAP/MYCEL/TEND | Stake cap in voting | Aligned |

---

## Proposed Constitutional Amendments

The following amendments would incorporate the anti-tyranny hardening
into the Constitution, giving them constitutional authority:

### Amendment 1: Anti-Tyranny Invariants (Add to Immutable Core)

**Amend Article IV, Section 2** to add:

> * Permission-less enforcement of time-limited powers
> * Participation insurance for veto override processes
> * Consciousness threshold absolute floors (basic <= 0.4,
>   voting <= 0.6, constitutional <= 0.8)
> * AI agent governance ceiling (Steward tier maximum)
> * Self-reported consciousness score cap for Guardian-tier actions
> * Config change authorization verification

**Rationale**: These protections close the Philosopher-King Trap,
config authorization bypass, consciousness score gaming, and AI
sovereignty risks identified through adversarial red-teaming.

### Amendment 2: Participation Insurance (New Article III, Section 6)

**Add to Article III**:

> **Section 6. Participation Insurance**
>
> If a veto override vote fails to reach quorum, the override threshold
> shall decrease by 5 percentage points for each subsequent attempt,
> to a floor of 60%. This ensures that sustained voter suppression
> cannot permanently block collective action while preserving
> supermajority legitimacy.
>
> Override threshold schedule:
> - Attempt 1: 67% (standard constitutional threshold)
> - Attempt 2: 62%
> - Attempt 3+: 60% (floor)

### Amendment 3: Ethics-Governance Binding (New Article I, Section 2.x)

**Add to Article I, Section 2 (Core Principles)**:

> **Ethical Accountability**: All governance proposals affecting Core
> Principles, Member Rights, or resource allocation exceeding 100,000 SAP
> shall be subject to ethics assessment. Proposals flagged as ethically
> Blocked require mandatory disclosure to all voters and escalated
> approval thresholds (one tier higher than standard requirements).
> The ethics assessment system shall fail to Caution (not silent
> approval) when assessment infrastructure is unavailable.

### Amendment 4: Council Membership Term Limits (Governance Charter)

**Add to Governance Charter, Section on Council Operations**:

> Council members shall serve terms of no more than 365 days.
> Membership expires automatically and may be renewed through
> re-election. Expiration enforcement is permission-less — any
> network participant may trigger the expiration of overdue terms.

---

## Implementation Priority

1. **Align override threshold to 67%** (constitutional requirement)
2. **Add 3-per-year veto limit** (constitutional requirement)
3. **Soften unamendable core to 90% process** (constitutional requirement)
4. **Draft and propose Amendments 1-4** (constitutional evolution)
5. **Add veto registry fields** (constitutional requirement)
6. **Add veto sunset tracking** (constitutional requirement)

---

*The Constitution is the supreme governing instrument. The code serves it.*

# Beyond Algorithmic Trust (Draft)

**Context (Oct 2025)**  
PoGQ + reputation + robust aggregation covers the algorithmic surface, but
label-skew stress-tests show that pure gradient scoring remains brittle. To
ship a production Mycelix deployment we need layered trust—mathematical,
economic, hardware and social.

---

## Phase 1 Additions (Q4 2025 – Q1 2026)

| Layer | Work Item | Notes |
|-------|-----------|-------|
| Robust aggregation | Integrate trimmed-mean / KRUM fallback for skewed distributions | `ROBUST_AGGREGATOR=trimmed_mean|krum` now available; tune via sweeps + matrix summary |
| Behavioural analytics | Augment reputation with temporal features (suspicious oscillations, rapid recoveries) | Low effort; helps catch adaptive attackers |
| Label-skew tuning | Automated threshold sweeps (see `scripts/sweep_label_skew.py`) | Drive red cells → green before promoting results |

---

## Phase 2 Enhancements (Mid 2026)

1. **Economic Incentives**
   - PoGQ credits → staking/slashing (integrate with Polygon backend)
   - Reputation-weighted payouts (malicious updates burn stake)

2. **Hardware Attestation**
   - Collect SGX/TEE quotes in the edge proof payload
   - Optional policy: only accept gradients with verified attestation

3. **Committee Diversity**
   - Mix algorithmic, hardware and human verifiers
   - Require stake or attestation to join committee

---

## Phase 3 (2026+)

- **Social Verification**: Verified champions sign off on committee decisions; reputation anchored in real-world identity.
- **Governance**: DAO-style votes to demote or slash misbehaving nodes/conductors.
- **Audit Logs**: Public registry of matrix sweeps + attestation records so researchers can reproduce trust claims.

---

## Immediate Actions

- Keep `results/bft-matrix/latest_summary.md` up to date—hat tip to reviewers.
- Log label-skew red cells in `docs/cleanup-plan.md` (done).
- Track trust-layer progress in `30_BFT_VALIDATION_RESULTS.md` (IID green, label-skew WIP).

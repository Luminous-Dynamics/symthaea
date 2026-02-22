# Mycelix DNA Size Audit

**Date**: 2026-02-22
**Build**: `cargo build --release --target wasm32-unknown-unknown`

## Summary

| Cluster | WASM Files | Raw WASM Total | DNA Bundle (compressed) | Zome Pairs |
|---------|-----------|----------------|------------------------|------------|
| Commons | 78 | 144.89 MB | 27 MB | 39 (integrity + coordinator) |
| Civic | 32 | 63.71 MB | 12 MB | 16 (integrity + coordinator) |
| **Total** | **110** | **208.60 MB** | **39 MB** | **55** |

## Per-Domain Breakdown — Commons

| Domain | Zome Pairs | Raw WASM | Notes |
|--------|-----------|----------|-------|
| Mutualaid | 7 | 26.81 MB | Largest domain (7 sub-zomes) |
| Housing | 6 | 22.17 MB | CLT + governance + finances |
| Water | 5 | 18.25 MB | 5 sub-zomes |
| Care | 5 | 17.95 MB | Timebank + circles + matching + plans + credentials |
| Property | 4 | 15.35 MB | Registry + transfer + disputes + commons |
| Food | 4 | 14.49 MB | Production + distribution + preservation + knowledge |
| Support | 3 | 11.25 MB | Knowledge + tickets + diagnostics |
| Transport | 3 | 10.86 MB | Routes + sharing + impact |
| Commons Bridge | 1 | 3.89 MB | Cross-domain dispatch |
| Space | 1 | 3.85 MB | Single zome |

## Per-Domain Breakdown — Civic

| Domain | Zome Pairs | Raw WASM | Notes |
|--------|-----------|----------|-------|
| Justice | 5 | 22.76 MB | Largest civic domain; arbitration coordinator is 3.01 MB |
| Emergency | 6 | 21.74 MB | 6 sub-zomes |
| Media | 4 | 15.16 MB | Publication + attribution + factcheck + curation |
| Civic Bridge | 1 | 4.04 MB | Cross-domain + cross-cluster dispatch |

## Zome Size Statistics

| Type | Cluster | Count | Avg | Min | Max |
|------|---------|-------|-----|-----|-----|
| Integrity | Commons | 39 | 1.40 MB | 1.34 MB | 1.51 MB |
| Coordinator | Commons | 39 | 2.31 MB | 2.16 MB | 2.57 MB |
| Integrity | Civic | 16 | 1.50 MB | 1.32 MB | 1.90 MB |
| Coordinator | Civic | 16 | 2.48 MB | 2.16 MB | 3.01 MB |

Coordinator-to-integrity ratio: ~1.65x across both clusters.

## Headroom Analysis

Holochain DNA bundles use zlib compression on WASM. Compression ratio: ~5.4x for commons, ~5.3x for civic.

| Metric | Commons | Civic |
|--------|---------|-------|
| Current DNA size | 27 MB | 12 MB |
| Practical limit per DNA | ~64 MB | ~64 MB |
| Remaining headroom | ~37 MB (~137%) | ~52 MB (~433%) |
| Equivalent new zome pairs | ~10 | ~14 |

**Civic has ample headroom** — could absorb 14+ new zome pairs before approaching limits.

**Commons is larger but still has room** — ~10 new zome pairs possible. However, adding a full new domain (5+ zome pairs) would consume half the remaining headroom.

## Composting / Food-Forest Feasibility

| Feature | Approach | DNA Impact |
|---------|----------|------------|
| **Composting** | Fold into `food_preservation` as batch method | Zero (no new zome) |
| **Food forest** | Fold into `food_production` as `plot_type` variant | Zero (no new zome) |
| **New food zome** | Would add ~3.6 MB raw / ~0.67 MB compressed | Feasible but unnecessary |

**Recommendation**: Both features should be implemented as extensions of existing zomes, not new zomes. The food domain's 4 zomes (14.49 MB raw, ~2.7 MB compressed) are well within bounds.

## Largest Individual WASMs

| Zome | Size | Notes |
|------|------|-------|
| `justice_arbitration` (coordinator) | 3.01 MB | Complex multi-party arbitration logic |
| `justice_evidence` (coordinator) | 2.88 MB | Chain of custody + tamper detection |
| `justice_restorative` (coordinator) | 2.76 MB | Multi-phase restorative process |
| `commons_bridge` (coordinator) | 2.57 MB | 38-zome dispatch router |
| `civic_bridge` (coordinator) | 2.48 MB | 15-zome dispatch + cross-cluster |

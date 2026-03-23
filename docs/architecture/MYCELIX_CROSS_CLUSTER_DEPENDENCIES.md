# Mycelix Cross-Cluster Dependency Matrix

Last updated: 2026-03-23

## Overview

Mycelix uses `CallTargetCell::OtherRole` for cross-cluster (inter-DNA) calls. Each cluster runs as a separate DNA within the unified hApp. This document maps every cross-cluster dependency, its fail mode, and operational implications.

## Design Rule

> **Use `call_local` (fail-closed) for security decisions. Use `call_local_best_effort` only for audit logging, telemetry, and optional enrichment.**

This rule was established after the March 2026 security audit that found 8 fail-open vulnerabilities in governance and identity clusters.

## Dependency Matrix

### Governance Cluster

| Target | Zome | Function | Purpose | Fail Mode |
|--------|------|----------|---------|-----------|
| **Identity** | did_registry | verify_did | Voter DID validation | Fail-Closed |
| **Identity** | identity_bridge | get_voter_matl_score | MATL trust for vote weighting | Fail-Closed |
| **Identity** | verifiable_credential | verify_governance_credential | Expertise claims | Fail-Closed |
| **Personal** | personal_bridge | present_phi_credential | Phi score (Phase A) | Best-Effort (PhiProvenance tracked) |
| **Personal** | personal_bridge | present_k_vector | K-vector trust | Best-Effort (default 0.5) |

### Commons Cluster

| Target | Zome | Function | Purpose | Fail Mode |
|--------|------|----------|---------|-----------|
| **Identity** | identity_bridge | get_consciousness_credential | Consciousness tier gating | Fail-Closed |
| **Identity** | identity_bridge | refresh_consciousness_credential | Credential refresh | Best-Effort (non-blocking) |
| **Identity** | did_registry | verify_agent_did | DID active check | Fail-Closed |
| **Civic** | civic_bridge | dispatch_civic_call | Cross-cluster dispatch | Fail-Closed |

### Civic Cluster

| Target | Zome | Function | Purpose | Fail Mode |
|--------|------|----------|---------|-----------|
| **Commons** | commons_bridge | dispatch_commons_call | Cross-cluster dispatch | Fail-Closed |
| **Commons** | housing_units | check_housing_capacity | Emergency shelter | Best-Effort (emergency degradation) |
| **Commons** | water_purity | query_water_safety | Emergency water check | Fail-Closed |
| **Commons** | food_distribution | query_food_stocks | Emergency food check | Fail-Closed |
| **Identity** | identity_bridge | get_consciousness_credential | Tier gating | Fail-Closed |

### Finance Cluster

| Target | Zome | Function | Purpose | Fail Mode |
|--------|------|----------|---------|-----------|
| **Governance** | governance_bridge | get_dao_member_count | Community size check | Fail-Closed (STRICT_GOVERNANCE_MODE) |
| **Governance** | governance_bridge | get_proposal_status | Proposal verification | Fail-Closed (STRICT_GOVERNANCE_MODE) |

### Hearth Cluster

| Target | Zome | Function | Purpose | Fail Mode |
|--------|------|----------|---------|-----------|
| **Identity** | identity_bridge | dispatch_identity_call | DID operations | Fail-Closed |
| **Personal** | personal_bridge | dispatch_personal_call | Vault operations | Fail-Closed |
| **Commons** | commons_bridge | dispatch_commons_call | Resource operations | Fail-Closed |
| **Civic** | civic_bridge | dispatch_civic_call | Civic operations | Fail-Closed |

### Space Cluster

| Target | Zome | Function | Purpose | Fail Mode |
|--------|------|----------|---------|-----------|
| **Identity** | identity_bridge | verify_operator | Operator credential check | Fail-Closed |

## Critical Dependency Paths

```
Governance ──→ Identity (voter verification)     [CRITICAL, fail-closed]
Commons    ──→ Identity (consciousness gating)   [CRITICAL, fail-closed]
Civic      ──→ Commons  (emergency resources)    [CRITICAL, fail-closed]
Finance    ──→ Governance (proposal verification) [CRITICAL, fail-closed]
```

## Identity Cluster: Central Hub

Identity is the most-connected cluster — 6+ clusters depend on it. If identity goes down:
- All consciousness-gated operations across commons, civic, hearth block
- Governance voting blocks (voter DID verification)
- Space operations block (operator verification)

**Mitigation**: Offline credential system (`offline_credential.rs`) provides degraded-but-functional access for up to 24h with progressive tier reduction.

## Acceptable Best-Effort Patterns

These are intentionally fail-open and documented:

1. **Governance → Personal (Phi/K-vector)**: Returns `PhiProvenance::Unavailable` honestly; voting still works with standard weights
2. **Civic → Commons (emergency housing)**: Emergency response must degrade gracefully; advisory-only failure
3. **Credential refresh**: Background operation; current credential remains valid

## Operational Notes

- **Circuit breaker**: `call_role()` includes circuit-breaker error messages for diagnosis
- **Rate limiting**: 100 calls/agent/60s on all bridge dispatch (prevents DoS via cross-cluster amplification)
- **Metrics**: `get_bridge_metrics()` extern on all bridges exposes `total_cross_cluster`, `gate_pass`, `gate_fail`, `tier_counts`
- **Audit trail**: All governance gate decisions logged via `log_governance_gate()` (sampled 10% for approvals, 100% for rejections)

# TEND Mutual Credit: Consciousness-Gated Community Currencies Without Debt

## Target Venue
Journal of Institutional Economics, IEEE Blockchain, or DeFi Workshop

## Status
OUTLINE (April 2026)

## Abstract (Draft)

We present TEND, a mutual credit system where one TEND equals one hour of labor, integrated with consciousness-gated governance to prevent the accumulation pathologies that plague both fiat and cryptocurrency economies. Unlike debt-based currencies (fiat, DAI) or speculative tokens (ETH, governance tokens), TEND operates as a non-debt clearing mechanism where credits and debits balance to zero across the network. Anti-hoarding compliance (D3: Economic Velocity in the 8D sovereign profile) penalizes accumulation and rewards circulation, creating economic incentives aligned with community rather than individual wealth maximization. The system implements three currency layers---SAP (local mutual credit), TEND (labor-backed), and MYCEL (cross-community bridge)---with consciousness-gated transaction limits preventing Sybil-driven economic attacks. Restitution agreements use TEND as a restorative justice mechanism, replacing punitive fines with labor-backed community service tracked on individual source chains. The system operates across 8 financial zomes with staking, recognition, treasury, and payment coordination.

## Paper Structure

### 1. Introduction
- Mutual credit vs debt-based money
- Why governance-gated economics
- TEND as 1 hour = 1 TEND

### 2. Three-Currency Architecture
- SAP: local mutual credit (community-scoped)
- TEND: labor-backed (cross-community)
- MYCEL: bridge currency (inter-bioregion)
- Clearing house design

### 3. Anti-Hoarding Mechanism
- D3 (Economic Velocity) in sovereign profile
- Circulation metrics vs accumulation
- Demurrage as optional community parameter

### 4. Consciousness-Gated Transactions
- Tier-gated financial operations (TEND exchange: Participant, Treasury: Steward, Minting: Guardian)
- Sybil resistance through sovereign profile

### 5. Restorative Justice Integration
- RestitutionAgreement entry type
- TEND-tracked community service
- CircleTendExchange for mediated resolution

### 6. Evaluation
- Financial zome test suite
- Simulation of economic dynamics under various policy configs

### 7. Related Work
- Sardex (Sardinian mutual credit)
- WIR Bank (Swiss complementary currency)
- Circles UBI, GoodDollar
- Bancor (automated market maker for community currencies)

## Key Code References
- `mycelix-finance/zomes/payments-*/coordinator/src/lib.rs` — SAP/TEND/MYCEL
- `mycelix-finance/zomes/treasury/coordinator/src/lib.rs` — Treasury operations
- `mycelix-civic/zomes/justice-restorative/coordinator/src/lib.rs` — Restitution
- `crates/mycelix-bridge-common/src/sovereign_gate.rs` — Financial gating

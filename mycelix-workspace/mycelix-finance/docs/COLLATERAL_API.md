# Mycelix Finance: Collateral & Resource-Backed Value API

> Added March 2026. Addresses gaps identified in BIS Project Agora comparison.

## Overview

The collateral API extends Mycelix Finance with:
- **Settlement hardening**: Compost delivery queue, oracle rate attestation, mint cap enforcement
- **Collateral lifecycle**: Covenants, LTV monitoring, multi-collateral positions
- **Resource-backed value**: Energy certificates, agricultural assets as collateral
- **Selective interoperability**: One-way fiat bridge, blended oracle feeds
- **Resilience**: Circuit breaker on cross-cluster calls

All externs live in `zomes/bridge/coordinator/src/lib.rs` (the finance bridge coordinator zome). Types are defined in `types/src/lib.rs`. Shared utilities (consciousness gating, circuit breaker, oracle verification) are in `zomes/shared/src/lib.rs`.

## Consciousness Gating

All write operations require consciousness tier verification via cross-cluster calls to the identity cluster. Both tier checks are **fail-closed** -- if the identity cluster is unreachable, the operation is denied. A circuit breaker protects the identity cluster call path.

| Tier | Requirements | Operations |
|------|-------------|------------|
| Participant+ | combined consciousness >= 0.3 | `process_payment`, `register_collateral`, `update_collateral_status`, `deposit_collateral`, `redeem_collateral`, `update_collateral_health`, `register_energy_certificate`, `register_agricultural_asset`, `create_multi_collateral_position` |
| Citizen+ | identity >= 0.25, reputation >= 0.10 | `create_covenant`, `verify_energy_certificate`, `verify_agricultural_asset`, `deposit_fiat`, `verify_fiat_deposit` |

Source: `zomes/shared/src/lib.rs` module `consciousness_gating` (lines 776-858).

---

## Bridge Externs

### Payment Processing

#### `process_payment`

Process a cross-hApp payment in SAP currency.

```rust
#[hdk_extern]
pub fn process_payment(input: ProcessPaymentInput) -> ExternResult<Record>
```

**Input**: `ProcessPaymentInput`

| Field | Type | Description |
|-------|------|-------------|
| `source_happ` | `String` | Originating hApp ID |
| `from_did` | `String` | Sender DID (must match caller) |
| `to_did` | `String` | Recipient DID |
| `amount` | `u64` | Amount in micro-SAP |
| `currency` | `String` | Must be `"SAP"` |
| `reference` | `String` | Payment reference string |

**Returns**: `Record` (CrossHappPayment entry, status=Processing)

**Requires**: Participant+ tier. Caller must own `from_did`.

**Errors**:
- Currency is not `"SAP"`
- Caller DID mismatch
- Consciousness tier not met

---

#### `get_payment_history`

Get paginated payment history for a DID.

```rust
#[hdk_extern]
pub fn get_payment_history(input: GetPaymentHistoryInput) -> ExternResult<Vec<Record>>
```

**Input**: `GetPaymentHistoryInput`

| Field | Type | Description |
|-------|------|-------------|
| `did` | `String` | DID to query |
| `limit` | `Option<usize>` | Max results (default: 100) |

**Returns**: `Vec<Record>` (CrossHappPayment entries)

---

### Collateral Registration

#### `register_collateral`

Register collateral from another hApp.

```rust
#[hdk_extern]
pub fn register_collateral(input: RegisterCollateralInput) -> ExternResult<Record>
```

**Input**: `RegisterCollateralInput`

| Field | Type | Description |
|-------|------|-------------|
| `owner_did` | `String` | Owner DID (must match caller) |
| `source_happ` | `String` | Originating hApp |
| `asset_type` | `AssetType` | Type of asset (integrity-defined enum) |
| `asset_id` | `String` | Unique asset identifier |
| `value_estimate` | `u64` | Estimated value in micro-SAP |
| `currency` | `String` | Denomination currency |

**Returns**: `Record` (CollateralRegistration entry, status=Available)

**Requires**: Participant+ tier. Caller must own `owner_did`.

---

#### `update_collateral_status`

Update the status of registered collateral. Enforces covenant checks.

```rust
#[hdk_extern]
pub fn update_collateral_status(input: UpdateCollateralStatusInput) -> ExternResult<Record>
```

**Input**: `UpdateCollateralStatusInput`

| Field | Type | Description |
|-------|------|-------------|
| `collateral_id` | `String` | ID of the collateral registration |
| `owner_did` | `String` | Owner DID (must match caller) |
| `new_status` | `CollateralStatus` | Target status: `Available`, `Pledged`, `Frozen`, `Released` |

**Returns**: `Record` (updated CollateralRegistration entry)

**Requires**: Participant+ tier. Caller must own `owner_did`.

**Errors**:
- Collateral not found
- Only the owner can change status
- Cannot transition to `Pledged`, `Frozen`, or `Released` while active covenants exist

---

### Collateral Bridge Deposits

#### `deposit_collateral`

Deposit collateral to mint SAP. Verifies oracle rate against consensus.

```rust
#[hdk_extern]
pub fn deposit_collateral(input: DepositCollateralInput) -> ExternResult<Record>
```

**Input**: `DepositCollateralInput`

| Field | Type | Description |
|-------|------|-------------|
| `depositor_did` | `String` | Depositor DID (must match caller) |
| `collateral_type` | `String` | Must be `"ETH"` or `"USDC"` |
| `collateral_amount` | `u64` | Amount of collateral |
| `oracle_rate` | `f64` | Claimed conversion rate (verified against consensus) |

**Returns**: `Record` (CollateralBridgeDeposit entry, status=Pending). SAP is credited immediately via cross-zome call to `payments::credit_sap`.

**Requires**: Participant+ tier. Caller must own `depositor_did`.

**Rate limit**: Tier-scaled daily limit as percentage of total vault value:
- Newcomer: 1%, Member: 5%, Steward: 10%

**Errors**:
- Oracle rate is NaN, infinite, or non-positive
- Collateral amount is zero
- Collateral type is not `"ETH"` or `"USDC"`
- Oracle rate deviates >5% from consensus (tolerance: `ORACLE_RATE_TOLERANCE`)
- Daily rate limit exceeded
- SAP credit fails

---

#### `confirm_deposit`

Confirm a pending deposit after collateral verification.

```rust
#[hdk_extern]
pub fn confirm_deposit(deposit_id: String) -> ExternResult<Record>
```

**Input**: `String` (deposit_id)

**Returns**: `Record` (CollateralBridgeDeposit entry, status=Confirmed)

**Errors**:
- Deposit not found
- Only the original depositor can confirm
- Deposit must be in `Pending` status

---

#### `redeem_collateral`

Redeem collateral by marking a deposit as redeemed (SAP returned, collateral released).

```rust
#[hdk_extern]
pub fn redeem_collateral(deposit_id: String) -> ExternResult<Record>
```

**Input**: `String` (deposit_id)

**Returns**: `Record` (CollateralBridgeDeposit entry, status=Redeemed). SAP is debited via cross-zome call to `payments::debit_sap`.

**Requires**: Participant+ tier. Caller must own the deposit.

**Rate limit**: Same tier-scaled daily limit as deposits.

**Errors**:
- Only `Confirmed` deposits can be redeemed
- Active covenants block redemption
- Daily rate limit exceeded
- SAP debit fails

---

### Covenant Management

#### `create_covenant`

Create a restriction (covenant) on registered collateral.

```rust
#[hdk_extern]
pub fn create_covenant(input: CreateCovenantInput) -> ExternResult<Record>
```

**Input**: `CreateCovenantInput`

| Field | Type | Description |
|-------|------|-------------|
| `collateral_id` | `String` | ID of the collateral registration |
| `restriction` | `String` | Restriction type (see `CovenantRestriction` enum) |
| `beneficiary_did` | `String` | DID of the covenant beneficiary (must match caller) |
| `expires_at` | `Option<Timestamp>` | Expiry timestamp, `None` = permanent until released |

**Restriction types** (`CovenantRestriction` enum):
- `TransferLock` -- cannot transfer ownership
- `SaleProhibition` -- cannot sell the asset
- `LienHolder` -- priority claim on the asset
- `InsuranceRequirement` -- insurance must be maintained
- `CollateralPledge { obligation_id }` -- pledged for a specific obligation

**Returns**: `Record` (Covenant entry, released=false)

**Requires**: Citizen+ tier. Caller must own `beneficiary_did`.

---

#### `release_covenant`

Release an active covenant. Only the beneficiary can release.

```rust
#[hdk_extern]
pub fn release_covenant(covenant_id: String) -> ExternResult<Record>
```

**Input**: `String` (covenant_id)

**Returns**: `Record` (updated Covenant entry with released=true, released_by and released_at populated)

**Errors**:
- Covenant not found
- Only the beneficiary can release
- Covenant is already released

---

#### `check_covenants`

Query all active (unreleased) covenants for a collateral ID.

```rust
#[hdk_extern]
pub fn check_covenants(collateral_id: String) -> ExternResult<Vec<Record>>
```

**Input**: `String` (collateral_id)

**Returns**: `Vec<Record>` (active Covenant entries where released=false)

---

### Collateral Health (LTV Monitoring)

#### `update_collateral_health`

Compute and store the current LTV health status for a collateral position.

```rust
#[hdk_extern]
pub fn update_collateral_health(input: UpdateCollateralHealthInput) -> ExternResult<Record>
```

**Input**: `UpdateCollateralHealthInput`

| Field | Type | Description |
|-------|------|-------------|
| `collateral_id` | `String` | ID of the collateral registration |
| `obligation_amount` | `u64` | Outstanding obligation in micro-SAP |

**Returns**: `Record` (CollateralHealth entry)

**Requires**: Participant+ tier.

**Health tiers** (from `CollateralHealthStatus` enum):

| Status | LTV Range | Action |
|--------|-----------|--------|
| `Healthy` | < 75% | No action needed |
| `Warning` | 75-85% | Should add collateral or reduce obligation |
| `MarginCall` | 85-95% | Must add collateral within 72 hours |
| `Liquidation` | > 95% or non-finite | Eligible for governance-approved liquidation |

Note: The bridge coordinator uses slightly different thresholds (75%/85%/95%) than the canonical type constants (`LTV_WARNING_THRESHOLD=0.80`, `LTV_MARGIN_CALL_THRESHOLD=0.90`, `LTV_LIQUIDATION_THRESHOLD=0.95`). The type-level `CollateralHealthStatus::from_ltv()` method uses 80/90/95.

---

### Energy Certificates

#### `register_energy_certificate`

Register a verified energy production certificate.

```rust
#[hdk_extern]
pub fn register_energy_certificate(input: RegisterEnergyCertificateInput) -> ExternResult<Record>
```

**Input**: `RegisterEnergyCertificateInput`

| Field | Type | Description |
|-------|------|-------------|
| `project_id` | `String` | Energy project ID (links to mycelix-energy) |
| `source` | `String` | Energy source type |
| `kwh_produced` | `f64` | Quantity produced in kWh |
| `period_start` | `Timestamp` | Production period start |
| `period_end` | `Timestamp` | Production period end |
| `location_lat` | `f64` | Geographic latitude |
| `location_lon` | `f64` | Geographic longitude |
| `producer_did` | `String` | Producer DID (must match caller) |
| `terra_atlas_id` | `Option<String>` | Terra Atlas cross-reference (optional) |

**Returns**: `Record` (EnergyCertificate entry, status=Pending)

**Requires**: Participant+ tier. Caller must own `producer_did`.

**Energy source types** (`EnergySource` enum): `Solar`, `Wind`, `Hydro`, `Geothermal`, `Biomass`, `BatteryDischarge`, `Other(String)`

---

#### `verify_energy_certificate`

Verify a pending energy certificate and assign SAP value. Cross-validates against energy cluster.

```rust
#[hdk_extern]
pub fn verify_energy_certificate(input: VerifyEnergyCertificateInput) -> ExternResult<Record>
```

**Input**: `VerifyEnergyCertificateInput`

| Field | Type | Description |
|-------|------|-------------|
| `cert_id` | `String` | Certificate ID to verify |
| `verifier_did` | `String` | Verifier DID (must match caller) |
| `sap_value` | `u64` | SAP value to assign (micro-SAP) |

**Returns**: `Record` (EnergyCertificate entry, status=Verified, verifier_did and sap_value populated)

**Requires**: Citizen+ tier. Caller must own `verifier_did`.

**Cross-cluster**: Calls `energy_bridge::get_project` via `CallTargetCell::OtherRole("energy")` to verify the project exists. Non-fatal if energy cluster is unreachable (offline mode).

**Errors**:
- Certificate not found
- Certificate is not in `Pending` status

---

### Agricultural Assets

#### `register_agricultural_asset`

Register agricultural production as a verifiable asset.

```rust
#[hdk_extern]
pub fn register_agricultural_asset(input: RegisterAgriculturalAssetInput) -> ExternResult<Record>
```

**Input**: `RegisterAgriculturalAssetInput`

| Field | Type | Description |
|-------|------|-------------|
| `asset_type` | `String` | Type of agricultural asset |
| `quantity_kg` | `f64` | Quantity in kilograms |
| `location_lat` | `f64` | Geographic latitude |
| `location_lon` | `f64` | Geographic longitude |
| `production_date` | `Timestamp` | Harvest/production date |
| `viability_duration_micros` | `Option<i64>` | Expected shelf life (microseconds from production) |
| `producer_did` | `String` | Producer DID (must match caller) |

**Returns**: `Record` (AgriculturalAsset entry, status=Pending)

**Requires**: Participant+ tier. Caller must own `producer_did`.

**Asset types** (`AgriAssetType` enum):
- `Grain { crop }` -- harvested grain (maize, wheat, rice, etc.)
- `Produce { crop }` -- fresh produce (vegetables, fruits)
- `SoilAmendment { amendment_type }` -- compost, Terra Preta, biochar
- `Fertilizer { fertilizer_type }` -- NPK, organic, etc.
- `Seeds { variety }` -- seed stock
- `Livestock { species }` -- livestock (by weight)
- `Other(String)` -- other agricultural product

---

#### `verify_agricultural_asset`

Verify a pending agricultural asset and assign SAP value.

```rust
#[hdk_extern]
pub fn verify_agricultural_asset(input: VerifyAgriculturalAssetInput) -> ExternResult<Record>
```

**Input**: `VerifyAgriculturalAssetInput`

| Field | Type | Description |
|-------|------|-------------|
| `asset_id` | `String` | Asset ID to verify |
| `verifier_did` | `String` | Verifier DID (must match caller) |
| `sap_value` | `u64` | SAP value to assign (micro-SAP) |

**Returns**: `Record` (AgriculturalAsset entry, status=Verified)

**Requires**: Citizen+ tier. Caller must own `verifier_did`.

**Errors**:
- Asset not found
- Asset is not in `Pending` status

---

### Multi-Collateral Positions

#### `create_multi_collateral_position`

Create a diversified collateral position from multiple assets.

```rust
#[hdk_extern]
pub fn create_multi_collateral_position(input: CreateMultiCollateralPositionInput) -> ExternResult<Record>
```

**Input**: `CreateMultiCollateralPositionInput`

| Field | Type | Description |
|-------|------|-------------|
| `holder_did` | `String` | DID of the position holder (must match caller) |
| `components_json` | `String` | JSON array of objects with `"type"` field (asset type string) |
| `aggregate_value` | `u64` | Total SAP-equivalent value of all components |
| `aggregate_obligation` | `u64` | Total obligation against the position |

**Returns**: `Record` (MultiCollateralPosition entry with computed diversification bonus and effective LTV)

**Requires**: Participant+ tier. Caller must own `holder_did`.

**Diversification bonus**: 1% per distinct asset type in `components_json`, capped at 5%. The effective LTV is `base_ltv - diversification_bonus` (floored at 0.0).

Note: The bridge coordinator uses 1%/5% (per type, cap). The canonical type constants define `DIVERSIFICATION_BONUS_PER_CLASS = 0.05` (5%) and `DIVERSIFICATION_BONUS_CAP = 0.20` (20%) for the type-level `compute_diversification_bonus()` function, which counts distinct `CollateralAssetClass` values and starts the bonus from the 2nd distinct class.

**Asset classes** (`CollateralAssetClass` enum): `RealEstate`, `Cryptocurrency`, `EnergyCertificate`, `AgriculturalAsset`, `CarbonCredit`, `Vehicle`, `Equipment`, `Other`

**Health status**: Uses the same LTV thresholds as `update_collateral_health` (75%/85%/95% in the coordinator).

**Errors**:
- Invalid `components_json` (must be valid JSON array)
- Position must have at least one component

---

### Fiat Bridge

#### `deposit_fiat`

Create a one-way fiat deposit (fiat enters, SAP is minted). Follows the "digestion" model -- fiat is consumed to create SAP with no reverse dependency on fiat settlement systems.

```rust
#[hdk_extern]
pub fn deposit_fiat(input: DepositFiatInput) -> ExternResult<Record>
```

**Input**: `DepositFiatInput`

| Field | Type | Description |
|-------|------|-------------|
| `depositor_did` | `String` | DID of the depositor |
| `fiat_currency` | `String` | ISO 4217 code |
| `fiat_amount` | `u64` | Amount in minor units (cents) |
| `exchange_rate` | `f64` | Fiat-to-SAP conversion rate (must be finite and positive) |
| `verifier_did` | `String` | DID of the governance-approved verifier (must match caller) |
| `external_reference` | `String` | Bank transfer ID or proof of deposit |

**Supported currencies** (`SUPPORTED_FIAT_CURRENCIES`): `USD`, `ZAR`, `EUR`, `GBP`, `MXN`, `KRW`, `JPY`, `CHF`

**Returns**: `Record` (FiatBridgeDeposit entry, status=Pending). SAP is computed as `fiat_amount * exchange_rate` but not minted until verification.

**Requires**: Citizen+ tier. Caller must own `verifier_did`.

**Errors**:
- Fiat amount is zero
- Exchange rate is NaN, infinite, or non-positive

---

#### `verify_fiat_deposit`

Verify a pending fiat deposit and mint SAP to the depositor.

```rust
#[hdk_extern]
pub fn verify_fiat_deposit(input: VerifyFiatDepositInput) -> ExternResult<Record>
```

**Input**: `VerifyFiatDepositInput`

| Field | Type | Description |
|-------|------|-------------|
| `deposit_id` | `String` | Deposit ID to verify |
| `verifier_did` | `String` | Verifier DID (must match caller) |

**Returns**: `Record` (FiatBridgeDeposit entry, status=Verified). SAP is credited via cross-zome call to `payments::credit_sap`.

**Requires**: Citizen+ tier. Caller must own `verifier_did`.

**Errors**:
- Deposit not found
- Deposit is not in `Pending` status
- SAP credit fails

---

### Cross-Cluster Queries

#### `get_member_fee_tier`

Get a member's fee tier based on their MYCEL score.

```rust
#[hdk_extern]
pub fn get_member_fee_tier(member_did: String) -> ExternResult<FeeTierResponse>
```

**Returns**: `FeeTierResponse { member_did, mycel_score, tier_name, base_fee_rate }`

**Fee tiers** (`FeeTier` enum): Newcomer (MYCEL < threshold), Member, Steward. Each has a different `base_fee_rate`.

---

#### `get_member_tend_limit`

Get a member's effective TEND limit based on current network vitality.

```rust
#[hdk_extern]
pub fn get_member_tend_limit(member_did: String) -> ExternResult<TendLimitResponse>
```

**Returns**: `TendLimitResponse { member_did, vitality, tier_name, effective_limit }`

---

#### `query_sap_balance`

Query a member's SAP balance. Called by other clusters.

```rust
#[hdk_extern]
pub fn query_sap_balance(member_did: String) -> ExternResult<BalanceResponse>
```

**Returns**: `BalanceResponse { member_did, currency: "SAP", balance, available }`

---

#### `query_tend_balance`

Query a member's TEND balance. Called by other clusters.

```rust
#[hdk_extern]
pub fn query_tend_balance(member_did: String) -> ExternResult<TendBalanceResponse>
```

**Returns**: `TendBalanceResponse { member_did, balance, mycel_score, available }`

---

#### `get_finance_summary`

Get a unified financial summary for a member across all currencies.

```rust
#[hdk_extern]
pub fn get_finance_summary(member_did: String) -> ExternResult<FinanceSummaryResponse>
```

**Returns**: `FinanceSummaryResponse { member_did, sap_balance, tend_balance, mycel_score, fee_tier, fee_rate, tend_limit, tend_tier }`

---

#### `get_community_member_count`

Get the number of members in a community/DAO. Used by currency-mint governance gate.

```rust
#[hdk_extern]
pub fn get_community_member_count(dao_did: String) -> ExternResult<u32>
```

**Fallback**: When `STRICT_GOVERNANCE_MODE = false` (default), returns 0 if governance cluster is unreachable (permissive -- skips the >10-member governance requirement). When `true`, returns an error (fail-closed).

---

#### `verify_governance_proposal`

Verify a governance proposal exists and is in Approved/Executed state.

```rust
#[hdk_extern]
pub fn verify_governance_proposal(proposal_id: String) -> ExternResult<bool>
```

**Fallback**: When `STRICT_GOVERNANCE_MODE = false`, returns `true` if governance cluster is unreachable (permissive). When `true`, returns `false` (fail-closed).

---

#### `broadcast_finance_event`

Broadcast a finance event (linked to `recent_events` anchor).

```rust
#[hdk_extern]
pub fn broadcast_finance_event(input: BroadcastFinanceEventInput) -> ExternResult<Record>
```

**Input**: `BroadcastFinanceEventInput { event_type, subject_did, amount, payload }`

**Event types** (`FinanceEventType` enum): `PaymentCompleted`, `CollateralDeposited`, `CollateralRedeemed`, `CovenantCreated`, `CovenantReleased`, `CollateralHealthUpdated`, `EnergyCertificateCreated`, `AgriAssetRegistered`, `MultiCollateralCreated`, `FiatDeposited`

---

#### `health_check`

Standard bridge health endpoint.

```rust
#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<FinanceBridgeHealth>
```

**Returns**: `FinanceBridgeHealth { healthy, agent, zomes: ["payments", "treasury", "tend", "staking", "recognition", "currency_mint"] }`

---

## Settlement Hardening

### Oracle Rate Attestation

Collateral deposit operations verify claimed oracle rates against consensus via the `price_oracle` zome.

- **Tolerance**: +/-5% from median consensus price (`ORACLE_RATE_TOLERANCE = 0.05`)
- **Fallback**: Accepts the claimed rate if the oracle is unreachable (bootstrap/standalone mode)
- **Implementation**: `verify_oracle_rate_against_consensus()` in the bridge coordinator; also available as `oracle_verification::verify_oracle_rate()` in the shared crate

### Compost Delivery Queue

Failed demurrage redistributions are queued for retry via `PendingCompost` entries:

- **Max retries**: 3 per delivery (`COMPOST_MAX_RETRIES`)
- **Max queue size**: 256 entries (`PENDING_COMPOST_MAX_QUEUE`)
- **Staleness**: 24 hours (`PENDING_COMPOST_STALE_SECONDS`)
- **Queue anchor**: `pending_compost_queue`
- **Drain trigger**: Opportunistic drain on next `credit_sap`, `debit_sap`, or `apply_demurrage`
- **Pool tiers**: Local (70%), Regional (20%), Global (10%)

### Annual Mint Cap

Governance SAP mints tracked via `SapMintCapCounter` with CAS-based optimistic locking:

- **Annual maximum**: 1,000,000 SAP (`SAP_MINT_ANNUAL_MAX = 1_000_000_000_000` micro-SAP)
- **Per-proposal maximum**: 100,000 SAP (`SAP_MINT_PER_PROPOSAL_MAX = 100_000_000_000` micro-SAP)
- **Period expiry**: 365 days from `period_start_micros`
- **Anchor**: `sap_mint_annual_cap` (`SAP_MINT_CAP_ANCHOR`)
- **Methods**: `would_exceed_cap()`, `remaining_capacity()`, `is_period_expired()`

### Circuit Breaker

Cross-cluster calls are protected by a circuit breaker pattern (module `circuit_breaker` in `zomes/shared/src/lib.rs`):

- **Failure threshold**: 5 consecutive failures (`FAILURE_THRESHOLD`)
- **Cooldown**: 60 seconds (`COOLDOWN_MICROS = 60_000_000`)
- **States**: Closed (normal) -> Open (blocking) -> Half-open (probe one call)
- **Applied to**: Identity cluster consciousness gating calls
- **Storage**: Thread-local `HashMap<String, BreakerState>` (safe in WASM single-threaded context)
- **API**: `check_breaker()`, `record_success()`, `record_failure()`, `get_state()`, `reset_breaker()`, `reset_all()`

---

## Cross-Cluster Bridges

### Finance <-> Identity (Consciousness Gating)

- `verify_participant_tier()`: Calls `identity::consciousness_gating::check_participant_tier` via `CallTargetCell::OtherRole("identity")`
- `verify_citizen_tier()`: Calls `identity::consciousness_gating::check_citizen_tier` via `CallTargetCell::OtherRole("identity")`

### Finance <-> Governance

- `get_community_member_count()`: Calls `governance_bridge::get_dao_member_count` via `CallTargetCell::OtherRole("governance")`
- `verify_governance_proposal()`: Calls `governance_bridge::get_proposal_status` via `CallTargetCell::OtherRole("governance")`

### Finance <-> Energy

- `verify_energy_certificate()`: Calls `energy_bridge::get_project` via `CallTargetCell::OtherRole("energy")` to verify project existence (non-fatal if unreachable)

### Finance Internal (Cross-Zome)

- `deposit_collateral()` / `verify_fiat_deposit()`: Calls `payments::credit_sap` via `CallTargetCell::Local`
- `redeem_collateral()`: Calls `payments::debit_sap` via `CallTargetCell::Local`
- `get_member_fee_tier()`: Calls `recognition::get_mycel_score` via `CallTargetCell::Local`
- `get_member_tend_limit()`: Calls `tend::get_oracle_state` via `CallTargetCell::Local`
- `query_sap_balance()`: Calls `payments::get_sap_balance` via `CallTargetCell::Local`
- `query_tend_balance()`: Calls `tend::get_balance` via `CallTargetCell::Local`

---

## Types Reference

All types are defined in `mycelix-finance/types/src/lib.rs`:

### Settlement Hardening (Phase 1)

| Type | Kind | Description |
|------|------|-------------|
| `PendingCompost` | struct | Failed compost delivery queued for retry |
| `CompostPoolTier` | enum | `Local`, `Regional`, `Global` |
| `SapMintCapCounter` | struct | CAS-based annual mint cap tracker |

### Collateral Lifecycle (Phase 2)

| Type | Kind | Description |
|------|------|-------------|
| `Covenant` | struct | Restriction on registered collateral |
| `CovenantRestriction` | enum | `TransferLock`, `SaleProhibition`, `LienHolder`, `InsuranceRequirement`, `CollateralPledge { obligation_id }` |
| `CollateralHealth` | struct | LTV health snapshot |
| `CollateralHealthStatus` | enum | `Healthy`, `Warning`, `MarginCall`, `Liquidation` |

### Resource-Backed Value (Phase 3)

| Type | Kind | Description |
|------|------|-------------|
| `EnergyCertificate` | struct | Verified energy production certificate |
| `EnergySource` | enum | `Solar`, `Wind`, `Hydro`, `Geothermal`, `Biomass`, `BatteryDischarge`, `Other(String)` |
| `CertificateStatus` | enum | `Pending`, `Verified`, `Rejected { reason }`, `Pledged`, `Retired` |
| `AgriculturalAsset` | struct | Registered agricultural asset |
| `AgriAssetType` | enum | `Grain { crop }`, `Produce { crop }`, `SoilAmendment`, `Fertilizer`, `Seeds`, `Livestock`, `Other(String)` |
| `AgriAssetStatus` | enum | `Pending`, `Verified`, `Rejected { reason }`, `Pledged`, `Consumed` |
| `MultiCollateralPosition` | struct | Diversified multi-asset collateral position |
| `CollateralComponent` | struct | Single component of a multi-collateral position |
| `CollateralAssetClass` | enum | `RealEstate`, `Cryptocurrency`, `EnergyCertificate`, `AgriculturalAsset`, `CarbonCredit`, `Vehicle`, `Equipment`, `Other` |

### Selective Interoperability (Phase 4)

| Type | Kind | Description |
|------|------|-------------|
| `FiatBridgeDeposit` | struct | One-way fiat deposit record |
| `FiatDepositStatus` | enum | `Pending`, `Verified`, `Rejected { reason }` |
| `ExternalOracleFeed` | struct | External price feed subscription |

---

## Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `ORACLE_RATE_TOLERANCE` | 0.05 (5%) | types | Max deviation from consensus |
| `COMPOST_MAX_RETRIES` | 3 | types | Delivery retry attempts |
| `PENDING_COMPOST_MAX_QUEUE` | 256 | types | Max queued deliveries |
| `PENDING_COMPOST_STALE_SECONDS` | 86,400 (24h) | types | Queue entry staleness |
| `COMPOST_LOCAL_PCT` | 70 | types | Local pool share |
| `COMPOST_REGIONAL_PCT` | 20 | types | Regional pool share |
| `COMPOST_GLOBAL_PCT` | 10 | types | Global pool share |
| `SAP_MINT_PER_PROPOSAL_MAX` | 100,000 SAP (100B micro) | types | Per-proposal mint cap |
| `SAP_MINT_ANNUAL_MAX` | 1,000,000 SAP (1T micro) | types | Annual mint cap |
| `SAP_MINT_CAP_ANCHOR` | `"sap_mint_annual_cap"` | types | DHT anchor for cap counter |
| `LTV_WARNING_THRESHOLD` | 0.80 (80%) | types | Warning tier |
| `LTV_MARGIN_CALL_THRESHOLD` | 0.90 (90%) | types | Margin call tier |
| `LTV_LIQUIDATION_THRESHOLD` | 0.95 (95%) | types | Liquidation tier |
| `MARGIN_CALL_GRACE_HOURS` | 72 | types | Hours before forced liquidation |
| `DIVERSIFICATION_BONUS_PER_CLASS` | 0.05 (5%) | types | Per distinct asset class |
| `DIVERSIFICATION_BONUS_CAP` | 0.20 (20%) | types | Maximum diversification bonus |
| `ORACLE_BLEND_ALPHA` | 0.70 | types | Community weight in blended oracle |
| `SUPPORTED_FIAT_CURRENCIES` | `["USD","ZAR","EUR","GBP","MXN","KRW","JPY","CHF"]` | types | Accepted fiat currencies |
| `FAILURE_THRESHOLD` | 5 | shared | Circuit breaker failure count |
| `COOLDOWN_MICROS` | 60,000,000 (60s) | shared | Circuit breaker cooldown |
| `COMMUNITY_GOVERNANCE_THRESHOLD` | 10 | shared | Members requiring governance proposal |
| `DEMURRAGE_RATE` | 0.02 (2%) | types | Annual demurrage rate |
| `DEMURRAGE_EXEMPT_FLOOR` | 1,000,000,000 (1000 SAP) | types | Demurrage-exempt balance floor |
| `INALIENABLE_RESERVE_RATIO` | 0.25 (25%) | types | Minimum reserve ratio |
| `STRICT_GOVERNANCE_MODE` | false | bridge coordinator | Fail-closed governance mode |
| `DAY_MICROS` | 86,400,000,000 | bridge coordinator | 24 hours in microseconds |

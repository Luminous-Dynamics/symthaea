# Mycelix Sweettest Integration Tests

End-to-end integration tests using Holochain's [sweettest](https://docs.rs/holochain/latest/holochain/sweettest/index.html) framework with real conductors.

## Quick Start

```bash
# From mycelix-workspace/
just test-sweettest             # Run all 33 tests
just test-sweettest-identity    # Run identity suite only (4 tests)
just test-sweettest-governance  # Run governance suite only (3 tests)
just test-sweettest-bridge      # Run bridge suite only (2 tests)
just test-sweettest-privacy     # Run privacy/ZK suite only (6 tests)
just test-sweettest-edunet      # Run edunet suite only (7 tests)
just test-sweettest-supplychain # Run supplychain suite only (11 tests)
```

Or directly:
```bash
cd tests/sweettest
cargo test --release -- --ignored --nocapture
```

## IMPORTANT: `--release` is Required

Sweettest MUST run with `--release` due to Holochain's hardcoded 5-minute nonce lifetime (`FRESH_NONCE_EXPIRES_AFTER` in `holochain_nonce-0.6.0`). In debug mode, conductor startup + WASM compilation exceeds this timeout, causing cryptic failures.

## Test Suites

### EduNet Workflow (7 tests) - NEW
**File:** `tests/edunet_workflow.rs`

| Test | Agents | Description |
|------|--------|-------------|
| `test_create_and_get_course` | 1 | Create course, retrieve by hash |
| `test_list_courses` | 1 | Create multiple courses, verify list |
| `test_enrollment_and_progress` | 2 | Instructor creates, learner enrolls and updates progress |
| `test_record_activity` | 1 | Record quiz activity with outcome |
| `test_fl_gradient_submission` | 2 | Create FL model, submit clipped gradient |
| `test_issue_completion_credential` | 2 | Issue and verify course completion credential |
| `test_curriculum_proposal_voting` | 3 | DAO proposal for curriculum change |

### Identity Workflow (4 tests)
**File:** `tests/identity_workflow.rs`

| Test | Agents | Description |
|------|--------|-------------|
| `test_create_and_resolve_did` | 1 | Create DID, retrieve by agent pubkey |
| `test_did_cross_agent_resolution` | 2 | Alice creates DID, Bob resolves via DHT |
| `test_did_deactivation_propagation` | 2 | DID deactivation visible to other agents |
| `test_service_endpoint_crud` | 1 | Add service endpoint to DID document |

### Governance Workflow (3 tests)
**File:** `tests/governance_workflow.rs`

| Test | Agents | Description |
|------|--------|-------------|
| `test_create_proposal_and_vote` | 2 | Proposer creates, voter casts vote |
| `test_multi_voter_quorum` | 4 | 4 agents vote, verify propagation |
| `test_vote_delegation` | 2 | Alice delegates voting power to Bob |

### Cross-hApp Bridge (2 tests)
**File:** `tests/cross_happ_bridge.rs`

| Test | Agents | Description |
|------|--------|-------------|
| `test_identity_to_governance_flow` | 1 | Single agent with identity + governance hApps |
| `test_multi_agent_multi_happ_sync` | 2 | Two agents, each with both hApps, DHT sync |

### Privacy/ZK Attestation (6 tests)
**File:** `tests/privacy_zk_attestation.rs`

| Test | Agents | Description |
|------|--------|-------------|
| `test_submit_proof_attestation` | 1 | Submit ZK proof attestation |
| `test_get_subject_attestations` | 1 | Submit 3, retrieve all by subject hash |
| `test_get_attestations_by_type` | 1 | Filter attestations by proof type |
| `test_has_valid_attestation` | 1 | Check validity before/after submission |
| `test_attestation_expiration` | 1 | Expired attestation not counted as valid |
| `test_multi_agent_attestation` | 2 | Agent 1 submits, Agent 2 retrieves via DHT |

### Supply Chain Workflow (11 tests) - NEW
**File:** `tests/supplychain_workflow.rs`

| Test | Agents | Description |
|------|--------|-------------|
| `test_create_and_get_item` | 1 | Create inventory item, retrieve by hash |
| `test_list_inventory_items` | 1 | Create multiple items, verify list |
| `test_stock_level_management` | 1 | Update stock at multiple locations, verify totals |
| `test_stock_movement` | 1 | Record inbound movement, verify history |
| `test_create_and_get_shipment` | 1 | Create shipment with carrier/destination |
| `test_shipment_tracking_events` | 2 | Add tracking events (pickup, in-transit) |
| `test_create_and_get_claim` | 1 | Create provenance claim, retrieve it |
| `test_provenance_chain` | 3 | Build multi-actor chain (farmer→processor→distributor) |
| `test_multi_agent_claim_visibility` | 2 | Claim visible to verifier via DHT |
| `test_provider_profile` | 1 | Create and retrieve provider profile |

## Architecture

```
tests/sweettest/
├── Cargo.toml
├── README.md
└── tests/
    ├── harness.rs               # Shared test utilities
    ├── identity_workflow.rs     # Identity hApp tests (4)
    ├── governance_workflow.rs   # Governance hApp tests (3)
    ├── cross_happ_bridge.rs     # Multi-hApp tests (2)
    ├── privacy_zk_attestation.rs # LUCID privacy tests (6)
    ├── edunet_workflow.rs       # EduNet hApp tests (7)
    └── supplychain_workflow.rs  # Supply chain hApp tests (11)
```

### Test Harness

`harness.rs` provides:

- **`DnaPaths`**: DNA bundle locations for each hApp
- **`TestAgent`**: Wrapper with conductor, cell, and `call_zome_fn()` helper
- **`setup_test_agents()`**: Creates N agents with peer exchange for DHT sync
- **`wait_for_dht_sync()`**: 2-second delay for DHT propagation

### DNA Paths

Tests expect pre-built DNA bundles at:
- `mycelix-identity/dna/mycelix_identity_dna.dna`
- `mycelix-governance/dna/mycelix_governance.dna`
- `mycelix-workspace/happs/lucid/lucid.dna`
- `mycelix-edunet/dna/edunet.dna`
- `mycelix-supplychain/holochain/dna/supplychain.dna`

Build them with:
```bash
cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown && hc dna pack dna/
cd mycelix-governance && cargo build --release --target wasm32-unknown-unknown && hc dna pack dna/
cd mycelix-edunet && cargo build --release --target wasm32-unknown-unknown && hc dna pack dna/
cd mycelix-supplychain/holochain && cargo build --release --target wasm32-unknown-unknown && hc dna pack dna/
```

## Version Compatibility

| Component | Version |
|-----------|---------|
| holochain | 0.6.0 |
| hdk | 0.6.0 |
| hdi | 0.7.0 |
| sweettest | 0.6.0 (via holochain crate) |

## CI Integration

The GitHub Actions workflow (`holochain.yml`) runs all 4 suites:

```yaml
- name: Run sweettest - Identity (4 tests)
  run: cargo test --release identity -- --ignored --test-threads=1 --nocapture

- name: Run sweettest - Governance (3 tests)
  run: cargo test --release governance -- --ignored --test-threads=1 --nocapture

- name: Run sweettest - Cross-hApp Bridge (2 tests)
  run: cargo test --release bridge -- --ignored --test-threads=1 --nocapture

- name: Run sweettest - Privacy/ZK Attestation (6 tests)
  run: cargo test --release privacy -- --ignored --test-threads=1 --nocapture
```

## Troubleshooting

### "Nonce is stale" or timeout errors
You're running in debug mode. Use `--release`.

### "DNA bundle not found"
Build the DNA first:
```bash
cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown
hc dna pack dna/ -o dna/mycelix_identity_dna.dna
```

### "getrandom" / wasm-bindgen errors
Ensure `.cargo/config.toml` has:
```toml
[target.wasm32-unknown-unknown]
rustflags = ['--cfg', 'getrandom_backend="custom"']
```

And workspace `Cargo.toml` uses:
```toml
getrandom_03 = { package = "getrandom", version = "0.3" }
```

Do NOT use `getrandom v0.2 features=["js"]`.

### Tests hang or fail intermittently
- Increase `wait_for_dht_sync()` duration for more agents
- Use `--test-threads=1` to serialize test execution
- Check for port conflicts (each conductor uses random ports)

## Adding New Tests

1. Add test function with `#[ignore]` attribute (requires DNA bundle)
2. Use `setup_test_agents()` for conductor setup
3. Use `call_zome_fn()` for zome calls
4. Add `wait_for_dht_sync()` between multi-agent operations
5. Update this README with test count and description

Example:
```rust
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_new_feature() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;

    // Agent 0 does something
    let result: Record = agents[0].call_zome_fn("zome", "function", input).await;

    wait_for_dht_sync().await;

    // Agent 1 verifies
    let verified: bool = agents[1].call_zome_fn("zome", "verify", result).await;
    assert!(verified);
}
```

## Related Documentation

- [ECOSYSTEM_STATUS.md](../../ECOSYSTEM_STATUS.md) - Full test coverage breakdown
- [Holochain Sweettest Docs](https://docs.rs/holochain/latest/holochain/sweettest/)
- Commit `1deaeb047` - getrandom fix across ecosystem

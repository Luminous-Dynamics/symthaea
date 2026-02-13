---
description: "Create or run Holochain sweettest integration tests for a Mycelix zome"
---

# Mycelix Sweettest Helper

Create a new sweettest integration test or run existing ones. Sweettests validate zome behavior with a real Holochain conductor.

Arguments: $ARGUMENTS

- If args are a zome/hApp name: scaffold a new sweettest for it
- If args are "run" or "run {name}": execute sweettest(s)
- If no args: list available sweettests and their status

## Context - Existing Patterns

### Test harness
@mycelix-workspace/tests/sweettest/tests/harness.rs

### Example sweettest (identity PQC - most comprehensive)
Read a few lines from the top to understand the pattern:
@mycelix-workspace/tests/sweettest/tests/identity_pqc_workflow.rs

### Sweettest Cargo.toml
@mycelix-workspace/tests/sweettest/Cargo.toml

## CRITICAL Requirements for Sweettests

1. **Release mode required**: `cargo test --release -- --ignored`
   - Debug mode causes Holochain nonce lifetime (5 min) to expire during test
2. **Mirror types**: Cannot import WASM zome crate types directly (symbol conflicts)
   - Must define mirror structs with identical serde layout
3. **Test attributes**: Every test MUST have:
   ```rust
   #[tokio::test(flavor = "multi_thread")]
   #[serial]
   #[ignore]
   ```
4. **DHT sync waits**: Add `tokio::time::sleep(Duration::from_secs(2))` between write and read operations
5. **Single thread**: Use `#[serial]` from `serial_test` crate for sequential execution

## Scaffolding a New Sweettest

### Step 1: Create Test File

Create `mycelix-workspace/tests/sweettest/tests/{name}_workflow.rs`:

```rust
//! Sweettest integration tests for {Name}
//!
//! Run with: cargo test --release -p sweettest -- --ignored --nocapture

use holochain::sweettest::*;
use holochain_types::prelude::*;
use serde::{Deserialize, Serialize};
use serial_test::serial;
use std::time::Duration;

mod harness;
use harness::*;

// ============================================================
// Mirror Types (must match zome entry/input types exactly)
// ============================================================

#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub struct Create{Entity}Input {
    // Mirror the exact fields from the zome's input type
    pub field1: String,
    pub field2: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub struct {Entity}Entry {
    // Mirror the exact fields from the zome's entry type
    pub field1: String,
    pub field2: u64,
}

// ============================================================
// Tests
// ============================================================

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_create_{entity}() {
    // Setup
    let dna_path = DnaPaths::new("{happ_name}");
    let agents = setup_test_agents(&dna_path.dna_path(), "{happ_name}", 2).await;
    let alice = &agents[0];

    // Create
    let input = Create{Entity}Input {
        field1: "test".to_string(),
        field2: 42,
    };

    let record: Record = alice
        .conductor
        .call(&alice.cell.zome("{zome_name}"), "create_{entity}", input)
        .await;

    assert!(record.action_hashed().is_some());

    // Wait for DHT propagation
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Read back (from another agent to verify DHT sync)
    let bob = &agents[1];
    let action_hash = record.action_address().clone();

    let fetched: Option<Record> = bob
        .conductor
        .call(&bob.cell.zome("{zome_name}"), "get_{entity}", action_hash)
        .await;

    assert!(fetched.is_some(), "Bob should be able to read Alice's record");
}

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_{entity}_validation_rejects_invalid() {
    let dna_path = DnaPaths::new("{happ_name}");
    let agents = setup_test_agents(&dna_path.dna_path(), "{happ_name}", 1).await;
    let alice = &agents[0];

    // Attempt to create with invalid data
    let invalid_input = Create{Entity}Input {
        field1: "".to_string(),  // Should be rejected by validation
        field2: 0,
    };

    // This should fail validation
    let result: Result<Record, _> = alice
        .conductor
        .call_fallible(&alice.cell.zome("{zome_name}"), "create_{entity}", invalid_input)
        .await;

    assert!(result.is_err(), "Empty field1 should be rejected by validation");
}
```

### Step 2: Update Sweettest Cargo.toml

If the test needs a new DNA path, ensure the DnaPaths struct in harness.rs supports it, or add a custom path.

### Step 3: Verify Mirror Types

Compare mirror types against the actual zome types:
- Read the zome's integrity `lib.rs` for entry types
- Read the zome's coordinator `lib.rs` for input types
- Ensure field names, types, and order match exactly
- Serde attributes must match (especially `rename`, `default`, `skip_serializing_if`)

## Running Sweettests

### Run all sweettests:
```bash
cd mycelix-workspace
cargo test --release -p sweettest -- --ignored --nocapture
```

### Run a specific sweettest:
```bash
cd mycelix-workspace
cargo test --release -p sweettest -- --ignored --nocapture test_create_{entity}
```

### Run a specific test file:
```bash
cd mycelix-workspace
cargo test --release -p sweettest --test {name}_workflow -- --ignored --nocapture
```

## Common Issues

### "nonce has expired"
- You're running in debug mode. Use `--release`.

### "WASM symbol conflict"
- You imported the zome crate directly. Use mirror types instead.

### "DHT not synced"
- Add `tokio::time::sleep(Duration::from_secs(3)).await` after writes.

### "conductor not found"
- Ensure Holochain is installed: `which holochain`
- Use `nix develop` in mycelix-workspace to get Holochain tools.

## Test File Naming Convention

- `{happ_name}_workflow.rs` for full workflow tests
- `{feature}_e2e.rs` for end-to-end cross-hApp tests
- Place in `mycelix-workspace/tests/sweettest/tests/`

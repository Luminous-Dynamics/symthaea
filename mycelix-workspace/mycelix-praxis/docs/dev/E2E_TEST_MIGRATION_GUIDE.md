# E2E Test Migration Guide - Holochain 0.6 API

**Date**: 2025-12-16
**Status**: Implementation Ready
**Issue**: 115 compilation errors due to Holochain 0.6 sweettest API changes

---

## 🎯 Migration Overview

The Holochain 0.6 sweettest framework changed the `conductor.call()` API from 4 parameters to 3 parameters:

**Old API** (what tests currently use):
```rust
conductor.call(&agent_pubkey, "zome_name", "function_name", input).await
// ↑ 4 parameters: agent, zome, function, input
```

**New API** (Holochain 0.6 requires):
```rust
conductor.call(&sweet_zome, "function_name", input).await
// ↑ 3 parameters: zome reference, function, input
```

**Key Changes**:
1. First parameter changed from `&AgentPubKey` to `&SweetZome`
2. Zome name is now implicit in the `SweetZome` reference
3. Need to extract `SweetZome` from installed apps instead of using `AgentPubKey`

---

## 📋 Step-by-Step Migration Plan

### Phase 1: Update Setup Functions

#### 1.1 Modify `install_praxis_dna()` to Return Zome References

**Current Code** (tests/e2e_test.rs:36-52):
```rust
async fn install_praxis_dna(
    conductor: &mut SweetConductor,
) -> Result<(SweetAgents, DnaHash), Box<dyn std::error::Error>> {
    let dna_path = std::path::PathBuf::from("../dna/praxis.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await?;

    let agents = SweetAgents::get(conductor, 2).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-praxis", &agents, &[dna.clone()])
        .await?;

    let dna_hash = dna.dna_hash().clone();
    Ok((agents, dna_hash))
}
```

**Updated Code** (Holochain 0.6):
```rust
async fn install_praxis_dna(
    conductor: &mut SweetConductor,
) -> Result<(Vec<SweetCell>, DnaHash), Box<dyn std::error::Error>> {
    let dna_path = std::path::PathBuf::from("../dna/praxis.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await?;

    let agents = SweetAgents::get(conductor, 2).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-praxis", &agents, &[dna.clone()])
        .await?;

    // Extract cells from installed apps
    let mut cells = Vec::new();
    for app in apps.iter() {
        cells.push(app.cells()[0].clone());  // Get first cell from each app
    }

    let dna_hash = dna.dna_hash().clone();
    Ok((cells, dna_hash))
}
```

#### 1.2 Modify `setup_conductor_with_agents()` to Return Zome References

**Current Code** (tests/e2e_test.rs:54-78):
```rust
async fn setup_conductor_with_agents(
    num_agents: usize,
) -> Result<(SweetConductor, Vec<AgentPubKey>), Box<dyn std::error::Error>> {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_path = std::path::PathBuf::from("../dna/praxis.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await?;

    let agents = SweetAgents::get(&conductor, num_agents).await;
    let _apps = conductor
        .setup_app_for_agents("mycelix-praxis", &agents, &[dna.clone()])
        .await?;

    // Extract agent public keys
    let agent_keys: Vec<AgentPubKey> = agents.iter().map(|a| a.clone()).collect();

    Ok((conductor, agent_keys))
}
```

**Updated Code** (Holochain 0.6):
```rust
async fn setup_conductor_with_agents(
    num_agents: usize,
) -> Result<(SweetConductor, Vec<SweetCell>), Box<dyn std::error::Error>> {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_path = std::path::PathBuf::from("../dna/praxis.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await?;

    let agents = SweetAgents::get(&conductor, num_agents).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-praxis", &agents, &[dna.clone()])
        .await?;

    // Extract cells from all apps (one cell per agent)
    let mut cells = Vec::new();
    for app in apps.iter() {
        cells.push(app.cells()[0].clone());
    }

    Ok((conductor, cells))
}
```

### Phase 2: Create Helper Functions for Zome Access

Add these helper functions to make test code cleaner:

```rust
/// Get a specific zome from a cell by zome name
fn get_zome<'a>(cell: &'a SweetCell, zome_name: &str) -> &'a SweetZome {
    cell.zome(zome_name)
}

/// Helper to get learning zome from a cell
fn learning_zome(cell: &SweetCell) -> &SweetZome {
    cell.zome("learning")
}

/// Helper to get FL zome from a cell
fn fl_zome(cell: &SweetCell) -> &SweetZome {
    cell.zome("fl")
}

/// Helper to get credential zome from a cell
fn credential_zome(cell: &SweetCell) -> &SweetZome {
    cell.zome("credential")
}

/// Helper to get DAO zome from a cell
fn dao_zome(cell: &SweetCell) -> &SweetZome {
    cell.zome("dao")
}
```

### Phase 3: Update All Test Zome Calls

#### Example: Scenario 1 - Learning Journey

**Before** (Old API):
```rust
let (mut conductor, agent_keys) = setup_conductor_with_agents(2).await?;
let alice = &agent_keys[0];
let bob = &agent_keys[1];

// Create course
let course_hash: ActionHash = conductor
    .call(
        alice,           // ❌ &AgentPubKey - OLD API
        "learning",      // Zome name
        "create_course",
        course_input,
    )
    .await?;
```

**After** (New API):
```rust
let (mut conductor, cells) = setup_conductor_with_agents(2).await?;
let alice_cell = &cells[0];
let bob_cell = &cells[1];

// Create course
let course_hash: ActionHash = conductor
    .call(
        learning_zome(alice_cell),  // ✅ &SweetZome - NEW API
        "create_course",             // Function name only
        course_input,
    )
    .await?;
```

#### Example: Scenario 5 - DAO Governance

**Before** (Old API):
```rust
// Cast votes
for i in 0..5 {
    let vote_input = CastVoteInput {
        proposal_id: proposal_id.clone(),
        proposal_hash: proposal_hash.clone(),
        choice: VoteChoice::For,
        justification: Some(format!("I support this proposal - Agent {}", i)),
    };

    conductor
        .call(
            &agents[i + 1],  // ❌ &AgentPubKey - OLD API
            "dao",
            "cast_vote",
            vote_input,
        )
        .await
        .expect("Failed to cast vote");
}
```

**After** (New API):
```rust
// Cast votes
for i in 0..5 {
    let vote_input = CastVoteInput {
        proposal_id: proposal_id.clone(),
        proposal_hash: proposal_hash.clone(),
        choice: VoteChoice::For,
        justification: Some(format!("I support this proposal - Agent {}", i)),
    };

    conductor
        .call(
            dao_zome(&cells[i + 1]),  // ✅ &SweetZome - NEW API
            "cast_vote",               // Function name only
            vote_input,
        )
        .await
        .expect("Failed to cast vote");
}
```

---

## 🔧 Migration Checklist

### Helper Functions
- [ ] Add `get_zome()` helper function
- [ ] Add `learning_zome()` helper
- [ ] Add `fl_zome()` helper
- [ ] Add `credential_zome()` helper
- [ ] Add `dao_zome()` helper

### Setup Functions
- [ ] Update `install_praxis_dna()` to return `Vec<SweetCell>`
- [ ] Update `setup_conductor_with_agents()` to return `Vec<SweetCell>`

### Scenario 1: Complete Learning Journey (lines 97-175)
- [ ] Update variable names: `agent_keys` → `cells`
- [ ] Update all `create_course` calls (~3)
- [ ] Update all `enroll_in_course` calls (~2)
- [ ] Update all `update_progress` calls (~5)
- [ ] Update all `get_learner_progress` calls (~2)
- [ ] Update credential issuance calls (~3)

### Scenario 2: FL Round Lifecycle (lines 177-360)
- [ ] Update variable names: `agent_keys` → `cells`
- [ ] Update all `create_round` calls (~2)
- [ ] Update all `submit_update` calls (~10)
- [ ] Update all `get_round_updates` calls (~2)
- [ ] Update all `aggregate_round` calls (~2)
- [ ] Update privacy parameter calls (~5)

### Scenario 3: Credential Issuance Flow (lines 362-520)
- [ ] Update variable names: `agent_keys` → `cells`
- [ ] Update all `issue_credential` calls (~5)
- [ ] Update all `verify_credential` calls (~5)
- [ ] Update all `revoke_credential` calls (~2)
- [ ] Update all `get_credentials_by_learner` calls (~3)

### Scenario 4: Multi-Zome Integration (lines 522-680)
- [ ] Update variable names: `agent_keys` → `cells`
- [ ] Update learning zome calls (~5)
- [ ] Update FL zome calls (~5)
- [ ] Update credential zome calls (~5)
- [ ] Update cross-zome integration calls (~5)

### Scenario 5: DAO Governance Lifecycle (lines 682-1279)
- [ ] Update variable names: `agent_keys` → `cells`
- [ ] Update all `create_proposal` calls (~2)
- [ ] Update all `cast_vote` calls (~10)
- [ ] Update all `tally_votes` calls (~2)
- [ ] Update all `get_proposal` calls (~3)
- [ ] Update all query calls (~3)

---

## ⏱️ Estimated Time

| Task | Duration | Notes |
|------|----------|-------|
| Add helper functions | 15 min | Straightforward additions |
| Update setup functions | 30 min | Need to understand SweetCell API |
| Scenario 1 (20 calls) | 30 min | Template for other scenarios |
| Scenario 2 (30 calls) | 45 min | More complex aggregation |
| Scenario 3 (25 calls) | 30 min | Credential verification |
| Scenario 4 (20 calls) | 30 min | Cross-zome coordination |
| Scenario 5 (20 calls) | 30 min | DAO governance |
| Compilation verification | 15 min | `cargo check --test e2e_test` |
| **Total** | **4 hours** | Systematic migration |

---

## ✅ Success Criteria

1. **Compilation Success**: `cargo check --test e2e_test` passes with 0 errors
2. **API Correctness**: All `conductor.call()` uses 3 parameters with `&SweetZome`
3. **Test Structure**: All 5 scenarios updated consistently
4. **Code Quality**: No copy-paste errors, consistent naming
5. **Documentation**: This guide updated with any discoveries

---

## 🚀 Next Steps

1. **Start with Scenario 1**: Fix the complete learning journey first as a proof-of-concept
2. **Verify Compilation**: After Scenario 1, check if it compiles
3. **Apply Pattern**: Use Scenario 1 as template for Scenarios 2-5
4. **Full Verification**: Run `cargo check --test e2e_test` after all scenarios
5. **Run Tests**: Build DNA bundle and execute E2E tests

---

**Status**: Ready for implementation
**Blocked by**: None - all information available
**Next Action**: Begin Phase 1 - Update setup functions

# E2E Test Holochain 0.6 API Update Required

**Date**: 2025-12-16
**Status**: Compilation Blocked - API Migration Needed
**Issue**: E2E tests use outdated sweettest API patterns

---

## 🚨 Current Status

### Compilation Errors
- **115 compilation errors** in `tests/e2e_test.rs`
- All errors stem from incorrect `SweetConductor.call()` method signature
- Tests were written for an older Holochain API

### Root Cause
The Holochain 0.6 sweettest framework changed the `call` method signature:

**Old Pattern** (what tests currently use):
```rust
conductor.call(&agent_pubkey, "zome_name", "function_name", input).await
```

**New Pattern** (Holochain 0.6 requires):
```rust
conductor.call(&sweet_zome, "function_name", input).await
```

**Key Changes**:
1. First parameter changed from `&AgentPubKey` to `&SweetZome`
2. Reduced from 4 parameters to 3 (zome name now implicit in `SweetZome`)

---

## 📋 Affected Test Scenarios

All 5 E2E test scenarios are affected:

### 1. Scenario 1: Complete Learning Journey
- **File**: `tests/e2e_test.rs` lines 97-175
- **Errors**: ~20 `call()` invocations need updating
- **Functions Called**: `create_course`, `enroll_in_course`, `update_progress`, etc.

### 2. Scenario 2: FL Round Lifecycle
- **File**: `tests/e2e_test.rs` lines 177-360
- **Errors**: ~30 `call()` invocations need updating
- **Functions Called**: `create_round`, `submit_update`, `aggregate_round`, etc.

### 3. Scenario 3: Credential Issuance Flow
- **File**: `tests/e2e_test.rs` lines 362-520
- **Errors**: ~25 `call()` invocations need updating
- **Functions Called**: `issue_credential`, `verify_credential`, etc.

### 4. Scenario 4: Multi-Zome Integration
- **File**: `tests/e2e_test.rs` lines 522-680
- **Errors**: ~20 `call()` invocations need updating
- **Complexity**: Calls multiple zomes in sequence

### 5. Scenario 5: DAO Governance Lifecycle
- **File**: `tests/e2e_test.rs` lines 682-1279
- **Errors**: ~20 `call()` invocations need updating
- **Functions Called**: `create_proposal`, `cast_vote`, `tally_votes`, etc.

---

## 🔧 Required API Changes

### Pattern 1: Setup Function Updates

**Current (Incorrect)**:
```rust
async fn setup_conductor_with_agents(
    num_agents: usize,
) -> Result<(SweetConductor, Vec<AgentPubKey>), Box<dyn std::error::Error>> {
    // Returns AgentPubKey for calling zomes
    let agent_keys: Vec<AgentPubKey> = agents.iter().map(|a| a.clone()).collect();
    Ok((conductor, agent_keys))
}
```

**Needed (Correct for Holochain 0.6)**:
```rust
async fn setup_conductor_with_agents(
    num_agents: usize,
) -> Result<(SweetConductor, Vec<SweetZome>), Box<dyn std::error::Error>> {
    // Extract zomes from installed apps
    let mut zomes = Vec::new();
    for app in apps {
        // Get zome from app using Holochain 0.6 API
        // Exact API needs verification from Holochain docs
        let zome = app.get_zome("zome_name")?;  // Placeholder - verify actual API
        zomes.push(zome);
    }
    Ok((conductor, zomes))
}
```

### Pattern 2: Zome Call Updates

**Current (Incorrect)**:
```rust
let course_hash: ActionHash = conductor
    .call(
        &alice,  // AgentPubKey - WRONG!
        "learning",
        "create_course",
        course_input,
    )
    .await?;
```

**Needed (Correct for Holochain 0.6)**:
```rust
// Get the learning zome for this agent/app
let learning_zome = &zomes[0]; // Or however zomes are indexed

let course_hash: ActionHash = conductor
    .call(
        learning_zome,  // &SweetZome - CORRECT!
        "create_course",
        course_input,
    )
    .await?;
```

---

## 🎯 Migration Strategy

### Phase 1: Research Correct API (Immediate)
1. **Read Holochain 0.6 Documentation**
   - Check `holochain/src/sweettest/` source code
   - Find example tests in Holochain codebase
   - Understand `SweetZome` extraction from `SweetAgents`/`SweetApps`

2. **Create API Reference Document**
   - Document correct method signatures
   - Provide working examples
   - Note any breaking changes

### Phase 2: Update Helper Functions (1-2 hours)
1. **Modify `setup_conductor()`**
   - Return `SweetZome` references instead of `AgentPubKey`
   - Extract zomes from installed apps
   - Handle multiple zomes per agent

2. **Modify `setup_conductor_with_agents()`**
   - Same changes as above
   - Support N agents with N sets of zomes

3. **Create Zome Access Helpers**
   ```rust
   fn get_learning_zome(apps: &[SweetApp], agent_idx: usize) -> &SweetZome;
   fn get_fl_zome(apps: &[SweetApp], agent_idx: usize) -> &SweetZome;
   fn get_credential_zome(apps: &[SweetApp], agent_idx: usize) -> &SweetZome;
   fn get_dao_zome(apps: &[SweetApp], agent_idx: usize) -> &SweetZome;
   ```

### Phase 3: Update All Test Scenarios (2-3 hours)
1. **Scenario 1**: Learning journey (20 call sites)
2. **Scenario 2**: FL round lifecycle (30 call sites)
3. **Scenario 3**: Credential flow (25 call sites)
4. **Scenario 4**: Multi-zome integration (20 call sites)
5. **Scenario 5**: DAO governance (20 call sites)

**Total**: ~115 call sites need updating

### Phase 4: Compilation & Testing (1 hour)
1. **Verify Compilation**: `cargo check --test e2e_test`
2. **Build Tests**: `cargo test --test e2e_test --no-run`
3. **Run Tests**: Build DNA bundle and execute tests

---

## 📚 Resources Needed

### Documentation
- [ ] Holochain 0.6 sweettest API reference
- [ ] Example tests from Holochain core repository
- [ ] Migration guide (if available)

### Code References
- **Holochain Source**: `~/.cargo/registry/src/.../holochain-0.6.0/src/sweettest/`
- **API Method**: `sweet_conductor_handle.rs:13` - `pub async fn call<I, O>(...)`

---

## ⏱️ Time Estimate

| Phase | Duration | Description |
|-------|----------|-------------|
| Research | 30-60 min | Find correct API patterns |
| Helper Functions | 60-90 min | Update setup code |
| Test Updates | 120-180 min | Fix all 115 call sites |
| Testing | 30-60 min | Compile and verify |
| **Total** | **4-6 hours** | Complete migration |

---

## 🚀 Next Steps ✅ COMPLETE

1. **✅ Comprehensive Migration Guide Created**
   - Created `/docs/dev/E2E_TEST_MIGRATION_GUIDE.md`
   - Documented all API changes with before/after examples
   - Provided step-by-step migration plan
   - Estimated 4 hours for complete migration

2. **Ready for Implementation**
   - Phase 1: Update setup helper functions
   - Phase 2: Create zome access helpers
   - Phase 3: Update all 5 test scenarios (115 call sites)
   - Phase 4: Compilation verification

3. **Migration Strategy**
   ```bash
   # Step 1: Start with helper functions
   # Step 2: Fix Scenario 1 as proof-of-concept
   # Step 3: Apply pattern to Scenarios 2-5
   # Step 4: Verify compilation
   cd /srv/luminous-dynamics/mycelix-praxis
   nix develop --command cargo check --test e2e_test
   ```

4. **Success Criteria**
   - ✅ Problem identified and documented
   - ✅ Migration guide created with examples
   - ⏳ Implementation in progress (115 call sites)
   - ⏳ Compilation verification pending
   - ⏳ E2E test execution pending

---

## 📝 Notes

- **Unit Tests**: All 116 unit tests passing - zome implementation is correct
- **Integration Tests**: Similar updates may be needed if using sweettest
- **Web Client**: No changes needed - uses `@holochain/client` which is updated
- **DNA Bundle**: Already built and ready for testing

---

## ✅ Success Criteria

- [ ] All 115 compilation errors resolved
- [ ] `cargo check --test e2e_test` passes
- [ ] `cargo test --test e2e_test --no-run` succeeds
- [ ] Tests execute (even if some fail due to logic issues)
- [ ] Document any Holochain 0.6 API quirks discovered

---

**Status**: Ready for API research and migration
**Blocker**: Need Holochain 0.6 sweettest API documentation
**Priority**: High - Phase 7 completion depends on this

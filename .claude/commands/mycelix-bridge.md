---
description: "Add a new cross-domain or cross-cluster bridge call to Mycelix"
---

# Mycelix Bridge Call Helper

Add a new bridge call (intra-cluster or cross-cluster) with all required changes. The user will specify:
- **Source**: which cluster/zome initiates the call
- **Target**: which cluster/zome receives the call
- **Purpose**: what the call does (e.g., "check emergency status before property lease")

Arguments: $ARGUMENTS

## Context - Read These First

### Bridge dispatch implementation
@crates/mycelix-bridge-common/src/lib.rs

### Commons bridge coordinator (for allowlists + cross-cluster calls)
@mycelix-commons/zomes/commons-bridge/coordinator/src/lib.rs

### Civic bridge coordinator (for allowlists + cross-cluster calls)
@mycelix-civic/zomes/civic-bridge/coordinator/src/lib.rs

### Bridge entry types (shared DHT types)
@crates/mycelix-bridge-entry-types/src/lib.rs

## Determine Call Type

### Intra-Cluster (same DNA)
If both source and target are in the same cluster (both commons or both civic):
- Uses `dispatch_call_checked()` with `CallTargetCell::Local`
- Only needs the target zome in `ALLOWED_ZOMES`

### Cross-Cluster (different DNAs)
If source is in one cluster and target in the other:
- Uses `dispatch_call_cross_cluster()` with `CallTargetCell::OtherRole(role)`
- Needs the target zome in `ALLOWED_{CLUSTER}_ZOMES` on the source side
- Role names: `"commons"` or `"civic"`

## Steps to Execute

### 1. Verify Allowlist Coverage

Check that the target zome is in the appropriate allowlist:
- For intra-cluster: `ALLOWED_ZOMES` in `{source_cluster}-bridge/coordinator/src/lib.rs`
- For cross-cluster: `ALLOWED_CIVIC_ZOMES` or `ALLOWED_COMMONS_ZOMES` in the source bridge

If not present, add it.

### 2. Create Typed Convenience Function

Add a new `#[hdk_extern]` function in the source cluster's bridge coordinator. Follow the existing pattern:

```rust
/// {Description of what this call does}
#[hdk_extern]
pub fn {function_name}(input: {InputType}) -> ExternResult<{OutputType}> {
    // For intra-cluster:
    let dispatch_input = DispatchInput {
        zome: "{target_zome}".to_string(),
        fn_name: "{target_function}".to_string(),
        payload: serde_json::to_string(&input)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    };
    let result = bridge::dispatch_call_checked(&dispatch_input, ALLOWED_ZOMES)?;

    // For cross-cluster:
    let dispatch_input = CrossClusterDispatchInput {
        role: "{target_cluster}".to_string(),
        zome: "{target_zome}".to_string(),
        fn_name: "{target_function}".to_string(),
        payload: serde_json::to_string(&input)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    };
    let result = bridge::dispatch_call_cross_cluster(&dispatch_input, ALLOWED_{TARGET}_ZOMES)?;

    // Parse result
    if result.success {
        // ...
    } else {
        Err(wasm_error!(WasmErrorInner::Guest(
            result.error.unwrap_or_default()
        )))
    }
}
```

### 3. Define Input/Output Types

Add typed structs near the function:

```rust
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct {InputType} {
    // fields...
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct {OutputType} {
    pub success: bool,
    // result fields...
}
```

### 4. Add Serde Roundtrip Test

Add to the `#[cfg(test)]` module in the bridge coordinator:

```rust
#[test]
fn test_{function_name}_input_serde() {
    let input = {InputType} { /* ... */ };
    let json = serde_json::to_string(&input).unwrap();
    let deserialized: {InputType} = serde_json::from_str(&json).unwrap();
    assert_eq!(input.field, deserialized.field);
}
```

### 5. Add SDK TypeScript Client Method

Add to the appropriate client in `mycelix-workspace/sdk-ts/src/integrations/{cluster}/`:

```typescript
/**
 * {Description}
 */
async {functionName}(input: {InputType}): Promise<{OutputType}> {
  return this.callZome('{source_bridge}', '{function_name}', input);
}
```

### 6. Add Integration Test

Add a test in `mycelix-workspace/sdk-ts/tests/` that exercises the new call:

```typescript
describe('{function_name}', () => {
  it('should {expected behavior}', async () => {
    const result = await client.{functionName}({/* input */});
    expect(result.success).toBe(true);
  });
});
```

### 7. Summary Output

After all changes, output:
- Allowlist changes (if any)
- New function signature
- Files modified
- Reminder: run `cargo test` in the bridge crate to verify serde tests
- Reminder: run `just test-sdk-ts` to verify SDK tests

## Existing Cross-Cluster Call Patterns (Reference)

### Commons -> Civic:
- `check_emergency_for_area`: Housing checks emergencies before leasing
- `check_justice_disputes_for_property`: Property checks enforcement blocks

### Civic -> Commons:
- `query_property_for_enforcement`: Justice verifies property exists
- `check_housing_capacity_for_sheltering`: Emergency queries housing availability
- `verify_care_credentials_for_evidence`: Justice validates care qualifications

## Rate Limiting

All dispatch calls are rate-limited to 100 calls/60s per agent. The typed convenience function should call `enforce_rate_limit()` before dispatching.

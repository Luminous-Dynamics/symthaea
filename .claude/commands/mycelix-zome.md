---
description: "Scaffold a new Holochain zome pair (integrity + coordinator) in a Mycelix cluster"
---

# Mycelix Zome Scaffolding

Create a new zome pair following established Mycelix patterns. The user will specify:
- **Cluster**: commons or civic
- **Domain**: e.g., property, justice, care (existing domain to add a zome to, or new domain)
- **Zome name**: e.g., property-valuation, care-referrals

Arguments: $ARGUMENTS

## Context

Read the existing patterns first:

### Cluster workspace Cargo.toml
If the target cluster is commons:
@mycelix-commons/Cargo.toml

If the target cluster is civic:
@mycelix-civic/Cargo.toml

### Bridge allowlist patterns
@crates/mycelix-bridge-common/src/lib.rs

### Reference zome (use as template)
For a coordinator example, read an existing small coordinator like:
@mycelix-commons/zomes/water-capture/coordinator/src/lib.rs

For an integrity example:
@mycelix-commons/zomes/water-capture/integrity/src/lib.rs

## Steps to Execute

### 1. Create Directory Structure

Create these files for the new zome (replace `{domain}-{name}` with the actual zome name):
```
{cluster}/zomes/{domain}-{name}/
  integrity/
    Cargo.toml
    src/
      lib.rs
  coordinator/
    Cargo.toml
    src/
      lib.rs
```

### 2. Integrity Cargo.toml Template

```toml
[package]
name = "{domain}_{name}_integrity"
version.workspace = true
edition.workspace = true

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
hdi.workspace = true
serde.workspace = true
serde_json.workspace = true
holochain_integrity_types.workspace = true
# Add cluster-specific shared types:
# For commons: commons-types.workspace = true
# For civic: civic-types.workspace = true
```

### 3. Integrity lib.rs Template

```rust
use hdi::prelude::*;

/// Entry types for {domain}-{name}
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    // Define entry types here
    // Example:
    // #[entry_type]
    // Record(RecordEntry),
}

/// Link types for {domain}-{name}
#[hdk_link_types]
pub enum LinkTypes {
    // Define link types here
    // Example:
    // RecordToRecord,
    // AgentToRecord,
}

/// Validation callback
#[hdk_extern]
pub fn validate(_op: Op) -> ExternResult<ValidateCallbackResult> {
    // TODO: Implement proper validation
    Ok(ValidateCallbackResult::Valid)
}
```

### 4. Coordinator Cargo.toml Template

```toml
[package]
name = "{domain}_{name}_coordinator"
version.workspace = true
edition.workspace = true

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
hdk.workspace = true
serde.workspace = true
serde_json.workspace = true
{domain}_{name}_integrity = { path = "../integrity" }
# Add cluster-specific shared types:
# For commons: commons-types.workspace = true
# For civic: civic-types.workspace = true

[dev-dependencies]
serde_json.workspace = true
```

### 5. Coordinator lib.rs Template

```rust
use hdk::prelude::*;
use {domain}_{name}_integrity::*;

/// Create a new record
// #[hdk_extern]
// pub fn create_record(input: CreateRecordInput) -> ExternResult<ActionHash> {
//     let action_hash = create_entry(EntryTypes::Record(input.into()))?;
//     Ok(action_hash)
// }

#[cfg(test)]
mod tests {
    use super::*;

    // Add unit tests here
}
```

### 6. Update Workspace Cargo.toml

Add these two lines to the `[workspace] members` array in `{cluster}/Cargo.toml`:
```toml
"zomes/{domain}-{name}/integrity",
"zomes/{domain}-{name}/coordinator",
```

Place them in the correct domain section (alphabetically within the domain group).

### 7. Update Bridge Allowlist

Add the zome name (using underscores: `{domain}_{name}`) to the `ALLOWED_ZOMES` array in:
- `{cluster}/zomes/{cluster}-bridge/coordinator/src/lib.rs`

### 8. Summary Output

After creating all files, output:
- Files created
- Workspace Cargo.toml changes
- Bridge allowlist addition
- Reminder: run `just build-{cluster}` to verify compilation
- Reminder: add mirror types to sweettest if integration tests needed
- Reminder: add SDK client in `mycelix-workspace/sdk-ts/src/integrations/{domain}/`

## CRITICAL Constraints

- Use `getrandom_03 = { package = "getrandom", version = "0.3" }` if randomness needed (NEVER getrandom v0.2 with features=["js"])
- All `opt-level = "z"` in profiles (set at workspace level)
- Use `hdk = "0.6.0"` and `hdi = "0.7.0"` (workspace deps)
- Crate names use underscores: `water_capture_integrity` (not hyphens)
- Directory names use hyphens: `water-capture/integrity/` (not underscores)

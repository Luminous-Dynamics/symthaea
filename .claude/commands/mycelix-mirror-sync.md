---
description: "Check sweettest mirror types are in sync with actual zome types"
---

# Mycelix Mirror Type Synchronization

Sweettest integration tests use "mirror types" - structs that replicate the zome's entry/input types because WASM crate imports cause symbol conflicts. These can drift out of sync when zome types change.

This skill finds mismatches between zome types and their sweettest mirrors.

Arguments: $ARGUMENTS

If args provided, check only that zome/domain. Otherwise, check all sweettests.

## How Mirror Types Work

Zome code (can't import in sweettest):
```rust
// In zome integrity lib.rs
#[hdk_entry_helper]
pub struct PropertyRecord {
    pub owner: AgentPubKey,
    pub address: String,
    pub value: u64,
}
```

Sweettest mirror (must match exactly):
```rust
// In sweettest file
#[derive(Serialize, Deserialize, SerializedBytes)]
pub struct PropertyRecord {
    pub owner: AgentPubKey,
    pub address: String,
    pub value: u64,
}
```

If someone adds a field to the zome type but forgets the mirror -> test will fail at runtime with deserialization errors.

## Step 1: Find All Sweettest Files

!ls /srv/luminous-dynamics/mycelix-workspace/tests/sweettest/tests/*.rs 2>/dev/null | grep -v harness

## Step 2: Extract Mirror Types

For each sweettest file, find struct definitions that look like mirror types (typically annotated with `Serialize, Deserialize, SerializedBytes`).

Read each sweettest file and extract:
- Struct names
- Field names and types
- Any serde attributes (rename, default, skip)

## Step 3: Find Corresponding Zome Types

For each mirror type, find the original in the zome source:
- Search `mycelix-commons/zomes/*/src/lib.rs` and `mycelix-civic/zomes/*/src/lib.rs`
- Search `mycelix-identity/`, `mycelix-core/`, etc.
- Match by struct name

## Step 4: Compare

For each mirror type <-> zome type pair:

1. **Field count**: Same number of fields?
2. **Field names**: All names match?
3. **Field types**: All types match?
4. **Field order**: Same order? (important for some serde formats)
5. **Serde attributes**: Any `#[serde(rename = "...")]` or `#[serde(default)]` that differ?

## Step 5: Report

Output a table:

```
Mirror Type              | Zome Source                          | Status
------------------------|--------------------------------------|--------
PropertyRecord          | commons/property-registry/integrity  | IN SYNC
Create{Entity}Input     | commons/care-matching/coordinator    | DRIFT: missing field 'priority'
JusticeCase             | civic/justice-cases/integrity        | IN SYNC
...
```

For any drifted types, show:
- The mirror type definition (from sweettest)
- The zome type definition (from source)
- A diff highlighting the differences
- Suggested fix (usually: update the mirror to match the zome)

## Step 6: Auto-Fix Option

If the user passes "fix" as an argument, automatically update the mirror types to match the zome types. Be careful to:
- Preserve any sweettest-specific derives (like `SerializedBytes`)
- Keep the type in the same location in the test file
- Maintain any comments

## Common Drift Patterns

1. **New field added to zome type**: Mirror is missing the field -> add it
2. **Field renamed in zome**: Mirror has old name -> rename it
3. **Type changed**: e.g., `String` -> `Option<String>` -> update mirror
4. **Field removed**: Mirror has extra field -> remove it
5. **Serde attribute changed**: e.g., added `#[serde(default)]` -> add to mirror

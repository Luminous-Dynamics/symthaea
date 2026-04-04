# Building Mycelix Extensions

This guide covers how to extend the Mycelix fractal architecture with
new domain modules (Rust zomes) or external frontends (any web framework).

## What is a Mycelix Extension?

Mycelix is a modular system of **cluster DNAs** running on Holochain.
Each cluster (commons, civic, health, etc.) groups related domain zomes
into a single DNA for zero-latency local calls.

An **extension** adds new functionality in one of two ways:

1. **Rust domain module** -- a new zome compiled into an existing or new
   cluster DNA. Has full access to Holochain primitives (DHT, links,
   signals, bridge calls).
2. **External frontend** -- a standalone web app (React, Svelte, Leptos,
   etc.) that connects to the conductor via WebSocket and calls existing
   zome functions through the TypeScript SDK.

Both participate in consciousness gating and data sovereignty.

---

## Creating a Rust Domain Module

### 1. Scaffold the Zome

Each zome needs an integrity crate (entry/link types, validation) and a
coordinator crate (extern functions, logic):

```
mycelix-<cluster>/zomes/<domain>/
  integrity/
    Cargo.toml
    src/lib.rs    # Entry types, link types, validation
  coordinator/
    Cargo.toml
    src/lib.rs    # #[hdk_extern] functions
```

Copy the structure from an existing simple zome like
`mycelix-commons/zomes/food/` as a starting template.

### 2. Define Entry Types (Integrity)

```rust
// integrity/src/lib.rs
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone)]
pub struct GardenPlot {
    pub name: String,
    pub location: String,
    pub area_sqm: f64,
    pub owner_agent: AgentPubKey,
    pub created_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    GardenPlot(GardenPlot),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToPlots,
    PlotToHarvests,
}
```

### 3. Implement Coordinator Externs

```rust
// coordinator/src/lib.rs
use hdk::prelude::*;
use garden_integrity::*;

#[hdk_extern]
pub fn create_plot(input: GardenPlot) -> ExternResult<ActionHash> {
    let hash = create_entry(EntryTypes::GardenPlot(input.clone()))?;
    let my_key = agent_info()?.agent_initial_pubkey;
    create_link(my_key, hash.clone(), LinkTypes::AgentToPlots, ())?;
    Ok(hash)
}

#[hdk_extern]
pub fn get_my_plots(_: ()) -> ExternResult<Vec<(ActionHash, GardenPlot)>> {
    let my_key = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        GetLinksInputBuilder::try_new(my_key, LinkTypes::AgentToPlots)?.build(),
    )?;
    let mut plots = Vec::new();
    for link in links {
        let target = ActionHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Bad link target".into()))
        })?;
        if let Some(record) = get(target.clone(), GetOptions::default())? {
            let plot: GardenPlot = record.entry().to_app_option().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(e.to_string()))
            })?.ok_or(wasm_error!(WasmErrorInner::Guest("Missing entry".into())))?;
            plots.push((target, plot));
        }
    }
    Ok(plots)
}
```

### 4. Add to the Cluster DNA

In the cluster's DNA manifest (`dna.yaml`), add your zomes:

```yaml
coordinator:
  zomes:
    - name: garden
      bundled: "../../target/wasm32-unknown-unknown/release/garden.wasm"
      dependencies:
        - name: garden_integrity
integrity:
  zomes:
    - name: garden_integrity
      bundled: "../../target/wasm32-unknown-unknown/release/garden_integrity.wasm"
```

### 5. Add a Feature Flag (Optional)

For optional domains, gate behind a Cargo feature:

```toml
# Cargo.toml of the cluster workspace
[features]
garden = ["dep:garden-coordinator", "dep:garden-integrity"]
```

### 6. Build

```bash
cargo build --release --target wasm32-unknown-unknown -p garden-coordinator -p garden-integrity
```

---

## Creating an External Frontend

External frontends are standalone web apps that connect to the Holochain
conductor and call zome functions via the TypeScript SDK.

### 1. Create a Manifest

Register your extension with a JSON manifest that the portal loads:

```json
{
  "id": "garden-tracker",
  "name": "Garden Tracker",
  "version": "0.1.0",
  "author": "Your Name",
  "bioName": "Photosynthesis",
  "colorPrimary": "#4CAF50",
  "colorGlow": "#81C784",
  "minTier": "Participant",
  "frontendUrl": "https://garden.example.com",
  "requiredClusters": ["commons"],
  "optionalClusters": ["climate"],
  "description": "Track garden plots, plantings, and harvests."
}
```

Place it in `~/.mycelix/extensions.json` (array of manifests) or
publish to the extension registry DNA.

### 2. Connect via the TypeScript SDK

Install the SDK:

```bash
npm install @mycelix/sdk
```

Connect and call zomes:

```typescript
import { MycelixClient, deriveTier, meetsTier } from "@mycelix/sdk";

const client = new MycelixClient({
  conductor: {
    url: "ws://localhost:8300",
    installedAppId: "mycelix",
  },
});

await client.connect();

// Call a zome function
interface GardenPlot {
  name: string;
  location: string;
  area_sqm: number;
}

const result = await client.callZome<GardenPlot, string>(
  "commons",        // hApp role
  "garden",         // zome name
  "create_plot",    // function name
  { name: "Plot A", location: "Backyard", area_sqm: 25.0 },
);

if (result.ok) {
  console.log("Created plot:", result.data);
}
```

### 3. Mock Mode for Development

Develop without a running conductor:

```typescript
const client = new MycelixClient({
  mock: true,
  mockHandler: async (role, zome, fnName, input) => {
    if (fnName === "get_my_plots") {
      return [
        { name: "Test Plot", location: "Mock Garden", area_sqm: 10 },
      ];
    }
    return null;
  },
});
```

---

## Consciousness Gating

Every governance action in Mycelix is gated by a 4-dimensional
consciousness profile:

| Dimension    | Source                       | Range   |
|-------------|------------------------------|---------|
| Identity    | MFA assurance level           | 0.0-1.0 |
| Reputation  | Cross-hApp aggregated score   | 0.0-1.0 |
| Community   | Peer trust attestations       | 0.0-1.0 |
| Engagement  | Domain-specific participation | 0.0-1.0 |

The combined score maps to a **trust tier**:

| Tier        | Threshold | Capabilities                    |
|-------------|-----------|----------------------------------|
| Observer    | < 0.3     | Read-only access                 |
| Participant | >= 0.3    | Basic proposals                  |
| Citizen     | >= 0.4    | Voting rights                    |
| Steward     | >= 0.6    | Constitutional actions           |
| Guardian    | >= 0.8    | Emergency powers, treasury       |

### Gating in Rust (Coordinator Zome)

Use the bridge to fetch the caller's profile and check their tier:

```rust
use mycelix_bridge_common::consciousness_profile::{ConsciousnessProfile, ConsciousnessTier};

#[hdk_extern]
pub fn create_proposal(input: Proposal) -> ExternResult<ActionHash> {
    // Fetch consciousness profile from the bridge
    let profile: ConsciousnessProfile = call(
        CallTargetCell::Local,
        "commons_bridge",
        "get_consciousness_credential".into(),
        None,
        (),
    )?;

    // Gate: must be at least Participant to create proposals
    let tier = profile.evaluate_tier();
    if tier < ConsciousnessTier::Participant {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient consciousness tier for proposal creation".into()
        )));
    }

    create_entry(EntryTypes::Proposal(input))
}
```

### Gating in TypeScript (Frontend)

```typescript
import { meetsTier } from "@mycelix/sdk";

const profile = await client.getConsciousnessProfile();
if (profile && meetsTier(profile, "Citizen")) {
  // Show voting UI
} else {
  // Show read-only view
}
```

---

## Data Sovereignty

Mycelix enforces data sovereignty through three mechanisms:

### 1. Bridge Allowlists

Each cluster declares which zomes can be called across bridges.
Undeclared zomes are unreachable from other clusters:

```rust
BridgeDeclaration {
    target_cluster: "identity".into(),
    direction: BridgeDirection::Outbound,
    allowed_zomes: vec!["did_registry".into(), "trust_credentials".into()],
    purpose: "DID resolution and trust verification".into(),
}
```

Only `did_registry` and `trust_credentials` can be called -- all other
identity zomes are invisible to this cluster.

### 2. Data Sensitivity Classification

Every entry type is tagged with a sensitivity level:

- **Public** -- visible on the DHT to all participants
- **Internal** -- visible within the cluster only
- **Sensitive** -- requires explicit user consent to share
- **Restricted** -- encrypted, never leaves the source agent

```rust
EntryTypeDeclaration {
    zome: "garden".into(),
    entry_type: "GardenPlot".into(),
    label: "Garden Plot".into(),
    sensitivity: DataSensitivity::Public,
    description: "Community garden plot registration".into(),
}
```

### 3. User Blocking (BRG-011)

Users can block specific data flows via sharing preferences. When a
bridge call involves blocked data, the system returns error `BRG-011`
("User blocked this data flow via sharing preferences") instead of
executing the call.

The data sovereignty dashboard (powered by `ClusterManifest` and
`DataFlow` types) shows users exactly which clusters exchange data
and lets them toggle individual flows on/off.

### Viewing Data Flows in TypeScript

```typescript
import { computeDataFlows, ClusterManifest } from "@mycelix/sdk";

// Given your cluster manifests and installed cluster IDs:
const flows = computeDataFlows(manifests, ["commons", "identity", "governance"]);

for (const flow of flows) {
  console.log(`${flow.sourceId} -> ${flow.targetId}: ${flow.purpose}`);
}
```

---

## Example: Garden Tracker Extension

Putting it all together, here is a minimal Garden Tracker that:

1. Uses an existing `commons` cluster zome (no new Rust code needed
   if the zome already exists)
2. Connects via the TypeScript SDK
3. Respects consciousness gating
4. Declares its data sovereignty requirements in the manifest

```typescript
// garden-tracker/src/main.ts
import { MycelixClient, meetsTier, deriveTier } from "@mycelix/sdk";

async function main() {
  const client = new MycelixClient({
    conductor: { url: "ws://localhost:8304", installedAppId: "mycelix" },
  });
  await client.connect();

  // Check consciousness tier
  const profile = await client.getConsciousnessProfile();
  if (!profile) {
    console.log("No consciousness profile available -- read-only mode");
    return;
  }

  const tier = deriveTier(profile);
  console.log(`Welcome! Your trust tier: ${tier}`);

  // Participants and above can create plots
  if (meetsTier(profile, "Participant")) {
    const result = await client.callZome(
      "commons", "garden", "create_plot",
      { name: "Herb Spiral", location: "Community Park", area_sqm: 8.0 },
    );
    if (result.ok) {
      console.log("Plot created successfully");
    } else {
      console.error("Failed:", result.error);
    }
  }

  // Everyone can view plots
  const plots = await client.callZome(
    "commons", "garden", "get_my_plots", null,
  );
  if (plots.ok) {
    console.log("My plots:", plots.data);
  }

  await client.disconnect();
}

main();
```

Extension manifest (`~/.mycelix/extensions.json`):

```json
[
  {
    "id": "garden-tracker",
    "name": "Garden Tracker",
    "version": "0.1.0",
    "author": "Community Contributor",
    "bioName": "Photosynthesis",
    "colorPrimary": "#4CAF50",
    "colorGlow": "#81C784",
    "minTier": "Observer",
    "frontendUrl": "http://localhost:3000",
    "requiredClusters": ["commons"],
    "optionalClusters": [],
    "description": "Track community garden plots and harvests."
  }
]
```

---

## Quick Reference

| Task | Command / Location |
|------|--------------------|
| Build WASM zomes | `cargo build --release --target wasm32-unknown-unknown` |
| Run tests | `cargo test -p garden-coordinator --lib` |
| Install SDK | `npm install @mycelix/sdk` |
| Extension manifests | `~/.mycelix/extensions.json` |
| Cluster manifests | `crates/mycelix-cluster-manifest/` |
| Bridge types | `crates/mycelix-bridge-common/` |
| Consciousness profile | `crates/mycelix-bridge-common/src/consciousness_profile.rs` |
| Port allocation | `.claude/rules/PORTS.md` |

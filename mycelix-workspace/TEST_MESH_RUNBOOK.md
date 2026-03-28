# Test Mesh Deployment Runbook

How to spin up a local multi-node Mycelix mesh for integration testing.

## Prerequisites

- Holochain CLI (`hc`) and conductor (`holochain`) installed via Nix
- Built WASM zomes for all clusters (`just build-commons`, `just build-civic`, etc.)
- Packed hApp bundles in `happs/`

## Quick Start: 3-Node Local Mesh

### 1. Build all zomes

```bash
cd /srv/luminous-dynamics
just build-commons
just build-civic
just build-identity
just build-governance
just build-finance
```

### 2. Pack the unified hApp

```bash
cd mycelix-workspace
hc app pack happs/mycelix-unified-happ.yaml -o happs/mycelix-unified.happ
```

### 3. Start 3 conductor sandboxes

```bash
# Terminal 1: Node Alice
hc sandbox generate happs/mycelix-unified.happ --run=8888 -d alice

# Terminal 2: Node Bob
hc sandbox generate happs/mycelix-unified.happ --run=8889 -d bob

# Terminal 3: Node Carol
hc sandbox generate happs/mycelix-unified.happ --run=8890 -d carol
```

Each sandbox creates a conductor with the unified hApp installed and an agent key.

### 4. Connect nodes (bootstrap)

Nodes discover each other via the bootstrap server or local mDNS. For local testing:

```bash
# Get Alice's agent key
hc sandbox call --running=8888 list-apps

# Nodes on the same machine will auto-discover via mDNS
# For cross-machine: use --bootstrap-url
```

### 5. Basic Integration Flow

Run this sequence to verify the mesh works end-to-end:

```bash
# Alice creates a DID
hc sandbox call --running=8888 did_registry create_did '{}'

# Alice creates a care circle
hc sandbox call --running=8888 care_circles create_circle '{
  "name": "Test Circle",
  "description": "Integration test circle",
  "location": "Local",
  "max_members": 10,
  "circle_type": "Neighborhood",
  "active": true
}'

# Bob joins the circle (needs circle hash from previous call)
hc sandbox call --running=8889 care_circles join_circle '{
  "circle_hash": "<HASH_FROM_ABOVE>",
  "role": "Member"
}'

# Alice records a care exchange with Bob
hc sandbox call --running=8888 care_circles record_care_exchange '{
  "circle_hash": "<CIRCLE_HASH>",
  "receiver": "<BOB_AGENT_KEY>",
  "hours": 2.0,
  "service_description": "Helped with garden"
}'

# Check TEND balance
hc sandbox call --running=8888 care_circles get_circle_tend_balance '<CIRCLE_HASH>'
```

## What This Tests

| Step | Validates |
|------|-----------|
| DID creation | Agent-centric identity, 1-DID-per-agent enforcement |
| Circle creation | Consciousness gating, entry creation, anchor linking |
| Circle joining | Cross-agent DHT gossip, membership validation |
| Care exchange | TEND mutual credit entry creation, circle linking |
| Balance query | DHT read consistency, aggregation logic |

## What It Does NOT Test

- Byzantine resistance (needs adversarial nodes)
- Consciousness tier enforcement (needs credential issuance pipeline)
- Cross-cluster bridge calls (needs all clusters in the unified hApp)
- Performance under load (needs dedicated benchmark harness)

## Troubleshooting

- **Nodes don't discover each other**: Check firewall rules. Holochain uses UDP for mDNS discovery.
- **Consciousness gate rejects all calls**: Bootstrap credentials may be needed. Check `bootstrap_community_threshold` (currently 5 members).
- **WASM compilation errors**: Run `cargo build --release --target wasm32-unknown-unknown` for each zome.
- **hApp pack fails**: Ensure all DNA YAML files reference correct WASM paths.

## Next Steps

After basic flow works:
1. Test governance flow (create proposal → vote → tally → reflection)
2. Test justice flow (file case → restorative circle → restitution agreement → payment)
3. Test identity flow (DID creation → trust credential → consciousness gating)
4. Adversarial testing (spin up cartel nodes, test detection + slash)

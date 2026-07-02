# Agents Zome API Documentation

## Overview

The Agents Zome provides agent registration and model update submission functionality for the Mycelix federated learning system on Holochain. It serves as the foundation for participant identity and basic model contribution tracking.

## Purpose

- Register agents participating in federated learning
- Track agent capabilities and reputation
- Submit and retrieve model updates by training round
- Link agents and updates via DHT paths for efficient queries

---

## Entry Types

### AgentRegistration

Represents a registered agent in the federated learning network.

```rust
pub struct AgentRegistration {
    pub agent_id: String,       // Unique agent identifier
    pub capabilities: Vec<String>, // List of capabilities (e.g., ["gpu", "high_memory"])
    pub reputation_score: f64,  // Trust score [0.0, 1.0]
}
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `String` | Unique identifier for the agent |
| `capabilities` | `Vec<String>` | List of agent capabilities |
| `reputation_score` | `f64` | Reputation score between 0.0 and 1.0 |

### ModelUpdate

Represents a model update submission for a specific training round.

```rust
pub struct ModelUpdate {
    pub agent_id: String,    // Submitting agent's ID
    pub round_id: u32,       // Training round number
    pub weights: Vec<f32>,   // Model weights/gradients
    pub timestamp: Timestamp // When the update was submitted
}
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `String` | ID of the agent submitting the update |
| `round_id` | `u32` | Training round this update belongs to |
| `weights` | `Vec<f32>` | Model weight updates |
| `timestamp` | `Timestamp` | Submission timestamp |

---

## Link Types

| Link Type | Description |
|-----------|-------------|
| `AgentLink` | Links from "agents/all" anchor to agent registrations |
| `ModelUpdateLink` | Links from round anchor to model updates |

---

## Extern Functions

### register_agent

Registers a new agent in the federated learning network.

**Signature:**
```rust
#[hdk_extern]
pub fn register_agent(registration: AgentRegistration) -> ExternResult<ActionHash>
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `registration` | `AgentRegistration` | Agent registration data |

**Returns:**
- `ExternResult<ActionHash>` - The action hash of the created entry

**Example Usage:**
```javascript
// JavaScript/TypeScript client
const result = await client.callZome({
  cap_secret: null,
  cell_id: cellId,
  zome_name: "agents",
  fn_name: "register_agent",
  payload: {
    agent_id: "node-001",
    capabilities: ["gpu", "high_bandwidth"],
    reputation_score: 0.75
  }
});
```

**Behavior:**
1. Creates an `AgentRegistration` entry
2. Creates a link from the "agents/all" anchor to the entry
3. Uses the `agent_id` as the link tag for filtering

**Error Conditions:**
- Entry creation fails (DHT error)
- Link creation fails (DHT error)

---

### get_all_agents

Retrieves all registered agents from the DHT.

**Signature:**
```rust
#[hdk_extern]
pub fn get_all_agents(_: ()) -> ExternResult<Vec<AgentRegistration>>
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `_` | `()` | Unit type (no input required) |

**Returns:**
- `ExternResult<Vec<AgentRegistration>>` - List of all registered agents

**Example Usage:**
```javascript
const agents = await client.callZome({
  cap_secret: null,
  cell_id: cellId,
  zome_name: "agents",
  fn_name: "get_all_agents",
  payload: null
});
// agents: [{agent_id: "node-001", capabilities: [...], reputation_score: 0.75}, ...]
```

**Behavior:**
1. Gets all links from the "agents/all" anchor
2. Retrieves each linked entry
3. Deserializes and returns all `AgentRegistration` entries

**Error Conditions:**
- Link retrieval fails
- Entry deserialization fails

---

### submit_model_update

Submits a model update for a specific training round.

**Signature:**
```rust
#[hdk_extern]
pub fn submit_model_update(update: ModelUpdate) -> ExternResult<ActionHash>
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `update` | `ModelUpdate` | The model update to submit |

**Returns:**
- `ExternResult<ActionHash>` - The action hash of the created entry

**Example Usage:**
```javascript
const result = await client.callZome({
  cap_secret: null,
  cell_id: cellId,
  zome_name: "agents",
  fn_name: "submit_model_update",
  payload: {
    agent_id: "node-001",
    round_id: 5,
    weights: [0.1, 0.2, 0.3, ...],
    timestamp: Date.now() * 1000 // microseconds
  }
});
```

**Behavior:**
1. Creates a `ModelUpdate` entry
2. Creates a link from the round anchor (`training_rounds/{round_id}`) to the entry
3. Uses the `agent_id` as the link tag

**Error Conditions:**
- Entry creation fails
- Link creation fails

---

### get_round_updates

Retrieves all model updates for a specific training round.

**Signature:**
```rust
#[hdk_extern]
pub fn get_round_updates(round_id: u32) -> ExternResult<Vec<ModelUpdate>>
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `round_id` | `u32` | The training round number |

**Returns:**
- `ExternResult<Vec<ModelUpdate>>` - All model updates for the round

**Example Usage:**
```javascript
const updates = await client.callZome({
  cap_secret: null,
  cell_id: cellId,
  zome_name: "agents",
  fn_name: "get_round_updates",
  payload: 5 // round_id
});
// updates: [{agent_id: "node-001", round_id: 5, weights: [...], timestamp: ...}, ...]
```

**Behavior:**
1. Gets all links from the round anchor (`training_rounds/{round_id}`)
2. Retrieves each linked entry
3. Deserializes and returns all `ModelUpdate` entries

**Error Conditions:**
- Link retrieval fails
- Entry deserialization fails

---

## Security Considerations

### Authorization
- The agents zome does not implement role-based access control
- Any agent can register and submit model updates
- For production, consider integrating with the Federated Learning zome's coordinator role system

### Input Validation
- No explicit input validation is performed on agent_id or capabilities
- The reputation_score should be validated to be within [0.0, 1.0]
- Consider adding length limits on capabilities arrays

### Data Integrity
- Model updates are immutable once committed to the DHT
- Reputation scores stored here are static; use the FL zome's dynamic reputation system for real-time scoring

### Recommendations
1. Validate `agent_id` format before registration
2. Add capability allow-lists to prevent injection attacks
3. Implement rate limiting for model update submissions
4. Consider adding signature verification for updates

---

## Integration Example

```rust
// Rust integration from another zome
use agents::{AgentRegistration, ModelUpdate};

// Register a new agent
let registration = AgentRegistration {
    agent_id: "my-agent-001".to_string(),
    capabilities: vec!["gpu".to_string(), "docker".to_string()],
    reputation_score: 0.8,
};

let action_hash = call(
    CallTargetCell::Local,
    ZomeName::from("agents"),
    FunctionName::from("register_agent"),
    None,
    registration,
)?;

// Submit model update
let update = ModelUpdate {
    agent_id: "my-agent-001".to_string(),
    round_id: 1,
    weights: vec![0.1, 0.2, 0.3],
    timestamp: sys_time()?,
};

call(
    CallTargetCell::Local,
    ZomeName::from("agents"),
    FunctionName::from("submit_model_update"),
    None,
    update,
)?;
```

---

## Related Documentation

- [Federated Learning Zome API](./FL_ZOME_API.md) - Advanced FL with Byzantine resistance
- [Bridge Zome API](./BRIDGE_ZOME_API.md) - Cross-hApp communication

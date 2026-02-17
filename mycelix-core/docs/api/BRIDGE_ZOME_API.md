# Bridge Zome API Documentation

## Overview

The Bridge Zome implements inter-hApp communication for the Mycelix ecosystem on Holochain. It enables cross-hApp reputation sharing, credential verification, and event broadcasting between registered hApps.

## Purpose

- **hApp Registration**: Register hApps to enable cross-hApp communication
- **Reputation Management**: Share and aggregate reputation across multiple hApps
- **Credential Verification**: Request and verify credentials across hApp boundaries
- **Event Broadcasting**: Publish and subscribe to cross-hApp events

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_ID_LENGTH` | `256` | Maximum length for identifiers |
| `MAX_NAME_LENGTH` | `512` | Maximum length for names |
| `MAX_EVENT_PAYLOAD_SIZE` | `65536` | Maximum event payload (64KB) |
| `MAX_CAPABILITIES` | `50` | Maximum capabilities per hApp |
| `MAX_EVENT_TARGETS` | `100` | Maximum event target hApps |
| `MAX_REPUTATION_UPDATES_PER_MINUTE` | `30` | Rate limit for reputation updates |
| `MAX_BROADCASTS_PER_MINUTE` | `100` | Rate limit for event broadcasts |

---

## Entry Types (Integrity Zome)

### HappRegistration

Registers a hApp with the bridge for cross-hApp communication.

```rust
pub struct HappRegistration {
    pub happ_id: String,           // Unique hApp identifier
    pub dna_hash: Option<String>,  // DNA hash for verification
    pub happ_name: String,         // Human-readable name
    pub capabilities: Vec<String>, // Supported capabilities
    pub registered_at: u64,        // Unix timestamp (seconds)
    pub registrant: String,        // Registering agent pubkey
    pub active: bool,              // Whether hApp is active
}
```

**Validation Rules:**
- `happ_id`: Non-empty, max 256 characters
- `happ_name`: Non-empty, max 512 characters
- `registrant`: Non-empty, max 256 characters
- `capabilities`: Max 50 items, each max 256 characters

---

### ReputationRecord

Stores reputation for an agent from a specific hApp.

```rust
pub struct ReputationRecord {
    pub agent: String,                    // Agent identifier
    pub happ_id: String,                  // Source hApp ID
    pub happ_name: String,                // Source hApp name
    pub score: f64,                       // Reputation score [0.0, 1.0]
    pub interactions: u64,                // Positive interactions
    pub negative_interactions: u64,       // Negative interactions
    pub updated_at: u64,                  // Last update timestamp
    pub evidence_hash: Option<String>,    // Proof of reputation calculation
}
```

**Validation Rules:**
- `score`: Finite, between 0.0 and 1.0
- `agent`, `happ_id`, `happ_name`: Non-empty, within length limits
- `evidence_hash`: If present, max 128 characters

---

### CrossHappReputationRecord

Aggregated reputation across multiple hApps.

```rust
pub struct CrossHappReputationRecord {
    pub agent: String,              // Agent identifier
    pub scores_json: String,        // JSON map of hApp ID to score
    pub aggregated_score: f64,      // Weighted aggregate [0.0, 1.0]
    pub total_interactions: u64,    // Total interactions across hApps
    pub happ_count: u32,            // Number of contributing hApps
    pub computed_at: u64,           // Computation timestamp
}
```

---

### BridgeEventRecord

Cross-hApp event for pub/sub notifications.

```rust
pub struct BridgeEventRecord {
    pub event_type: String,      // Event type identifier
    pub source_happ: String,     // Originating hApp
    pub payload: Vec<u8>,        // Serialized payload (max 64KB)
    pub timestamp: u64,          // Event timestamp
    pub targets: Vec<String>,    // Target hApps (empty = all)
    pub priority: u8,            // 0=normal, 1=high, 2=critical
}
```

**Standard Event Types:**
- `reputation_update` - Reputation score changed
- `credential_issued` - New credential issued
- `credential_revoked` - Credential revoked
- `happ_registered` - New hApp registered
- `happ_deregistered` - hApp deactivated
- `trust_alert` - Trust-related warning
- `custom` - Custom event type

---

### CredentialVerificationRequest

Request to verify a credential from another hApp.

```rust
pub struct CredentialVerificationRequest {
    pub request_id: String,        // Unique request ID
    pub credential_hash: String,   // Hash of credential to verify
    pub credential_type: String,   // Type of credential
    pub issuer_happ: String,       // Issuing hApp ID
    pub subject_agent: String,     // Agent the credential belongs to
    pub requester: String,         // Requesting agent
    pub requested_at: u64,         // Request timestamp
    pub status: String,            // "pending", "verified", "failed", "expired"
}
```

---

### CredentialVerificationResponse

Response to a credential verification request.

```rust
pub struct CredentialVerificationResponse {
    pub request_id: String,        // Request this responds to
    pub is_valid: bool,            // Whether credential is valid
    pub issuer: String,            // Verified issuer
    pub credential_type: String,   // Credential type
    pub claims_json: String,       // JSON array of claims
    pub issued_at: u64,            // When credential was issued
    pub expires_at: u64,           // Expiration (0 = no expiry)
    pub verified_at: u64,          // Verification timestamp
    pub verifier: String,          // Verifying agent
    pub proof: Option<Vec<u8>>,    // Cryptographic proof
}
```

---

## Link Types

| Link Type | Description |
|-----------|-------------|
| `HappIdToRegistration` | hApp ID path to registration entry |
| `AgentToReputations` | Agent path to reputation records |
| `HappToReputations` | hApp path to reputation records |
| `EventTypeToEvents` | Event type path to event entries |
| `HappToCredentialRequests` | hApp to pending verification requests |
| `RequestToResponse` | Request ID to verification response |
| `AgentToAggregateReputation` | Agent to aggregated reputation |
| `AllHapps` | Path to all registered hApps |

---

## Extern Functions

### hApp Registration

#### register_happ

Register a hApp with the bridge.

```rust
#[hdk_extern]
pub fn register_happ(input: RegisterHappInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct RegisterHappInput {
    pub name: String,                 // Human-readable hApp name
    pub dna_hash: Option<String>,     // DNA hash for verification
    pub capabilities: Vec<String>,    // Supported capabilities
}
```

**Example:**
```javascript
const result = await client.callZome({
  zome_name: "bridge",
  fn_name: "register_happ",
  payload: {
    name: "Mycelix Mail",
    dna_hash: "uhC0k...",
    capabilities: ["reputation", "credentials", "messaging"]
  }
});
```

**Behavior:**
1. Validates input fields
2. Generates hApp ID from DNA hash or name + agent
3. Creates `HappRegistration` entry
4. Links from hApp ID path and all_happs collection
5. Broadcasts `happ_registered` event

**Returns:** ActionHash of created registration

---

#### deregister_happ

Deactivate a hApp registration.

```rust
#[hdk_extern]
pub fn deregister_happ(happ_id: String) -> ExternResult<ActionHash>
```

**Security:** Only the hApp owner (registrant) can deregister.

**Behavior:**
1. Verifies caller is hApp owner
2. Sets `active = false` on registration
3. Broadcasts `happ_deregistered` event

---

#### get_registered_happs

Get all active hApp registrations.

```rust
#[hdk_extern]
pub fn get_registered_happs(_: ()) -> ExternResult<Vec<HappRegistration>>
```

**Example:**
```javascript
const happs = await client.callZome({
  zome_name: "bridge",
  fn_name: "get_registered_happs",
  payload: null
});
// happs: [{happ_id: "...", happ_name: "Mycelix Mail", ...}, ...]
```

---

#### get_happ_registration

Get a specific hApp registration.

```rust
#[hdk_extern]
pub fn get_happ_registration(happ_id: String) -> ExternResult<Option<HappRegistration>>
```

---

### Reputation Management

#### record_reputation

Record reputation for an agent from a specific hApp.

```rust
#[hdk_extern]
pub fn record_reputation(input: RecordReputationInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct RecordReputationInput {
    pub agent: String,                     // Agent to record reputation for
    pub happ_id: String,                   // Recording hApp ID
    pub happ_name: String,                 // Recording hApp name
    pub score: f64,                        // Reputation score [0.0, 1.0]
    pub interactions: u64,                 // Positive interactions
    pub negative_interactions: u64,        // Negative interactions
    pub evidence_hash: Option<String>,     // Evidence hash
}
```

**Example:**
```javascript
const result = await client.callZome({
  zome_name: "bridge",
  fn_name: "record_reputation",
  payload: {
    agent: "uhCAk...",
    happ_id: "mail_happ",
    happ_name: "Mycelix Mail",
    score: 0.85,
    interactions: 150,
    negative_interactions: 5,
    evidence_hash: null
  }
});
```

**Security:**
- Only hApp owners can record reputation for their hApp
- Rate limited to 30 updates per minute
- Score validated to be within [0.0, 1.0]

**Behavior:**
1. Validates inputs
2. Verifies caller is hApp owner
3. Checks rate limit
4. Creates `ReputationRecord` entry
5. Links from agent and hApp paths
6. Broadcasts `reputation_update` event

---

#### query_reputation

Query reputation for an agent with optional hApp filter.

```rust
#[hdk_extern]
pub fn query_reputation(input: QueryReputationInput) -> ExternResult<Vec<ReputationRecord>>
```

**Input:**
```rust
pub struct QueryReputationInput {
    pub agent: String,           // Agent to query
    pub happ: Option<String>,    // Optional hApp filter
}
```

**Example:**
```javascript
// Get all reputation records for an agent
const records = await client.callZome({
  zome_name: "bridge",
  fn_name: "query_reputation",
  payload: {
    agent: "uhCAk...",
    happ: null
  }
});

// Get reputation from a specific hApp
const mailRep = await client.callZome({
  zome_name: "bridge",
  fn_name: "query_reputation",
  payload: {
    agent: "uhCAk...",
    happ: "mail_happ"
  }
});
```

**Returns:** Records sorted by `updated_at` descending (most recent first)

---

#### aggregate_cross_happ_reputation

Compute weighted aggregate reputation across all hApps.

```rust
#[hdk_extern]
pub fn aggregate_cross_happ_reputation(agent: String) -> ExternResult<CrossHappReputation>
```

**Output:**
```rust
// From mycelix_sdk::bridge
pub struct CrossHappReputation {
    pub agent: String,
    pub scores: Vec<HappReputationScore>,
    pub aggregate: f64,               // Weighted average score
    // ... methods
}

pub struct HappReputationScore {
    pub happ_id: String,
    pub happ_name: String,
    pub score: f64,
    pub interactions: u64,
    pub last_updated: u64,
}
```

**Calculation:**
- Weighted average based on interaction counts
- Formula: `sum(score_i * interactions_i) / sum(interactions_i)`

**Example:**
```javascript
const reputation = await client.callZome({
  zome_name: "bridge",
  fn_name: "aggregate_cross_happ_reputation",
  payload: "uhCAk..."
});
// reputation: { agent: "...", aggregate: 0.867, ... }
```

---

#### is_agent_trustworthy

Check if agent meets a trust threshold.

```rust
#[hdk_extern]
pub fn is_agent_trustworthy(input: TrustCheckInput) -> ExternResult<bool>
```

**Input:**
```rust
pub struct TrustCheckInput {
    pub agent: String,
    pub threshold: f64,    // Trust threshold [0.0, 1.0]
}
```

**Example:**
```javascript
const isTrusted = await client.callZome({
  zome_name: "bridge",
  fn_name: "is_agent_trustworthy",
  payload: {
    agent: "uhCAk...",
    threshold: 0.7
  }
});
// isTrusted: true
```

---

### Credential Verification

#### request_credential_verification

Request verification of a credential from another hApp.

```rust
#[hdk_extern]
pub fn request_credential_verification(input: CredentialVerifyInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct CredentialVerifyInput {
    pub credential_hash: String,   // Hash of credential to verify
    pub credential_type: String,   // Type of credential
    pub issuer_happ: String,       // Issuing hApp ID
    pub subject_agent: String,     // Agent the credential belongs to
}
```

**Example:**
```javascript
const requestHash = await client.callZome({
  zome_name: "bridge",
  fn_name: "request_credential_verification",
  payload: {
    credential_hash: "abc123...",
    credential_type: "training_completion",
    issuer_happ: "learning_platform",
    subject_agent: "uhCAk..."
  }
});
```

**Behavior:**
1. Validates input fields
2. Generates unique request ID
3. Creates `CredentialVerificationRequest` with status "pending"
4. Links from issuer hApp path

---

#### verify_credential

Respond to a verification request (issuing hApp).

```rust
#[hdk_extern]
pub fn verify_credential(input: VerifyCredentialInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct VerifyCredentialInput {
    pub request_id: String,        // Request to respond to
    pub is_valid: bool,            // Whether credential is valid
    pub issuer: String,            // Issuer identifier
    pub claims: Vec<String>,       // Claims in the credential
    pub issued_at: u64,            // When credential was issued
    pub expires_at: u64,           // When it expires (0 = no expiry)
    pub proof: Option<Vec<u8>>,    // Cryptographic proof
}
```

**Example:**
```javascript
const responseHash = await client.callZome({
  zome_name: "bridge",
  fn_name: "verify_credential",
  payload: {
    request_id: "req_abc123",
    is_valid: true,
    issuer: "learning_platform",
    claims: ["completed_course_ml101", "score_95"],
    issued_at: 1704067200,
    expires_at: 0,
    proof: null
  }
});
```

---

#### get_verification_requests

Get pending verification requests for a hApp.

```rust
#[hdk_extern]
pub fn get_verification_requests(happ_id: String) -> ExternResult<Vec<CredentialVerificationRequest>>
```

**Returns:** Only requests with status "pending"

---

#### get_verification_response

Get the response for a verification request.

```rust
#[hdk_extern]
pub fn get_verification_response(request_id: String) -> ExternResult<Option<CredentialVerificationResponse>>
```

---

### Event Broadcasting

#### broadcast_event

Broadcast an event to other hApps.

```rust
#[hdk_extern]
pub fn broadcast_event(input: BroadcastEventInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct BroadcastEventInput {
    pub event_type: String,        // Event type identifier
    pub payload: Vec<u8>,          // Serialized payload
    pub targets: Vec<String>,      // Target hApps (empty = all)
    pub priority: u8,              // 0=normal, 1=high, 2=critical
}
```

**Example:**
```javascript
const eventHash = await client.callZome({
  zome_name: "bridge",
  fn_name: "broadcast_event",
  payload: {
    event_type: "trust_alert",
    payload: new TextEncoder().encode(JSON.stringify({
      agent: "uhCAk...",
      reason: "unusual_activity",
      severity: "medium"
    })),
    targets: [],  // broadcast to all
    priority: 1   // high priority
  }
});
```

**Security:**
- Rate limited to 100 broadcasts per minute
- Payload size limited to 64KB
- Priority capped at 2

---

#### get_events

Query events by type and time range.

```rust
#[hdk_extern]
pub fn get_events(input: GetEventsInput) -> ExternResult<Vec<BridgeEventRecord>>
```

**Input:**
```rust
pub struct GetEventsInput {
    pub event_type: String,    // Event type to query
    pub since: u64,            // Only events after this timestamp
    pub limit: Option<usize>,  // Maximum events to return
}
```

**Example:**
```javascript
const events = await client.callZome({
  zome_name: "bridge",
  fn_name: "get_events",
  payload: {
    event_type: "reputation_update",
    since: Math.floor(Date.now() / 1000) - 3600, // Last hour
    limit: 50
  }
});
```

**Returns:** Events sorted by timestamp ascending

---

## Security Considerations

### Authorization

- **hApp Ownership**: Only the registrant can:
  - Deregister their hApp
  - Record reputation for their hApp
- Ownership verified via `require_happ_owner()` check

### Rate Limiting

| Action | Limit |
|--------|-------|
| Reputation updates | 30 per minute per agent |
| Event broadcasts | 100 per minute per agent |

Rate limits are enforced per agent using link-based tracking.

### Input Validation

- All string fields validated for:
  - Non-empty (unless explicitly optional)
  - Maximum length limits
- Scores validated for:
  - Finite values
  - Range [0.0, 1.0]
- Payloads validated for:
  - Maximum size (64KB)

### Credential Verification

- Requests are linked to issuing hApp
- Responses include optional cryptographic proofs
- Status tracking prevents duplicate responses

### Cross-hApp Trust

- Reputation aggregation uses weighted averages
- Evidence hashes provide audit trails
- Event targets allow selective broadcasting

---

## Error Conditions

| Error | Cause |
|-------|-------|
| `"{field} cannot be empty"` | Empty required field |
| `"{field} exceeds maximum length"` | Field exceeds length limit |
| `"Score must be between 0.0 and 1.0"` | Invalid score value |
| `"Only the hApp owner can perform this action"` | Unauthorized hApp modification |
| `"Rate limit exceeded"` | Too many requests per minute |
| `"Event payload exceeds maximum size"` | Payload > 64KB |
| `"hApp not found"` | Referenced hApp doesn't exist |

---

## Integration Example

### Cross-hApp Reputation Check

```javascript
// Service A wants to check if a user from Service B is trustworthy

async function checkUserTrust(agentPubKey, threshold = 0.7) {
  // 1. Get cross-hApp reputation
  const reputation = await client.callZome({
    zome_name: "bridge",
    fn_name: "aggregate_cross_happ_reputation",
    payload: agentPubKey
  });

  console.log(`Agent ${agentPubKey} has aggregate score: ${reputation.aggregate}`);
  console.log(`Based on ${reputation.happ_count} hApps`);

  // 2. Check against threshold
  const isTrusted = await client.callZome({
    zome_name: "bridge",
    fn_name: "is_agent_trustworthy",
    payload: {
      agent: agentPubKey,
      threshold: threshold
    }
  });

  return isTrusted;
}
```

### Credential Verification Flow

```javascript
// Verifier hApp: Request credential verification
async function requestCredentialCheck(credentialHash, issuerHapp, subjectAgent) {
  // 1. Submit verification request
  const requestHash = await client.callZome({
    zome_name: "bridge",
    fn_name: "request_credential_verification",
    payload: {
      credential_hash: credentialHash,
      credential_type: "certification",
      issuer_happ: issuerHapp,
      subject_agent: subjectAgent
    }
  });

  // 2. Poll for response (in real app, use signals)
  let response = null;
  while (!response) {
    await sleep(1000);
    response = await client.callZome({
      zome_name: "bridge",
      fn_name: "get_verification_response",
      payload: requestHash
    });
  }

  return response;
}

// Issuer hApp: Monitor and respond to verification requests
async function handleVerificationRequests(myHappId) {
  const requests = await client.callZome({
    zome_name: "bridge",
    fn_name: "get_verification_requests",
    payload: myHappId
  });

  for (const request of requests) {
    // Verify the credential locally
    const isValid = await verifyCredentialLocally(request.credential_hash);

    // Respond
    await client.callZome({
      zome_name: "bridge",
      fn_name: "verify_credential",
      payload: {
        request_id: request.request_id,
        is_valid: isValid,
        issuer: myHappId,
        claims: isValid ? ["verified_holder"] : [],
        issued_at: Date.now() / 1000,
        expires_at: 0,
        proof: null
      }
    });
  }
}
```

### Event-Driven Architecture

```javascript
// Publisher: Broadcast events when reputation changes
async function notifyReputationChange(agent, newScore, happ_id) {
  await client.callZome({
    zome_name: "bridge",
    fn_name: "broadcast_event",
    payload: {
      event_type: "reputation_update",
      payload: new TextEncoder().encode(JSON.stringify({
        agent,
        score: newScore,
        happ_id,
        timestamp: Date.now()
      })),
      targets: [],  // broadcast to all
      priority: 0   // normal priority
    }
  });
}

// Subscriber: Poll for relevant events
async function pollReputationEvents(sinceTimestamp) {
  const events = await client.callZome({
    zome_name: "bridge",
    fn_name: "get_events",
    payload: {
      event_type: "reputation_update",
      since: sinceTimestamp,
      limit: 100
    }
  });

  for (const event of events) {
    const data = JSON.parse(new TextDecoder().decode(new Uint8Array(event.payload)));
    console.log(`Reputation update: ${data.agent} now has score ${data.score}`);
  }

  return events.length > 0
    ? events[events.length - 1].timestamp
    : sinceTimestamp;
}
```

---

## Related Documentation

- [Agents Zome API](./AGENTS_ZOME_API.md) - Basic agent registration
- [Federated Learning Zome API](./FL_ZOME_API.md) - Byzantine-resistant FL
- [Mycelix SDK Bridge Module](../sdk/BRIDGE_MODULE.md) - Client-side utilities

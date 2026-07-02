# Praxis Examples

This directory contains example data demonstrating Praxis's key features.

## Structure

```
examples/
├── credentials/       # W3C Verifiable Credentials
├── fl-rounds/         # Federated Learning rounds and updates
├── courses/           # Course structures
├── notebooks/         # Jupyter notebooks (future)
└── README.md          # This file
```

## Examples

### Verifiable Credentials

**Valid credential:**
- **File**: `credentials/valid-achievement.json`
- **Description**: A valid EduAchievementCredential for completing "Rust Fundamentals"
- **Key features**:
  - W3C VC compliant
  - Linked to model ID for provenance
  - Includes skills array
  - Ed25519 signature
  - Revocation status

**Invalid credential (expired):**
- **File**: `credentials/invalid-expired.json`
- **Description**: An expired credential (for testing verification logic)
- **Use case**: Test expiration date handling

### FL Rounds

**Round in DISCOVER state:**
- **File**: `fl-rounds/round-001-discover.json`
- **Description**: A newly announced FL round accepting participants
- **Parameters**:
  - Model: Spanish learning model v1
  - Aggregation: Trimmed mean (10% trim)
  - Clipping: L2 norm ≤ 1.0
  - Min participants: 10
  - Max participants: 100

**Round in COMPLETED state:**
- **File**: `fl-rounds/round-001-completed.json`
- **Description**: Same round after successful completion
- **Results**:
  - 37 participants
  - Accuracy improved from 0.78 → 0.82
  - Median validation loss: 0.42
  - New model hash published

**Individual FL update:**
- **File**: `fl-rounds/update-001.json`
- **Description**: A single participant's gradient update submission
- **Key features**:
  - Gradient commitment (BLAKE3 hash)
  - Clipped L2 norm: 0.87
  - Local validation loss: 0.38
  - 156 training samples
  - Metadata: training time, epochs, batch size

**Note**: The actual gradient is NOT in this file. Only the commitment (hash) is published on-chain. The gradient itself is shared peer-to-peer with the coordinator during aggregation.

### Courses

**Spanish for Beginners:**
- **File**: `courses/spanish-beginner.json`
- **Description**: A complete beginner Spanish course structure
- **Features**:
  - 9 modules, ~50 hours
  - Federated learning enabled
  - Personalized AI tutoring
  - 500+ vocabulary words
  - Syllabus with learning outcomes

## Using Examples

### Validation

Validate VC schemas:
```bash
# Using ajv-cli (npm install -g ajv-cli)
ajv validate -s schemas/vc/EduAchievementCredential.schema.json -d examples/credentials/valid-achievement.json
```

### Testing

Load examples in tests:
```rust
use std::fs;

#[test]
fn test_valid_credential() {
    let json = fs::read_to_string("examples/credentials/valid-achievement.json").unwrap();
    let cred: VerifiableCredential = serde_json::from_str(&json).unwrap();
    assert_eq!(cred.credential_type, vec!["VerifiableCredential", "EduAchievementCredential"]);
}
```

### Simulation

Use FL round data to simulate aggregation:
```rust
use praxis_agg::trimmed_mean;

// Load round config
let round = serde_json::from_str(include_str!("../examples/fl-rounds/round-001-discover.json"))?;

// Simulate multiple updates (in real system, these come from participants)
let gradients: Vec<Vec<f32>> = generate_mock_gradients(37);  // 37 participants

// Aggregate using trimmed mean
let result = trimmed_mean(&gradients, &round.aggregation_config)?;
```

## Contributing Examples

When adding new examples:

1. **Name clearly**: `{category}/{descriptive-name}.json`
2. **Include comments**: Use `_comment` or `_note` fields for explanations
3. **Validate**: Run against schemas before committing
4. **Document**: Add entry to this README
5. **Keep realistic**: Use plausible values (not "test123")

### Example Template

```json
{
  "id": "unique-identifier",
  "type": "ExampleType",
  "field1": "value1",
  "field2": 42,
  "_comment": "This field demonstrates X feature",
  "metadata": {
    "created": "2025-11-15T00:00:00Z",
    "author": "Example Author"
  }
}
```

## Future Examples

Planned additions:
- [ ] DAO proposals (governance examples)
- [ ] Learner progress snapshots
- [ ] Jupyter notebooks for FL simulation
- [ ] Visualization scripts
- [ ] Attack scenarios (poisoning, sybil)
- [ ] Multi-round FL lifecycle

## Questions?

See [FAQ](../docs/faq.md) or open a [Discussion](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions).

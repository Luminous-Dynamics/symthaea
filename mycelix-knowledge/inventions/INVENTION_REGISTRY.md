# Luminous Dynamics Invention Registry

Prior art records for registration on the Mycelix Knowledge DHT via the
`invention` coordinator zome's `register_invention` function.

This document serves two purposes:

1. **Dogfooding** -- validates the sovereign licensing engine end-to-end.
2. **Prior art provenance** -- creates tamper-evident, DHT-timestamped records
   of each invention with Blake3 witness commitments.

All payloads conform to `RegisterInventionInput` (coordinator) and are validated
against `InventionClaim` (integrity).  See:

- Integrity types: `mycelix-knowledge/zomes/invention/integrity/src/lib.rs`
- Coordinator API: `mycelix-knowledge/zomes/invention/coordinator/src/lib.rs`

Inventor DID: `did:mycelix:tristan-stoltz`

---

## Witness Commitment Protocol

Each invention carries a 32-byte Blake3 witness commitment computed as:

```
witness_commitment = Blake3(title || description || inventor_did || created_at_iso8601)
```

The `||` operator is byte-level concatenation (UTF-8 encoded strings, no
separator).  The `created_at` field uses the ISO 8601 representation of the
Holochain `Timestamp` (microseconds since Unix epoch) for reproducibility.

To verify a commitment after retrieval from the DHT:

```rust
use blake3;

let mut hasher = blake3::Hasher::new();
hasher.update(claim.title.as_bytes());
hasher.update(claim.description.as_bytes());
hasher.update(claim.inventor_did.as_bytes());
hasher.update(claim.created_at.to_string().as_bytes());
let expected = hasher.finalize();
assert_eq!(expected.as_bytes(), &claim.witness_commitment[..]);
```

In Python (for scripting the registration):

```python
import blake3

def compute_witness(title: str, description: str, inventor_did: str, created_at: str) -> bytes:
    h = blake3.blake3()
    h.update(title.encode("utf-8"))
    h.update(description.encode("utf-8"))
    h.update(inventor_did.encode("utf-8"))
    h.update(created_at.encode("utf-8"))
    return h.digest()
```

---

## Invention 1: HDC-LTC Unified Neuron (P-001)

| Field | Value |
|-------|-------|
| ID | `ld-inv-001-hdc-ltc-neuron` |
| Patent | P-001 |
| Grace Expiry | 2027-02-05 |
| Domain | `machine-learning` |
| Status | `Published` |
| Classification | E3 / N2 / M3 |

**Title**: Hyperdimensional Liquid Time-Constant Neural Network with Closed-Form Temporal Evolution

**Description**: A neural network architecture combining 16,384-dimensional hypervectors with liquid time-constant dynamics. Neuron state is a continuous hypervector evolving via closed-form exponential interpolation (O(1) regardless of time step), replacing matrix operations with element-wise HDC binding. Achieves 234Hz cognitive cycle rate.

**Evidence**:
- `symthaea-core/src/hdc/hdc_ltc_unified.rs` (SourceCode)
- `symthaea-hdc-ltc/` standalone crate (SourceCode)
- Psych-bench validation: 27 domains, grand mean z=+1.32 (TestResults)

**Prior Art Refs**: None (foundational)

**License**: Dual (AGPL-3.0 open + Commercial), derivatives allowed, attribution required, 3% royalty on commercial use.

### `register_invention` Payload

```json
{
  "id": "ld-inv-001-hdc-ltc-neuron",
  "title": "Hyperdimensional Liquid Time-Constant Neural Network with Closed-Form Temporal Evolution",
  "description": "A neural network architecture combining 16,384-dimensional hypervectors with liquid time-constant dynamics. Neuron state is a continuous hypervector evolving via closed-form exponential interpolation (O(1) regardless of time step), replacing matrix operations with element-wise HDC binding. Achieves 234Hz cognitive cycle rate.",
  "inventor_did": "did:mycelix:tristan-stoltz",
  "co_inventors": [],
  "prior_art_refs": [],
  "evidence_hashes": [
    {
      "hash": "<Blake3 hash of symthaea-core/src/hdc/hdc_ltc_unified.rs>",
      "description": "Core unified HDC-LTC neuron implementation (~840 LOC)",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of symthaea-hdc-ltc/ crate tarball>",
      "description": "Standalone HDC-LTC crate (published)",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of psych-bench report JSON>",
      "description": "Psych-bench validation: 27 domains, grand mean z=+1.32 over human baselines",
      "evidence_type": "TestResults"
    }
  ],
  "license_terms": {
    "license_type": {
      "DualLicense": {
        "open": "AGPL3OrLater",
        "commercial": "Custom"
      }
    },
    "attribution_required": true,
    "derivative_allowed": true,
    "commercial_allowed": true,
    "royalty_percentage": 3.0,
    "royalty_receiver_did": "did:mycelix:tristan-stoltz",
    "custom_terms": "Commercial license available from Luminous Dynamics. Open-source use under AGPL-3.0-or-later requires attribution and source disclosure."
  },
  "domain": "machine-learning",
  "classification": {
    "empirical": 3,
    "normative": 2,
    "materiality": 3
  },
  "publish": true
}
```

---

## Invention 2: PoGQ v4.1 Byzantine Defense (P-005)

| Field | Value |
|-------|-------|
| ID | `ld-inv-002-pogq-v41` |
| Patent | P-005 |
| Grace Expiry | 2027-02-05 |
| Domain | `federated-learning` |
| Status | `Published` |
| Classification | E3 / N2 / M3 |

**Title**: Proof of Gradient Quality with Mondrian Conformal Detection and Reputation-Weighted Byzantine Fault Tolerance

**Description**: Byzantine fault tolerant federated learning achieving 45% tolerance (exceeds classical 33% limit) via reputation-weighted validation, Mondrian conformal per-class FPR guarantees, adaptive hybrid scoring, temporal EMA with hysteresis quarantine, and ZK-STARK verifiable state transitions.

**Evidence**:
- `mycelix-core/libs/mycelix-fl/src/pogq/` (SourceCode)
- `mycelix-core/0TML/src/defenses/pogq_v4_enhanced.py` (SourceCode)
- BFT validation: 45% Byzantine tolerance measured across multiple attack types (TestResults)

**Prior Art Refs**: `ld-inv-001-hdc-ltc-neuron` (uses HDC for gradient encoding)

**License**: Dual (AGPL-3.0 open + Commercial), derivatives allowed, attribution required, 3% royalty on commercial use.

### `register_invention` Payload

```json
{
  "id": "ld-inv-002-pogq-v41",
  "title": "Proof of Gradient Quality with Mondrian Conformal Detection and Reputation-Weighted Byzantine Fault Tolerance",
  "description": "Byzantine fault tolerant federated learning achieving 45% tolerance (exceeds classical 33% limit) via reputation-weighted validation, Mondrian conformal per-class FPR guarantees, adaptive hybrid scoring, temporal EMA with hysteresis quarantine, and ZK-STARK verifiable state transitions.",
  "inventor_did": "did:mycelix:tristan-stoltz",
  "co_inventors": [],
  "prior_art_refs": ["ld-inv-001-hdc-ltc-neuron"],
  "evidence_hashes": [
    {
      "hash": "<Blake3 hash of mycelix-core/libs/mycelix-fl/src/pogq/ directory>",
      "description": "Rust PoGQ implementation (core library)",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of pogq_v4_enhanced.py>",
      "description": "Python PoGQ v4.1 enhanced defense with Mondrian conformal detection",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of BFT sweep results>",
      "description": "45% BFT tolerance validation across sign-flip, Gaussian, label-flip, inner-product attacks",
      "evidence_type": "TestResults"
    }
  ],
  "license_terms": {
    "license_type": {
      "DualLicense": {
        "open": "AGPL3OrLater",
        "commercial": "Custom"
      }
    },
    "attribution_required": true,
    "derivative_allowed": true,
    "commercial_allowed": true,
    "royalty_percentage": 3.0,
    "royalty_receiver_did": "did:mycelix:tristan-stoltz",
    "custom_terms": "Commercial license available from Luminous Dynamics. Open-source use under AGPL-3.0-or-later requires attribution and source disclosure."
  },
  "domain": "federated-learning",
  "classification": {
    "empirical": 3,
    "normative": 2,
    "materiality": 3
  },
  "publish": true
}
```

---

## Invention 3: HDC-Native Homomorphic Encryption (P-019)

| Field | Value |
|-------|-------|
| ID | `ld-inv-003-hdc-fhe` |
| Patent | P-019 |
| Grace Expiry | 2027-03-17 |
| Domain | `cryptography` |
| Status | `Published` |
| Classification | E3 / N1 / M2 |

**Title**: Information-Theoretically Secure Homomorphic Computation on Hyperdimensional Binary Vectors

**Description**: Homomorphic encryption on 16,384-bit binary hypervectors via XOR one-time pad. Achieves perfect secrecy (Shannon 1949) with ~1.0x plaintext speed (vs 10,000x for lattice FHE). Supports homomorphic binding, distance-preserving similarity, approximate bundling, threshold secret splitting, and collective wisdom aggregation.

**Evidence**:
- `symthaea-hdc-crypto/` standalone crate (SourceCode)
- `symthaea-core/src/hdc/hdc_fhe.rs` (SourceCode)
- Security benchmark: zero accuracy loss, perfect secrecy verified (TestResults)
- Psych-bench Security domain: 6 benchmarks, z=+2.00 (TestResults)

**Prior Art Refs**: `ld-inv-001-hdc-ltc-neuron` (shared HDC foundation)

**License**: Dual (AGPL-3.0 open + Commercial), derivatives allowed, attribution required, 2% royalty on commercial use.

### `register_invention` Payload

```json
{
  "id": "ld-inv-003-hdc-fhe",
  "title": "Information-Theoretically Secure Homomorphic Computation on Hyperdimensional Binary Vectors",
  "description": "Homomorphic encryption on 16,384-bit binary hypervectors via XOR one-time pad. Achieves perfect secrecy (Shannon 1949) with ~1.0x plaintext speed (vs 10,000x for lattice FHE). Supports homomorphic binding, distance-preserving similarity, approximate bundling, threshold secret splitting, and collective wisdom aggregation.",
  "inventor_did": "did:mycelix:tristan-stoltz",
  "co_inventors": [],
  "prior_art_refs": ["ld-inv-001-hdc-ltc-neuron"],
  "evidence_hashes": [
    {
      "hash": "<Blake3 hash of symthaea-hdc-crypto/ crate tarball>",
      "description": "Standalone HDC-FHE crate with XOR OTP, threshold splitting, collective aggregation",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of symthaea-core/src/hdc/hdc_fhe.rs>",
      "description": "Core FHE implementation integrated with HDC engine",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of security benchmark results>",
      "description": "Zero accuracy loss, ~1.0x plaintext speed, perfect secrecy (ciphertext similarity ~0.500)",
      "evidence_type": "TestResults"
    }
  ],
  "license_terms": {
    "license_type": {
      "DualLicense": {
        "open": "AGPL3OrLater",
        "commercial": "Custom"
      }
    },
    "attribution_required": true,
    "derivative_allowed": true,
    "commercial_allowed": true,
    "royalty_percentage": 2.0,
    "royalty_receiver_did": "did:mycelix:tristan-stoltz",
    "custom_terms": "Commercial license available from Luminous Dynamics. Open-source use under AGPL-3.0-or-later requires attribution and source disclosure."
  },
  "domain": "cryptography",
  "classification": {
    "empirical": 3,
    "normative": 1,
    "materiality": 2
  },
  "publish": true
}
```

---

## Invention 4: Consciousness-Gated Authorization (P-011)

| Field | Value |
|-------|-------|
| ID | `ld-inv-004-consciousness-gating` |
| Patent | P-011 |
| Grace Expiry | 2027-02-21 |
| Domain | `governance` |
| Status | `Published` |
| Classification | E2 / N2 / M3 |

**Title**: Multi-Dimensional Continuous Authorization via Consciousness Profile Scoring

**Description**: Replaces discrete RBAC with continuous 4D scoring (identity, reputation, community, engagement). Vote weight computed via sigmoid with configurable temperature. Enables Sybil resistance without KYC, graduated privilege without hard thresholds.

**Evidence**:
- `crates/mycelix-bridge-common/src/consciousness_profile.rs` (SourceCode)
- `crates/mycelix-bridge-common/src/validation.rs` (SourceCode)
- 450+ tests in bridge-common (TestResults)

**Prior Art Refs**: None (foundational for governance)

**License**: Dual (AGPL-3.0 open + Commercial), derivatives allowed, attribution required, 2% royalty on commercial use.

### `register_invention` Payload

```json
{
  "id": "ld-inv-004-consciousness-gating",
  "title": "Multi-Dimensional Continuous Authorization via Consciousness Profile Scoring",
  "description": "Replaces discrete RBAC with continuous 4D scoring (identity, reputation, community, engagement). Vote weight computed via sigmoid with configurable temperature. Enables Sybil resistance without KYC, graduated privilege without hard thresholds.",
  "inventor_did": "did:mycelix:tristan-stoltz",
  "co_inventors": [],
  "prior_art_refs": [],
  "evidence_hashes": [
    {
      "hash": "<Blake3 hash of consciousness_profile.rs>",
      "description": "ConsciousnessProfile 4D scoring with sigmoid vote weight, 5 tiers (Observer-Guardian)",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of validation.rs>",
      "description": "Validation logic for consciousness-gated authorization",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of bridge-common test results>",
      "description": "450+ tests covering consciousness gating, cross-cluster routing, sub-passport recovery",
      "evidence_type": "TestResults"
    }
  ],
  "license_terms": {
    "license_type": {
      "DualLicense": {
        "open": "AGPL3OrLater",
        "commercial": "Custom"
      }
    },
    "attribution_required": true,
    "derivative_allowed": true,
    "commercial_allowed": true,
    "royalty_percentage": 2.0,
    "royalty_receiver_did": "did:mycelix:tristan-stoltz",
    "custom_terms": "Commercial license available from Luminous Dynamics. Open-source use under AGPL-3.0-or-later requires attribution and source disclosure."
  },
  "domain": "governance",
  "classification": {
    "empirical": 2,
    "normative": 2,
    "materiality": 3
  },
  "publish": true
}
```

---

## Invention 5: Sovereign Licensing Engine

| Field | Value |
|-------|-------|
| ID | `ld-inv-005-sovereign-licensing` |
| Patent | None yet (candidate P-020) |
| Grace Expiry | N/A |
| Domain | `intellectual-property` |
| Status | `Published` |
| Classification | E2 / N1 / M3 |

**Title**: Decentralized Prior Art Registry with ZK-STARK Attestation, Merkle Timestamp Chaining, and Automated Royalty Enforcement

**Description**: Combines InventionClaim zome (prior art registry with E/N/M classification), license enforcement in cross-cluster routing, royalty auto-deduction on derivative works, Blake3 merkle timestamp chain with external anchoring (file/git/multi), and ZK-STARK usage attestations.

**Evidence**:
- `mycelix-knowledge/zomes/invention/` -- integrity + coordinator zomes (SourceCode)
- `crates/mycelix-bridge-common/src/license_enforcement.rs` (SourceCode)
- `crates/mycelix-bridge-common/src/merkle_timestamp.rs` (SourceCode)
- `crates/mycelix-bridge-common/src/timestamp_anchor.rs` (SourceCode)
- `mycelix-attribution/` -- reciprocity + usage tracking zomes (SourceCode)

**Prior Art Refs**: `ld-inv-004-consciousness-gating` (uses consciousness gating for license enforcement)

**License**: Dual (AGPL-3.0 open + Commercial), derivatives allowed, attribution required, 5% royalty on commercial use.

### `register_invention` Payload

```json
{
  "id": "ld-inv-005-sovereign-licensing",
  "title": "Decentralized Prior Art Registry with ZK-STARK Attestation, Merkle Timestamp Chaining, and Automated Royalty Enforcement",
  "description": "Combines InventionClaim zome (prior art registry with E/N/M classification), license enforcement in cross-cluster routing, royalty auto-deduction on derivative works, Blake3 merkle timestamp chain with external anchoring (file/git/multi), and ZK-STARK usage attestations.",
  "inventor_did": "did:mycelix:tristan-stoltz",
  "co_inventors": [],
  "prior_art_refs": ["ld-inv-004-consciousness-gating"],
  "evidence_hashes": [
    {
      "hash": "<Blake3 hash of mycelix-knowledge/zomes/invention/ directory>",
      "description": "InventionClaim integrity + coordinator zomes (prior art registry)",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of license_enforcement.rs>",
      "description": "License enforcement in cross-cluster routing dispatch",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of merkle_timestamp.rs>",
      "description": "Blake3 Merkle timestamp chain with external anchoring",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of timestamp_anchor.rs>",
      "description": "File/git/multi external timestamp anchoring",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of mycelix-attribution/ directory>",
      "description": "Reciprocity and usage tracking zomes for automated royalty enforcement",
      "evidence_type": "SourceCode"
    }
  ],
  "license_terms": {
    "license_type": {
      "DualLicense": {
        "open": "AGPL3OrLater",
        "commercial": "Custom"
      }
    },
    "attribution_required": true,
    "derivative_allowed": true,
    "commercial_allowed": true,
    "royalty_percentage": 5.0,
    "royalty_receiver_did": "did:mycelix:tristan-stoltz",
    "custom_terms": "Commercial license available from Luminous Dynamics. The licensing engine itself is sovereign -- enforced by the DHT, not by any central authority."
  },
  "domain": "intellectual-property",
  "classification": {
    "empirical": 2,
    "normative": 1,
    "materiality": 3
  },
  "publish": true
}
```

---

## Invention 6: Mycelix Three-Currency Commons

| Field | Value |
|-------|-------|
| ID | `ld-inv-006-three-currency-commons` |
| Patent | None yet (candidate P-021) |
| Grace Expiry | N/A |
| Domain | `economics` |
| Status | `Published` |
| Classification | E2 / N2 / M3 |

**Title**: Consciousness-Gated Three-Currency Economic System with Counter-Cyclical Mutual Credit

**Description**: Three-currency commons model: MYCEL (soulbound reputation), SAP (transferable with 2% annual demurrage and progressive fees), TEND (mutual credit, 1 TEND = 1 hour, counter-cyclical limits). Consciousness profile gates fee tiers. 25% inalienable treasury reserves.

**Evidence**:
- `mycelix-finance/zomes/` -- payments, treasury, staking, recognition zomes (SourceCode)
- `crates/mycelix-bridge-common/src/consciousness_profile.rs` -- tier-gated fee schedule (SourceCode)

**Prior Art Refs**: `ld-inv-004-consciousness-gating` (consciousness profile gates fee tiers)

**License**: Dual (AGPL-3.0 open + Commercial), derivatives allowed, attribution required, 2% royalty on commercial use.

### `register_invention` Payload

```json
{
  "id": "ld-inv-006-three-currency-commons",
  "title": "Consciousness-Gated Three-Currency Economic System with Counter-Cyclical Mutual Credit",
  "description": "Three-currency commons model: MYCEL (soulbound reputation), SAP (transferable with 2% annual demurrage and progressive fees), TEND (mutual credit, 1 TEND = 1 hour, counter-cyclical limits). Consciousness profile gates fee tiers. 25% inalienable treasury reserves.",
  "inventor_did": "did:mycelix:tristan-stoltz",
  "co_inventors": [],
  "prior_art_refs": ["ld-inv-004-consciousness-gating"],
  "evidence_hashes": [
    {
      "hash": "<Blake3 hash of mycelix-finance/zomes/ directory>",
      "description": "Payment (SAP/TEND/MYCEL), treasury, staking, and recognition zomes",
      "evidence_type": "SourceCode"
    },
    {
      "hash": "<Blake3 hash of consciousness_profile.rs>",
      "description": "Consciousness-gated fee tier schedule used by the three-currency system",
      "evidence_type": "SourceCode"
    }
  ],
  "license_terms": {
    "license_type": {
      "DualLicense": {
        "open": "AGPL3OrLater",
        "commercial": "Custom"
      }
    },
    "attribution_required": true,
    "derivative_allowed": true,
    "commercial_allowed": true,
    "royalty_percentage": 2.0,
    "royalty_receiver_did": "did:mycelix:tristan-stoltz",
    "custom_terms": "Commercial license available from Luminous Dynamics. The commons model is designed for community ownership -- commercial use refers to proprietary deployments only."
  },
  "domain": "economics",
  "classification": {
    "empirical": 2,
    "normative": 2,
    "materiality": 3
  },
  "publish": true
}
```

---

## Dependency Graph

```
ld-inv-001-hdc-ltc-neuron (foundational)
  |
  +-- ld-inv-002-pogq-v41 (uses HDC for gradient encoding)
  |
  +-- ld-inv-003-hdc-fhe (shared HDC foundation)

ld-inv-004-consciousness-gating (foundational for governance)
  |
  +-- ld-inv-005-sovereign-licensing (consciousness gating for enforcement)
  |
  +-- ld-inv-006-three-currency-commons (consciousness gating for fee tiers)
```

Two independent trees rooted at Invention 1 (HDC-LTC) and Invention 4
(Consciousness Gating). Cross-references between trees exist at the
implementation level (PoGQ uses consciousness gating for validator
reputation), but the formal prior_art_refs track the primary dependency only.

---

## Royalty Rules (to be registered after invention claims)

Each invention should have at least one `InventionRoyaltyRule`. These are
registered via `create_royalty_rule` after the invention claims are on the DHT.

| Invention ID | Trigger | Percentage | Min TEND | Receiver |
|--------------|---------|-----------|----------|----------|
| `ld-inv-001-hdc-ltc-neuron` | PerDerivativeWork | 3.0 | 0.5 | `did:mycelix:tristan-stoltz` |
| `ld-inv-001-hdc-ltc-neuron` | PerCommercialUse | 3.0 | 1.0 | `did:mycelix:tristan-stoltz` |
| `ld-inv-002-pogq-v41` | PerDerivativeWork | 3.0 | 0.5 | `did:mycelix:tristan-stoltz` |
| `ld-inv-002-pogq-v41` | PerCommercialUse | 3.0 | 1.0 | `did:mycelix:tristan-stoltz` |
| `ld-inv-003-hdc-fhe` | PerDerivativeWork | 2.0 | 0.5 | `did:mycelix:tristan-stoltz` |
| `ld-inv-003-hdc-fhe` | PerCommercialUse | 2.0 | 1.0 | `did:mycelix:tristan-stoltz` |
| `ld-inv-004-consciousness-gating` | PerDerivativeWork | 2.0 | 0.25 | `did:mycelix:tristan-stoltz` |
| `ld-inv-004-consciousness-gating` | PerCommercialUse | 2.0 | 0.5 | `did:mycelix:tristan-stoltz` |
| `ld-inv-005-sovereign-licensing` | PerDerivativeWork | 5.0 | 1.0 | `did:mycelix:tristan-stoltz` |
| `ld-inv-005-sovereign-licensing` | PerCommercialUse | 5.0 | 2.0 | `did:mycelix:tristan-stoltz` |
| `ld-inv-006-three-currency-commons` | PerDerivativeWork | 2.0 | 0.25 | `did:mycelix:tristan-stoltz` |
| `ld-inv-006-three-currency-commons` | PerCommercialUse | 2.0 | 0.5 | `did:mycelix:tristan-stoltz` |

---

## Registration Procedure

Once a Holochain conductor is running with the `mycelix-knowledge` DNA
installed:

1. **Compute evidence hashes** -- for each evidence file/directory, compute
   `Blake3(file_contents)` or `Blake3(tar of directory)`.

2. **Register inventions in dependency order**:
   - First: `ld-inv-001-hdc-ltc-neuron` and `ld-inv-004-consciousness-gating`
     (no prior art refs)
   - Second: `ld-inv-002-pogq-v41`, `ld-inv-003-hdc-fhe`,
     `ld-inv-005-sovereign-licensing`, `ld-inv-006-three-currency-commons`
     (reference the foundational inventions)

3. **Verify witness commitments** -- after each `register_invention` call,
   retrieve the claim and verify the witness commitment matches.

4. **Register royalty rules** -- for each invention, call `create_royalty_rule`
   with the triggers from the table above.

5. **Anchor to external systems** -- use `timestamp_anchor.rs` to anchor the
   Merkle root to git commits and/or IPFS for independent verification.

### Example zome call (via Holochain client)

```javascript
const result = await client.callZome({
  cap_secret: null,
  cell_id: knowledgeCellId,
  zome_name: "invention",
  fn_name: "register_invention",
  payload: {
    id: "ld-inv-001-hdc-ltc-neuron",
    title: "Hyperdimensional Liquid Time-Constant Neural Network with Closed-Form Temporal Evolution",
    description: "A neural network architecture combining 16,384-dimensional hypervectors...",
    inventor_did: "did:mycelix:tristan-stoltz",
    co_inventors: [],
    prior_art_refs: [],
    evidence_hashes: [
      {
        hash: Array.from(evidenceHash),
        description: "Core unified HDC-LTC neuron implementation",
        evidence_type: "SourceCode"
      }
    ],
    license_terms: {
      license_type: {
        DualLicense: {
          open: "AGPL3OrLater",
          commercial: "Custom"
        }
      },
      attribution_required: true,
      derivative_allowed: true,
      commercial_allowed: true,
      royalty_percentage: 3.0,
      royalty_receiver_did: "did:mycelix:tristan-stoltz",
      custom_terms: "Commercial license available from Luminous Dynamics."
    },
    domain: "machine-learning",
    classification: { empirical: 3, normative: 2, materiality: 3 },
    publish: true
  }
});
```

---

## Notes

- **Evidence hashes are placeholders** (`<Blake3 hash of ...>`) -- they must be
  computed at registration time from the actual file contents at that moment.
  This ensures the evidence commitment matches the exact code version being
  claimed.

- **Prior art refs use invention IDs**, not ActionHashes, because the
  ActionHashes are not known until after registration. The coordinator zome
  resolves IDs to ActionHashes via the `IdToInvention` link.

- **The `witness_commitment` field is auto-computed** by the coordinator zome
  during `register_invention`. It is not included in `RegisterInventionInput`
  because the coordinator computes it from the provided fields plus the
  Holochain-assigned timestamp.

- **Patent grace periods** refer to the 12-month grace period from first public
  disclosure under US patent law (35 USC 102(b)(1)(A)). The DHT timestamp
  serves as evidence of the disclosure date.

- **This registry itself is Invention 5** -- the sovereign licensing engine
  dogfoods by registering itself as a prior art claim on the very system it
  defines.

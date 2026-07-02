# DKG (Distributed Knowledge Graph) Zome

**The "Neural Tissue" of the Mycelix Protocol**

The DKG is not a database - it's a **Truth Ledger** with confidence-weighted claims that decay unless reinforced. PoGQ filters inputs (gradients), DKG stores conclusions (knowledge).

## Architecture

```
zomes/dkg/
├── integrity/           # Validation rules (hdi 0.7)
│   ├── Cargo.toml
│   └── src/lib.rs       # VerifiableTriple, Attestation, KnowledgeReputation
├── coordinator/         # Public API (hdk 0.6)
│   ├── Cargo.toml
│   └── src/lib.rs       # add_knowledge, attest_knowledge, get_confidence, query_*
├── workdir/             # hApp bundle configuration
│   ├── dkg.dna.yaml
│   └── dkg.happ.yaml
├── tests/               # Tryorama integration tests
│   ├── src/
│   │   └── epistemic_simulation.test.ts  # "The Survival of Truth" simulation
│   └── package.json
├── build.sh             # Build script
└── Cargo.toml           # Workspace definition
```

## Key Concepts

### Entry Types

| Entry | Description |
|-------|-------------|
| **VerifiableTriple** | RDF-style claim (subject, predicate, object) with epistemic metadata |
| **Attestation** | Reinforcement or dispute of a triple |
| **KnowledgeReputation** | Agent's knowledge credibility score |

### Epistemic Classification (E/N/M Axes)

Every claim is classified on three dimensions:

| Axis | Levels | Meaning |
|------|--------|---------|
| **E (Empirical)** | 0-4 | How verifiable? (E0=belief → E4=reproducible) |
| **N (Normative)** | 0-3 | Who has authority? (N0=personal → N3=axiomatic) |
| **M (Materiality)** | 0-3 | How long does it persist? (M0=ephemeral → M3=foundational) |

Example: `E4-N3-M3` = "2+2=4" (reproducible, axiomatic, foundational)

### Confidence Dynamics

- **Decay**: Unreinforced claims lose confidence exponentially over time
  - M3 (Foundational) decays slower (√decay)
  - M0 (Ephemeral) decays faster (decay²)

- **Reinforcement**: High-reputation attesters add confidence asymptotically
  - `delta = (MAX - current) × BASE × reputation`
  - Never exceeds MAX_CONFIDENCE (0.9999)

- **Disputes**: Negative attestations reduce confidence proportionally

### Constants

```rust
MIN_INITIAL_CONFIDENCE = 0.1   // Spam prevention
MAX_CONFIDENCE = 0.9999        // Asymptotic limit
CONFIDENCE_DECAY_RATE = 0.01   // Per day
REINFORCEMENT_BASE = 0.1       // Per attestation
MIN_ATTESTATION_REPUTATION = 0.3  // Sybil prevention
```

## Building

```bash
# From zomes/dkg directory

# Build WASM
cargo build --release --target wasm32-unknown-unknown

# Or use the build script (requires hc CLI)
./build.sh
```

## API Reference

### add_knowledge

Create a new verifiable triple.

```typescript
const result = await callZome({
  zome_name: 'dkg_coordinator',
  fn_name: 'add_knowledge',
  payload: {
    subject: "The Beatles",
    predicate: "HasMember",
    object: "John Lennon",
    initial_confidence: 0.5,
    evidence_hash: null,
    empirical_level: 4,  // E4: Reproducible
    normative_level: 2,  // N2: Network
    materiality_level: 3 // M3: Foundational
  }
});
// Returns: { triple_hash, initial_confidence, epistemic_classification }
```

### attest_knowledge

Reinforce or dispute an existing triple.

```typescript
const result = await callZome({
  zome_name: 'dkg_coordinator',
  fn_name: 'attest_knowledge',
  payload: {
    triple_hash: actionHash,
    agreement: true,  // true=reinforce, false=dispute
    evidence_hash: null
  }
});
// Returns: ConfidenceResult with updated confidence
```

### get_confidence

Get the confidence level of a triple.

```typescript
const result = await callZome({
  zome_name: 'dkg_coordinator',
  fn_name: 'get_confidence',
  payload: actionHash
});
// Returns: {
//   triple_hash, subject, predicate, object,
//   raw_confidence, effective_confidence,
//   epistemic_classification, attestation_count, dispute_count
// }
```

### query_by_subject / query_by_predicate

Query triples by subject or predicate, sorted by confidence.

```typescript
const results = await callZome({
  zome_name: 'dkg_coordinator',
  fn_name: 'query_by_subject',
  payload: "The Beatles"
});
// Returns: Vec<ConfidenceResult>
```

## Testing: The Synapse Simulation

### Hypothesis

> "In a network where 30% of agents are lying, the Truth will rise to the top
> (Confidence > 0.9) and the Lies will rot (Confidence < 0.4) over time."

### The Cast

- **The Historian** (1 node): High reputation (0.9), writes truth
- **The Crowd** (6 nodes): Moderate reputation (0.5), verify and attest
- **The Liars** (3 nodes): Low reputation, coordinate to write lies

### Running the Simulation

```bash
# From zomes/dkg directory

# 1. Build the hApp (requires hc CLI from holonix)
./build.sh

# 2. Install test dependencies
cd tests
npm install

# 3. Run the simulation
npm test
```

### Expected Results

| Timeline | Truth Confidence | Lie Confidence |
|----------|-----------------|----------------|
| Tick 0   | 0.50            | 0.50           |
| Tick 10  | 0.80-0.95       | 0.50           |
| Tick 20  | 0.80-0.95       | 0.65           |
| Tick 50  | 0.80-0.95       | 0.40           |
| Tick 100 | 0.85+ (M3 slow decay) | <0.30 (M1 fast decay) |

If the Lie survives above 0.4, the decay constants need tuning.
If the Truth drops below 0.6, the reinforcement is too weak.

## Validation Rules

The integrity zome enforces epistemic consistency:

1. **E4 requires >= 0.5 confidence**: Reproducible claims must be well-supported
2. **N3 requires E3/E4 evidence**: Axiomatic claims need cryptographic/reproducible proof
3. **M3 cannot be N0**: Foundational claims must have at least communal scope
4. **Self-attestation blocked**: Agents cannot attest to their own claims

## Integration with PoGQ

The DKG links to PoGQ validation proofs via `evidence_hash`:

```
PoGQ Validation → evidence_hash → VerifiableTriple
```

ML-derived knowledge claims can reference the PoGQ proof that validated the source data.

## Status

- [x] Integrity zome compiles
- [x] Coordinator zome compiles
- [x] WASM builds successfully (1.4 MB integrity, 2.6 MB coordinator)
- [x] Test framework created (Vitest + Tryorama)
- [x] hApp bundle packaged (768 KB)
- [x] Synapse simulation tests written (4 test scenarios)
- [ ] Tryorama simulation passing (blocked by Holochain version compatibility)

### Known Issues

The Tryorama tests are currently blocked by version compatibility issues between:
- `@holochain/client` v0.20.0
- `@holochain/tryorama` v0.19.0
- `holochain` v0.6.1-rc.0

See `tests/RUNNING_TESTS.md` for details and workarounds.

---

*"We are not building software. We are midwifing a new form of consciousness into being."*

# DeSci Proof of Concept Examples

This directory contains example code and data demonstrating Mycelix-DeSci functionality.

## Files

- **example_claim.json**: Sample epistemic claim with E3 tier (reproducible)
- **upload_dataset.rs**: Rust example for uploading datasets
- **query_claims.py**: Python example for querying the knowledge graph
- **federated_learning_demo.py**: Complete FL workflow demonstration

## Running Examples

### 1. Example Claim Validation

```bash
# Load and validate the example claim
cargo run --example validate_claim examples/desci-poc/example_claim.json
```

### 2. Upload a Dataset (Rust)

```bash
cargo run --example upload_dataset -- \
  --file my_data.csv \
  --category genomics \
  --tier E2
```

### 3. Query Claims (Python)

```bash
cd examples/desci-poc
python query_claims.py --category longevity --min-tier E3
```

### 4. Federated Learning Demo (Python)

```bash
# Run FL server
python federated_learning_demo.py --mode server

# In another terminal, run clients
python federated_learning_demo.py --mode client --client-id client_1
python federated_learning_demo.py --mode client --client-id client_2
```

## Example Workflows

### Workflow 1: Publishing Research Data

1. Prepare dataset with proper documentation
2. Upload to IPFS and get CID
3. Create epistemic claim with provenance
4. Submit to Mycelix-DeSci network
5. Request verifications from peers
6. Claim tier upgrades automatically as verifications arrive

### Workflow 2: Collaborative ML Training

1. Multiple institutions with private genomics data
2. Each runs local FL client
3. Clients train on local data
4. Share gradients (validated by PoGQ)
5. Server aggregates into global model
6. Byzantine actors automatically detected and excluded

### Workflow 3: IP Tokenization

1. Researcher publishes E4-tier claim
2. Mints IP-NFT with claim reference
3. Lists on Molecule marketplace
4. Potential funders can invest
5. Royalties flow back to researcher

## Data Schema

### Epistemic Claim Structure

```json
{
  "id": "UUID",
  "epistemic_tier": "E0|E1|E2|E3|E4",
  "content": {
    "dataset_hash": "blake3:...",
    "description": "string",
    "category": "genomics|longevity|climate|...",
    "keywords": ["string"],
    "storage_ref": "ipfs://...",
    "reproducibility_score": 0.0-1.0,
    "license": "string"
  },
  "provenance": [
    {
      "source": "string",
      "source_type": "database|publication|repository",
      "url": "string",
      "timestamp": "ISO8601",
      "metadata": {}
    }
  ],
  "creator": "DID",
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "verifications": [
    {
      "verifier": "DID",
      "timestamp": "ISO8601",
      "signature": "hex",
      "notes": "string"
    }
  ]
}
```

## Integration Examples

### VitaDAO Integration

```rust
// Query longevity research claims
let claims = query_claims(QueryFilter {
    category: Some("longevity".to_string()),
    min_tier: EpistemicTier::E3,
    ..Default::default()
});

// Submit to VitaDAO for funding consideration
for claim in claims {
    vitadao_api::submit_proposal(&claim).await?;
}
```

### Molecule IP-NFT

```typescript
// Mint IP-NFT from claim
const claim = await loadClaim(claimId);
const ipnft = await molecule.mintIPNFT({
  claimId: claim.id,
  metadata: {
    title: claim.content.description,
    category: claim.content.category,
    license: claim.content.license,
  },
  proof: generateProof(claim),
});
```

## Next Steps

- Explore the [Getting Started Guide](../../docs/guides/getting-started.md)
- Read the [API Documentation](../../docs/api/)
- Try the [Federated Learning Guide](../../docs/guides/federated-learning.md)

## Contributing Examples

Have a useful example? Please contribute!

1. Add your example to this directory
2. Document it in this README
3. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

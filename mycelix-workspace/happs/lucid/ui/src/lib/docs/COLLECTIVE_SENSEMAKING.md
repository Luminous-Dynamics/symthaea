# Collective Sensemaking in LUCID

## Overview

Collective Sensemaking enables distributed consensus-building on beliefs across a peer-to-peer network. Using Holochain's DHT (Distributed Hash Table), LUCID allows communities to share, validate, and build consensus around knowledge without centralized authority.

## Key Concepts

### Shared Beliefs
A belief shared with the collective becomes a **SharedBelief** - your thought anonymized and published to the DHT for others to validate. You control what you share and can prove authorship without revealing identity through zero-knowledge proofs.

### Validation Votes
Peers can vote on shared beliefs using a 5-point scale:
- **Corroborate** (+2): Strong agreement with evidence
- **Plausible** (+1): Reasonable but not personally verified
- **Abstain** (0): No opinion or insufficient knowledge
- **Implausible** (-1): Seems unlikely but possible
- **Contradict** (-2): Directly conflicts with known evidence

### Consensus Levels
Based on aggregated votes, beliefs achieve consensus levels:
- **Strong** (>80% agreement): Widely accepted
- **Moderate** (60-80% agreement): Generally accepted with some dissent
- **Emerging** (40-60% agreement): Still being evaluated
- **Contested** (<40% agreement): Significant disagreement exists

### Trust Relationships
Your influence on consensus is weighted by your trust relationships:
- **Stranger** → **Acquaintance** → **Collaborator** → **Trusted**
- Higher trust = more weight in consensus calculations
- Trust is bilateral and earned through positive interactions

## Using the Features

### Sharing a Belief

1. Select a thought in your knowledge graph
2. Click "Share with Collective" in the thought panel
3. Choose anonymization level:
   - **Full**: Content shared with ZK proof of authorship
   - **Partial**: Tags/type visible, content hashed
   - **Transparent**: Full attribution (your agent ID visible)
4. Optionally add context or evidence links
5. Submit to the collective

### Voting on Beliefs

1. Open the **Collective Sensemaking Panel** (sidebar tab)
2. Select the **Vote** tab to see beliefs awaiting validation
3. For each belief:
   - Read the content and any linked evidence
   - Select your vote (Corroborate → Contradict)
   - Optionally add supporting evidence
   - Toggle anonymous voting if desired
4. Submit your vote

Keyboard shortcuts in Voting Dashboard:
- `1-5`: Quick vote selection
- `Enter`: Submit vote
- `Tab`: Navigate between beliefs

### Viewing Consensus

The **Consensus Reality Map** visualizes collective agreement:
- **Green nodes**: Strong consensus
- **Yellow nodes**: Emerging/moderate consensus
- **Red nodes**: Contested beliefs
- **Edges**: Relationships between beliefs
- **Clusters**: Automatically detected pattern groups

Click any node to see:
- Full belief content
- Vote breakdown
- Timeline of consensus changes
- Related patterns

### Trust Network

The **Agent Relationship Network** shows your social graph:
- **Center**: You
- **Node size**: Interaction frequency
- **Edge color**: Relationship stage
- **Edge thickness**: Trust score

Build trust by:
- Voting on shared beliefs
- Having your shared beliefs validated
- Consistent, quality contributions

## Privacy Features

### Zero-Knowledge Proofs

LUCID uses Winterfell STARKs for privacy-preserving proofs:

1. **Anonymous Belief Proof**: Prove you authored a belief without revealing your identity
2. **Reputation Range Proof**: Prove your reputation exceeds a threshold without revealing exact score
3. **Vote Eligibility Proof**: Prove collective membership without revealing your agent key

### Anonymization Levels

| Level | What's Hidden | What's Visible |
|-------|--------------|----------------|
| Full | Agent ID, exact content | Content hash, ZK commitment |
| Partial | Agent ID | Thought type, tags, hashed content |
| Transparent | Nothing | Full content, your agent ID |

### Data Sovereignty

- Your private thoughts never leave your device
- Only explicitly shared beliefs enter the DHT
- You can unshare (mark for deletion) at any time
- Local replicas sync only with peers you've interacted with

## Patterns and Emergence

The system automatically detects emergent patterns:

### Pattern Types
- **Convergence**: Multiple agents arriving at similar conclusions
- **Divergence**: Systematic disagreement on a topic
- **Clustering**: Beliefs naturally grouping by topic
- **Evolution**: How collective understanding changes over time

### Pattern Detection
Uses HDC (Hyperdimensional Computing) embeddings to:
1. Embed belief content into 16,384-dimensional vectors
2. Cluster similar beliefs using cosine similarity
3. Track cluster evolution across time
4. Surface emerging consensus and contradictions

## Technical Details

### Architecture

```
┌─────────────────────────────────────┐
│           LUCID UI (Svelte)        │
├─────────────────────────────────────┤
│   collective-sensemaking.ts         │
│   (Service Layer)                   │
├─────────────────────────────────────┤
│   Tauri Bridge (Rust)               │
│   - ZKP generation                  │
│   - HDC embeddings                  │
├─────────────────────────────────────┤
│   Holochain Zomes                   │
│   - collective/coordinator          │
│   - privacy/coordinator             │
├─────────────────────────────────────┤
│   DHT (Distributed Hash Table)      │
└─────────────────────────────────────┘
```

### Holochain Entry Types

- `SharedBelief`: Anonymized thought shared with collective
- `ValidationVote`: Vote record with optional evidence
- `ConsensusRecord`: Aggregated consensus state
- `AgentRelationship`: Bilateral trust relationship
- `DetectedPattern`: Emergent pattern from clustering
- `ZkProofAttestation`: Verified ZK proof reference

### Consensus Algorithm

```
consensus_score = Σ(vote_weight × trust_score × recency_factor) / total_weight

where:
  vote_weight = {+2, +1, 0, -1, -2}
  trust_score = relationship_stage_weight
  recency_factor = exp(-age_days / decay_constant)
```

## Best Practices

1. **Quality over Quantity**: Share well-formed, evidence-backed beliefs
2. **Engage Thoughtfully**: Vote based on genuine evaluation, not social pressure
3. **Build Trust Gradually**: Consistent quality contributions build reputation
4. **Use Evidence Links**: Support corroborations with linked thoughts/sources
5. **Embrace Disagreement**: Contested beliefs often reveal important nuances

## Troubleshooting

### "Holochain unavailable"
The UI will fall back to local simulation mode. Check:
- Holochain conductor is running
- WebSocket connection to conductor (default: ws://localhost:8888)
- hApp is installed and activated

### Votes not appearing
- DHT sync may take 5-30 seconds
- Check network connectivity
- Verify peer discovery is working

### ZK proofs failing
- Ensure Tauri backend is running (not web-only mode)
- Check for sufficient system memory (proofs are compute-intensive)
- Verify lucid-zkp crate is compiled

## API Reference

See `collective-sensemaking.ts` for full TypeScript API:

```typescript
// Share a belief
await shareBelief(thought, AnonymizationLevel.Full);

// Vote on a belief
await voteOnBelief(beliefHash, ValidationVoteType.Corroborate, evidence?);

// Get consensus
const consensus = await getConsensus(beliefHash);

// Detect patterns
const patterns = await detectPatterns();

// Get collective stats
const stats = await getStats();
```

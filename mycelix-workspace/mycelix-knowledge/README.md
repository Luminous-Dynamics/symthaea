# Mycelix Knowledge Graph

> Distributed Knowledge Graph with Truth Engine - Social consensus defeats misinformation

## Overview

The Knowledge hApp implements a **Distributed Knowledge Graph (DKG)** on Holochain, providing verifiable claims with dynamic confidence scoring. The "Truth Engine" uses 5-factor confidence calculation to determine which claims are most trustworthy.

## The Truth Engine

Claims are evaluated using 5 weighted factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Attestation Count | 25% | More attesters = higher confidence (diminishing returns) |
| Reputation Weighting | 30% | High-rep attesters contribute more (reputation squared voting) |
| Source Quality | 20% | Academic > institutional > commercial > social |
| Time Decay | 10% | Older claims decay exponentially (domain-specific) |
| Consistency | 15% | Contradictions reduce confidence |

## Genesis Simulation

The canonical test proving the Truth Engine works:

- Alice claims: "The sky is blue" (truth)
- Mallory claims: "The sky is green" (lie)
- Bob endorses Alice's claim

Result: Alice's claim achieves 0.6175 confidence (elevated from 0.5 baseline)

Verdict: Truth wins through social consensus

## Zome Functions

### Claims
- submit_claim - Create a new claim
- get_claims - Get all claims about a subject
- get_truth - Get confidence-filtered claims (Truth Engine)

### Attestations
- attest_claim - Endorse, challenge, or acknowledge a claim

### Reputation
- get_agent_reputation - Get an agent's reputation score

### Discovery
- list_subjects - List all subjects with claims
- ping - Health check

## License

Apache-2.0

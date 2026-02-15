# Example Federated Learning hApp

A complete example demonstrating how to build a federated learning application using the Mycelix SDK with Byzantine fault tolerance.

## Overview

This example shows:
- Setting up a FL coordinator with MATL trust integration
- Managing participant reputation over multiple rounds
- Using Byzantine-resistant aggregation (Krum)
- Cross-hApp reputation sharing via Bridge

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Learning hApp                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Client A   │    │   Client B   │    │   Client C   │      │
│  │   (Honest)   │    │   (Honest)   │    │ (Byzantine)  │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                    ┌────────┴────────┐                          │
│                    │  FL Coordinator  │                          │
│                    │   (Krum + MATL) │                          │
│                    └────────┬────────┘                          │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │               │
│  ┌──────┴───────┐    ┌──────┴───────┐    ┌──────┴───────┐      │
│  │    MATL      │    │   Bridge     │    │   Holochain  │      │
│  │  Reputation  │    │  Protocol    │    │   Conductor  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Enter the nix development environment
nix develop

# Run the example
npx tsx example-fl-happ/main.ts
```

## Files

- `main.ts` - Entry point and round orchestration
- `participant.ts` - Participant client implementation
- `coordinator.ts` - FL coordinator with Byzantine detection
- `config.ts` - Configuration settings

## Configuration

```typescript
const config = {
  // FL Settings
  minParticipants: 3,
  maxParticipants: 100,
  byzantineTolerance: 0.34,
  aggregationMethod: 'krum',

  // MATL Settings
  reputationThreshold: 0.3,
  initialReputation: 0.5,

  // Network Settings
  conductorUrl: 'ws://localhost:9988',
  appId: 'mycelix-fl-example',
};
```

## How It Works

### 1. Initialization

```typescript
import { FLCoordinator, LocalBridge, createReputation } from '@mycelix/sdk';

// Create coordinator
const coordinator = new FLCoordinator(config);

// Initialize participants with reputation
participants.forEach(p => {
  p.reputation = createReputation(p.id);
});
```

### 2. Training Round

```typescript
// Start round
coordinator.startRound(roundNumber);

// Collect gradients from participants
const gradients = await Promise.all(
  participants.map(p => p.computeGradient(model))
);

// Submit gradients
gradients.forEach((g, i) => {
  coordinator.submitGradient(participants[i].id, g);
});

// Aggregate with Krum (Byzantine-resistant)
const aggregated = coordinator.aggregate('krum', byzantineCount);

// Update model
model.applyGradient(aggregated);
```

### 3. Reputation Update

```typescript
// After each round, update reputation based on contribution quality
participants.forEach(p => {
  const quality = evaluateContribution(p.gradient, aggregated);
  if (quality > threshold) {
    p.reputation = recordPositive(p.reputation);
  } else {
    p.reputation = recordNegative(p.reputation);
  }
});
```

### 4. Byzantine Detection

```typescript
// Use hierarchical detection
const detector = new HierarchicalDetector(3, 2);
participants.forEach(p => {
  detector.assign(p.id, reputationValue(p.reputation));
});

const suspected = detector.getSuspectedByzantine();
console.log('Suspected Byzantine nodes:', suspected);
```

## Security Features

1. **Krum Aggregation**: Selects gradients closest to the majority, ignoring outliers
2. **MATL Reputation**: Tracks participant trustworthiness over time
3. **Byzantine Detection**: Identifies colluding adversaries
4. **Rate Limiting**: Prevents gradient flooding
5. **Input Validation**: Rejects malformed gradients

## Expected Output

```
=== Mycelix FL Example ===

Round 1:
  Participants: 10 (2 Byzantine)
  Aggregation: Krum
  Error: 0.0234
  Detected Byzantine: node-8, node-9

Round 2:
  Participants: 10 (2 Byzantine)
  Aggregation: Krum (trust-weighted)
  Error: 0.0156

...

Final Results:
  Rounds completed: 10
  Final model accuracy: 94.2%
  Byzantine detection rate: 100%
  Honest node avg reputation: 0.89
  Byzantine node avg reputation: 0.12
```

## Integration with Holochain

To deploy on Holochain:

1. Build the zomes:
   ```bash
   cd zomes && cargo build --release --target wasm32-unknown-unknown
   ```

2. Package the hApp:
   ```bash
   hc app pack workdir/
   ```

3. Install in conductor:
   ```bash
   hc sandbox generate workdir/mycelix-fl.happ
   ```

4. Connect SDK client:
   ```typescript
   import { MycelixClient } from '@mycelix/sdk';

   const client = await MycelixClient.connect({
     adminUrl: 'ws://localhost:9989',
     appUrl: 'ws://localhost:9988',
     installedAppId: 'mycelix-fl-example',
   });
   ```

## License

MIT

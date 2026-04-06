# Mycelix Observatory

Live dashboard for monitoring the Mycelix ecosystem - hApps, trust metrics, and the economic layer.

**Status**: ✅ **Demo-Ready** - Works without conductor (automatic mock data fallback)

🌐 **Live Demo**: [https://luminous-dynamics.github.io/mycelix-observatory/](https://luminous-dynamics.github.io/mycelix-observatory/)

## Quick Start

```bash
./start-demo.sh
```

Or manually:

```bash
npm install
npm run dev
```

Then open http://localhost:5173

**Demo Mode**: The Observatory automatically uses realistic mock data when no Holochain conductor is available. See [DEMO_MODE.md](./DEMO_MODE.md) for details.

## What You'll See

### Network Health
- **hApp Status**: Real-time health of Mycelix-Core, Mail, Marketplace, Praxis, SupplyChain
- **Byzantine Tolerance**: 34% BFT threshold monitoring
- **Trust Scores**: Network-wide reputation metrics

### Economic Layer (What Makes Mycelix Different)

The economic layer shows how Mycelix aligns incentives with long-term thinking:

| Primitive | Purpose |
|-----------|---------|
| **SAP** (Sentient Action Points) | Behavior-based tokens earned through contribution |
| **CIV** (Civilizational Index Value) | Reputation score (0-100%) based on behavior history |
| **HEARTH** | Community warmth pool - contributors share in network health |
| **KREDIT** | AI agent budgets - controlled spending on tasks |
| **Temporal Commitments** | Time-locked SAP with governance multipliers (1x-7x) |

### Sample Members

The demo shows 4 member archetypes:

1. **Alice (Patient Builder)**: Forest-tier commitment (15yr), 7x governance multiplier, 2.4x patience coefficient
2. **Bob (Active Contributor)**: Sapling-tier (18mo), moderate patience, community-focused
3. **Carol (New Member)**: Sprout-tier (3mo), building reputation
4. **Helper Agent (AI)**: KREDIT-funded, requires 65% CIV from sponsor

## Architecture

```
observatory/
├── src/
│   ├── lib/
│   │   ├── ecosystem.ts      # Network & economic simulation
│   │   ├── stores.ts         # Svelte state management
│   │   └── components/       # UI components
│   └── routes/
│       ├── +page.svelte      # Main dashboard
│       └── epistemic-markets/ # Prediction markets (WIP)
├── start-demo.sh             # Quick launcher
└── package.json
```

## Technology

- **SvelteKit**: Fast, reactive UI
- **TypeScript**: Type-safe simulation logic
- **Vite**: Modern build tooling

## Demo vs Live Mode

### Demo Mode (Default)
- ✅ Zero configuration
- ✅ Works offline
- ✅ Realistic mock data
- ✅ Updates every 5 seconds
- ✅ Perfect for presentations

### Live Mode (Conductor Connected)
- 🔌 Requires Holochain conductor running
- 🔌 Real DHT data from hApps
- 🔌 10-second health checks
- 🔌 Automatic mode switching

See [OBSERVATORY_DEMO_READY.md](./OBSERVATORY_DEMO_READY.md) for complete documentation.

## Deployment

The Observatory is deployed to GitHub Pages at [mycelix-observatory](https://github.com/Luminous-Dynamics/mycelix-observatory).

To deploy updates:

```bash
./deploy.sh
```

This will:
1. Build the static site with `adapter-static`
2. Commit the build output to the gh-pages repository
3. Push to GitHub (Pages updates within a few minutes)

## Related

- [Mycelix SDK (Rust)](/sdk) - Core economic primitives
- [Mycelix SDK (TypeScript)](/sdk-ts) - Client bindings
- [Mycelix-Core](/happs/core) - Federated learning hApp
- [Mycelix-Mail](/happs/mail) - Secure messaging hApp

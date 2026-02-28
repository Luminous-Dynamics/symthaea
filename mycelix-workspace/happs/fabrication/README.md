# Mycelix Fabrication hApp

[![CI](https://github.com/luminous-dynamics/luminous-dynamics/actions/workflows/fabrication-ci.yml/badge.svg)](https://github.com/luminous-dynamics/luminous-dynamics/actions/workflows/fabrication-ci.yml)
**348 unit tests** | **14 WASM zomes** | **Holochain 0.6**

**Distributed Manufacturing Commons for the Civilizational OS**

The Fabrication hApp transforms digital intent into grounded, sustainable, community-owned physical artifacts. It is the **physical hands** of the Mycelix organism.

## Key Innovations

### 1. HDC-Encoded Parametric Designs
Designs are stored as 16,384-dimensional (2^14) **Hyperdimensional Computing** vectors, enabling:
- Natural language design search ("bracket for 12mm pipe")
- Semantic similarity matching across languages
- AI-assisted parametric variant generation
- Intent-based design discovery

### 2. Proof of Grounded Fabrication (PoGF)
Every print carries metabolic accountability:

```
PoGF = (E_renewable × 0.3) + (M_circular × 0.3) + (Q_verified × 0.2) + (L_local × 0.2)

Where:
  E = Renewable energy fraction (linked to Terra Atlas)
  M = Material circularity score (Material Passports)
  Q = Cincinnati quality verification (0-1)
  L = Local economy participation (HEARTH funding)
```

High PoGF scores earn **MYCELIUM** (CIV) reputation tokens.

### 3. Cincinnati Algorithm Quality Monitoring
Real-time teleomorphic monitoring during prints:
- 1000 Hz sensor sampling
- Anomaly detection with configurable thresholds
- Automatic parameter adjustment
- Layer-by-layer quality scoring

### 4. Anticipatory Repair Loop
Property hApp digital twins predict failures, triggering automatic repair workflows:

```
Property hApp → Failure Prediction → Design Search → Local Printer → HEARTH Funding → Part Installed
       ↑                                                                                    |
       └────────────────────────────── Part arrives BEFORE failure ─────────────────────────┘
```

## Architecture

```
fabrication/
├── Cargo.toml                     # Workspace configuration
├── happ.yaml                      # hApp manifest
├── dna/
│   └── dna.yaml                   # DNA with all zomes
├── zomes/
│   ├── designs/                   # HDC-enhanced parametric designs
│   │   ├── integrity/             # Entry types, validation
│   │   └── coordinator/           # CRUD, search, versioning
│   ├── printers/                  # Printer registry
│   ├── prints/                    # Print jobs with PoGF + Cincinnati
│   ├── materials/                 # Material specifications
│   ├── verification/              # Safety claims + Knowledge bridge
│   ├── bridge/                    # Cross-hApp integration
│   └── symthaea/                  # HDC operations + AI
├── crates/
│   └── fabrication_common/        # Shared types
├── sdk-ts/                        # TypeScript SDK
├── docs/                          # Documentation
└── tests/                         # Integration tests
```

## Quick Start

### Building from Source

```bash
# Enter the workspace
cd mycelix-workspace/happs/fabrication

# Build all WASM zomes (14 zomes)
cargo build --release --target wasm32-unknown-unknown

# Run unit tests (348 tests)
cargo test --target x86_64-unknown-linux-gnu

# Package the DNA
hc dna pack dna/ -o workdir/fabrication.dna

# Package the hApp
hc app pack . -o fabrication.happ

# Clippy (deny warnings)
cargo clippy --target wasm32-unknown-unknown -- -D warnings
```

## Cross-hApp Integration

The Fabrication hApp integrates with the entire Mycelix civilizational OS:

| hApp | Integration Point | Purpose |
|------|-------------------|---------|
| **Knowledge** | Safety Claims | Epistemic classification (E/N/M) for design safety |
| **Marketplace** | Design Listings | Trade designs and print services |
| **Supply Chain** | Material Passports | Track material origin and circularity |
| **Property** | Anticipatory Repair | Digital twin failure prediction |
| **HEARTH** | Funding | Local economy funding for community repairs |
| **Terra Atlas** | Energy Grounding | Link prints to renewable energy sources |
| **MYCELIUM** | Reputation | CIV tokens earned from quality sustainable prints |

## Safety Classification

| Class | Description | Verification Required |
|-------|-------------|----------------------|
| 0 - Decorative | No safety concerns | None |
| 1 - Functional | Basic mechanical | Self-certification |
| 2 - Load Bearing | Structural use | Community verification |
| 3 - Body Contact | Touching skin | Material certification |
| 4 - Medical | Medical devices | Professional verification |
| 5 - Critical | Life-safety | Multi-party certification |

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Core Concepts](docs/CONCEPTS.md) — HDC, PoGF, Cincinnati Algorithm
- [API Reference](docs/API_REFERENCE.md)
- [Integration Guide](docs/INTEGRATION_GUIDE.md)
- [Cincinnati Algorithm](docs/CINCINNATI_ALGORITHM.md) — Teleomorphic quality monitoring
- [Design Lifecycle](docs/DESIGN_LIFECYCLE.md) — From creation to fabrication
- [PoGF Specification](docs/POGF_SPECIFICATION.md) — Proof of Grounded Fabrication

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

*Part of the Mycelix Civilizational OS - Technology that serves consciousness.*

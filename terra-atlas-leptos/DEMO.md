# Terra Atlas — Planetary Energy Transition Visualization

**Live:** [atlas.luminousdynamics.io](https://atlas.luminousdynamics.io)

## What You're Looking At

Terra Atlas is an interactive 3D globe showing the global energy transition in real time. It visualizes three clocks ticking on fossil fuels:

### 1. The Geological Clock (Reserves)
73 major fossil fuel deposits — oil fields, gas formations, coal basins, tar sands — positioned at their real geographic coordinates with reserves data from EIA, USGS, and BP Statistical Review. Marker size scales logarithmically with proven reserves (million barrels of oil equivalent).

### 2. The Thermodynamic Clock (EROI)
Each deposit is color-coded by its **Energy Return on Investment** — the ratio of energy extracted to energy spent extracting:
- **Green (>12:1)**: Can sustain modern civilization — Saudi Arabia's conventional fields
- **Amber (5-12:1)**: Basic sustainability — US shale (Permian Basin, Bakken)
- **Red (3-5:1)**: Marginal — Canadian tar sands (Athabasca, 4:1)
- **Dark Red (<3:1)**: Thermodynamically unviable — Venezuelan Orinoco Belt

Use the **timeline slider** to scrub through time. Watch EROI decline as easy reserves deplete — oil at 2%/year, tar sands at 3%/year. What was green in 1950 turns amber by 2030, red by 2060.

### 3. The Atmospheric Clock (Carbon)
Translucent red halos surround each fossil deposit, sized by annual CO2 emissions (production × emission factor per fuel type). The economics module computes:
- **Breakeven carbon price**: the carbon tax at which a deposit becomes unprofitable
- **Social cost of carbon**: $51/ton (EPA 2024 central estimate)
- Deposits that lose money when carbon is priced are **carbon-negative stranded assets**

## Data Layers (toggleable)

| Layer | Count | Source |
|-------|-------|--------|
| Energy Sites | 70 | EIA, clustered |
| Geothermal Nodes | 18 | Global survey |
| Maglev Corridors | 15 | Planned routes |
| Terra Lumina Sites | 12 | Renewable mega-sites |
| Resontia Vaults | 12 | Underground infrastructure |
| Fossil Deposits | 73 | EIA, USGS, BP |
| Nuclear / SMR | 50 | IAEA, planned SMR pipeline |

## The Transition Narrative

Scrub the timeline from Year 0 to Year 300:
- **Year 0**: Fossil deposits glow green (high EROI). Renewables are dim. No vaults or corridors.
- **Year 50**: US shale turns amber. Tar sands turn red. First maglev corridors appear. Renewable sites brighten.
- **Year 100**: Saudi oil enters amber zone (EROI ~7:1). Most coal still green. Vaults begin construction. Nuclear/SMR sites at peak brightness.
- **Year 150**: All oil is amber or red. Renewable sites at full brightness. Full maglev grid. All 12 vaults operational.
- **Year 300**: Fossil deposits invisible (EROI < 1:1). The globe shows only renewable infrastructure, nuclear, and deep civilization vaults.

## Grid Stress Overlay (FEP)

16 major population centers show **allostatic load** — a measure of grid fragility from Symthaea's Free Energy Principle (FEP) engine:
- **Johannesburg / Cape Town**: High stress (Eskom crisis, 0.7-0.8 load)
- **Lagos / Cairo**: High stress (growing demand, limited infrastructure)
- **Tokyo / Seoul**: Low stress (diversified, well-maintained grid)

As fossil EROI declines on the timeline, grid stress increases everywhere — but regions with high renewable penetration stabilize faster.

## P2P Energy Trading

Animated green arcs show simulated peer-to-peer energy trades between renewable sites — surplus solar kWh flowing between neighborhoods via the Mycelix Holochain DHT. When a conductor is running, these arcs show real trades.

## Technical Architecture

```
terra-atlas-core (Rust)     — shared types, math, economics, 51 tests
terra-atlas-leptos (WASM)   — WebGL2 renderer, live at atlas.luminousdynamics.io
terra-atlas-bevy (Bevy 0.18) — wgpu renderer, embedded in Symtropy game
mycelix-atlas (Holochain)   — DHT storage for all data layers
```

The same data flows to both renderers. The Holochain bridge (feature-gated) enables real-time decentralized data.

## Economics Engine

Pure Rust module computing per-deposit:
- Annual revenue, extraction cost, carbon cost
- ROI with and without carbon pricing
- Breakeven carbon price ($/ton CO2)
- IRR via Newton-Raphson
- Years to depletion

**Key insight**: Athabasca tar sands at $45/boe extraction cost and 0.58 tCO2/boe emission factor becomes **carbon-negative** at just $17/ton carbon price — well below the EPA's $51 social cost.

---

*Built with consciousness-first technology serving all beings.*
*Luminous Dynamics — luminousdynamics.org*

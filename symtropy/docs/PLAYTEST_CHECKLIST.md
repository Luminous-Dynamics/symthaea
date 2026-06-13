# Symtropy Playtest Checklist

Run with: `cargo run --features mycelix`
(Without `--features mycelix`, Mycelix systems are disabled — base game only.)

## Pre-Flight
- [ ] Game launches without crash
- [ ] Main menu displays
- [ ] Dungeon generates on "New Game"
- [ ] Player spawns (cyan square)
- [ ] 3 NPCs spawn (green squares)
- [ ] Fusion Core visible (yellow square)
- [ ] Leviathan visible when Stirring/Awake

## Core Gameplay
- [ ] WASD movement works
- [ ] E key extracts core when near it
- [ ] Flashlight flickers with stress
- [ ] Leviathan wakes from noise
- [ ] Game Over on Leviathan Hunting
- [ ] Victory on core extraction complete
- [ ] R restarts from Game Over / Victory

## Mycelix Governance (--features mycelix)
- [ ] HUD shows Phi, TEND balance, oppression, stability
- [ ] Governance log messages appear (proposals, votes)
- [ ] NPC proposals generated when FEP surprise > 0.4
- [ ] Consciousness evolves over time (HUD Phi changes)
- [ ] Oppression warning at index > 0.3
- [ ] Constitutional crisis at sustained oppression > 0.5

## Mycelix Economy (--features mycelix)
- [ ] T key near NPC triggers TEND exchange
- [ ] TEND balance changes in HUD
- [ ] NPC trust increases after TEND exchange
- [ ] Demurrage decays positive TEND balances over time

## Physicalized Cryptography (--features mycelix)

### Byzantine Leviathan (FL)
- [ ] FL rounds run every 8 seconds (log messages)
- [ ] When Leviathan Awake: "FL defense held" message
- [ ] When Leviathan Hunting: "FL DEFENSE OVERWHELMED" message
- [ ] Room tint shifts red when FL quality drops
- [ ] Additional noise from failed FL (accelerates Leviathan)

### DKG Extraction
- [ ] E key near core initiates DKG ceremony
- [ ] Log shows registered participants and defections
- [ ] NPCs with low trust defect (log explains why)
- [ ] TEND exchanges rebuild trust → NPCs participate
- [ ] Successful ceremony (3-of-4) unlocks core at 50%
- [ ] Core color changes: yellow → pulsing blue → green

### Epistemic Fog
- [ ] Player starts dim (E0: small visibility)
- [ ] Scavenging items advances epistemic level
- [ ] Higher E-level = brighter/wider flashlight
- [ ] Coercion (low trust) degrades all E-levels
- [ ] Player sprite color shifts with E-level

### Medical Commons
- [ ] Healing rate logged (60% hoarded → 85% voluntary)
- [ ] TEND dividends distributed to consenting NPCs
- [ ] Coercion detected when NPC trust < 0.2
- [ ] Coercion lifts when all trust restored > 0.4

## Stress Test
- [ ] Run for 5+ minutes without crash
- [ ] Headless test: `cargo run -p symtropy-sim-bridge --bin headless_test -- --ticks 300`
- [ ] All 11 scenarios pass

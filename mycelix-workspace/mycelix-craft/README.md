# Mycelix Craft

Decentralized talent marketplace with living credentials, guild-based mastery, and
employer-funded apprenticeship pathways. This DNA stores opt-in profile metadata,
publicly indexed credential pointers with Ebbinghaus vitality decay, and peer
attestations. Verifiable credentials remain in the Identity/Praxis DNAs and are
referenced via hashes.

## Scope
- Craft profiles (opt-in, user-authored professional identity)
- Living credentials with Ebbinghaus forgetting curve decay
- Skill endorsements (peer attestations)
- Job postings with apprenticeship stakes
- Work history with peer verification
- Connection graph and recommendations
- Application lifecycle (Draft -> Submitted -> Interview -> Offered -> Accepted)

## Build
```sh
cd mycelix-craft
cargo build --release --target wasm32-unknown-unknown
```

Then package the DNA/hApp:
```sh
hc dna pack -o dna/mycelix_craft.dna dna/dna.yaml
hc app pack -o mycelix-craft.happ happ.yaml
```

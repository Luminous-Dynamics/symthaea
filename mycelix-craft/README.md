# Mycelix Professional

Standalone hApp DNA for the Mycelix Professional Network. This DNA stores
opt-in profile metadata and publicly indexed credential pointers, while
verifiable credentials remain in the Identity/EduNet DNAs and are referenced
via hashes.

## Scope
- Professional profile metadata (opt-in, user-authored)
- Published credential pointers (explicitly opt-in)
- Skill endorsements (peer attestations)

## Build
```sh
cd mycelix-professional
cargo build --release --target wasm32-unknown-unknown
```

Then package the DNA/hApp:
```sh
hc dna pack -o dna/mycelix_professional.dna dna/dna.yaml
hc app pack -o mycelix-professional.happ happ.yaml
```

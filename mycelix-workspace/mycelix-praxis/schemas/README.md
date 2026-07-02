# Praxis Schemas

This directory contains schema definitions for Mycelix Praxis.

## Structure

```
schemas/
├── vc/                           # Verifiable Credentials schemas
│   ├── EduAchievementCredential.schema.json
│   └── contexts/
│       └── praxis-v1.jsonld      # JSON-LD context
└── dht/                          # Holochain DHT entry schemas
    └── entries.md                # Entry definitions reference
```

## Verifiable Credentials (VC)

Praxis uses [W3C Verifiable Credentials](https://www.w3.org/TR/vc-data-model/) for educational achievements.

### EduAchievementCredential

Represents completion or achievement in a course, tied to:
- **Course**: Which course was completed
- **Model**: Which ML model version assessed the learner (for provenance)
- **Rubric**: Assessment criteria used
- **Score/Band**: Achievement level

See `vc/EduAchievementCredential.schema.json` for full schema.

### JSON-LD Context

The `contexts/praxis-v1.jsonld` file defines our custom vocabulary for:
- Course and model identifiers
- Federated learning provenance
- Achievement scoring

## DHT Entries

Holochain entry types are defined in Rust code (in `zomes/*/src/`), with documentation in `dht/entries.md`.

Key entry types:
- **Course**: Learning content and structure
- **FlRound**: Federated learning round metadata
- **FlUpdate**: Gradient/update submissions (commitments only)
- **VerifiableCredential**: W3C VCs for achievements
- **Proposal**: Governance proposals

## Validation

- **VC schemas**: Validated against JSON Schema draft 2020-12
- **DHT entries**: Validated by Holochain zome validation functions

## Versioning

Schemas follow semantic versioning:
- **Major**: Breaking changes (e.g., remove required field)
- **Minor**: Backward-compatible additions (e.g., add optional field)
- **Patch**: Documentation or clarification

Current version: **0.1.0**

## Tools

Validate VC schemas:
```bash
# Using ajv-cli (npm install -g ajv-cli)
ajv validate -s vc/EduAchievementCredential.schema.json -d examples/credential.json
```

Generate TypeScript types from schemas:
```bash
# Using json-schema-to-typescript
cd apps/web
npx json-schema-to-typescript ../../schemas/vc/*.schema.json -o src/types/
```

## Examples

See `examples/` directory for sample credentials and entry payloads (TODO).

## References

- [W3C Verifiable Credentials](https://www.w3.org/TR/vc-data-model/)
- [JSON-LD 1.1](https://www.w3.org/TR/json-ld11/)
- [JSON Schema](https://json-schema.org/)
- [Holochain Entry Types](https://developer.holochain.org/concepts/entry-types/)

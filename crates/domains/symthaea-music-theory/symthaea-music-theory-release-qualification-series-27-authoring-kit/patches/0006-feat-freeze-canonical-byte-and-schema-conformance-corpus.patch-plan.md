# Patch 0006: feat freeze canonical byte and schema conformance corpus

**Series:** 27

## Objective

Make the final persisted contract independently implementable.

## Intended changes

- Publish exact positive, boundary, mutation, unknown-field, wrong-role, and unsupported-version vectors for all stable roles.
- Include canonical bytes, digests, semantic summaries, expected stage, and expected issue code.
- Version the corpus independently from implementation releases.

## Required tests

- Rust and at least one independent implementation agree exactly.
- Corpus archives reproduce byte-for-byte.
- Any stable-role encoding change requires explicit versioning.

## Non-claims

- Does not make the corpus exhaustive.
- Does not allow majority voting on disagreement.

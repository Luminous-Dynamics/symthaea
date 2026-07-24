# Patch 0012: Bind verification caches to complete context

**Series:** 24

## Objective

Prevent a result obtained under one policy, verifier, limit profile, or lineage context from being reused under another.

## Intended changes

- Key caches by exact artifact digest, schema/corpus version, expected authority policy, limit policy digest, verifier identity/version, and required predecessor/head identities.
- Never cache timeouts, cancellations, or partial results as acceptance.
- Separate structural parse caches from authority decisions.

## Required tests

- Policy substitution misses cache.
- Stricter limits do not reuse looser-profile acceptance blindly.
- Verifier-version change invalidates authority cache entries.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.

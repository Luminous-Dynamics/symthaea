# Patch 0009: Stream archive validation before extraction

**Series:** 24

## Objective

Verify deterministic public kits without trusting archive paths or loading entire contents.

## Intended changes

- Stream decompression and hashing with compressed-byte, expanded-byte, file-count, per-file, and ratio limits.
- Verify manifest coverage and duplicate paths before materialization.
- Support verification-only mode that performs no extraction.

## Required tests

- Expansion bomb fails within configured bytes.
- Duplicate archive entries fail.
- Manifest omissions and extra files fail.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.

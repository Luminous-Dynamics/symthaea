# Patch 0010: test run cross platform fixed width serialization lanes

**Series:** 27

## Objective

Prove stable artifacts do not depend on pointer width, endianness, or host-specific debug formatting.

## Intended changes

- Run native 64-bit and available 32-bit or cross-compiled serialization lanes.
- Check fixed-width counts, ordinals, enums, canonical lengths, and error codes.
- Compare byte fixtures across lanes.

## Required tests

- Stable bytes are identical across supported lanes.
- Host-dependent persistence fails CI.
- Unsupported targets are documented exactly.

## Non-claims

- Does not claim platforms not executed.
- Does not require runtime support on every compilation target.

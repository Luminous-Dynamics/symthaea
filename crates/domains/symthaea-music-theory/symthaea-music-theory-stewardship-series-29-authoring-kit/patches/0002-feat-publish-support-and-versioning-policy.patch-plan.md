# Patch 0002: feat publish support and versioning policy

**Series:** 29

## Objective

Establish predictable maintenance promises for code, schemas, tools, and evidence artifacts.

## Intended changes

- Define release channels, compatibility windows, security support, deprecation periods, and archive availability expectations.
- Separate library semver, persisted-schema compatibility, CLI stability, and evidence-format stability.
- Document emergency exceptions and required disclosure.

## Required evidence

- Policy is machine-readable and included in release artifacts.
- Declared compatibility is backed by fixtures.
- Unsupported combinations fail clearly.

## Non-claims

- Does not promise indefinite support.
- Does not imply old cryptography remains acceptable for new authorization.

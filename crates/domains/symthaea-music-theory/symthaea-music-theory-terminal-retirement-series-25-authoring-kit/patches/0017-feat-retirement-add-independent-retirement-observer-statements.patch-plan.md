# Patch 0017: feat retirement add independent retirement observer statements

**Series:** 25

## Objective

Allow external observers to attest that they verified the terminal package without becoming retirement authorities.

## Intended changes

- Define statements over the exact terminal checkpoint, package identity, verifier policy, observation epoch, and limitations.
- Support multiple observer classes and conflict reporting.
- Keep observer acceptance separate from the committed retirement receipt.

## Required tests

- Pre-checkpoint observations, wrong package, duplicate observers, and verifier-policy mismatch fail.
- Observer absence does not undo valid retirement.
- Observer statements cannot restart publication.

## Non-claims

- Does not prove observer independence.
- Does not make observations trusted timestamps automatically.

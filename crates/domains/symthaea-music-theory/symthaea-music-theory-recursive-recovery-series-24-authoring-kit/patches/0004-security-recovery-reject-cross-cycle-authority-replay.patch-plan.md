# Patch 0004: security recovery reject cross cycle authority replay

**Series:** 24

## Objective

Prevent recovery, witness, closure, resumption, and reopening signatures from one cycle authorizing another.

## Intended changes

- Include cycle identity and cycle-specific domain separators in every new signed payload.
- Require active expected authority and witness policy epochs for the intended cycle head.
- Explicitly classify legacy signatures as historical evidence only.

## Required tests

- Cycle-one signatures cannot authorize cycle two.
- Closure, resumption, reopening, and recovery signatures are not role-interchangeable.
- Changing cycle identity invalidates all affected canonical payloads.

## Non-claims

- Does not prove signers are independent organizations.
- Does not rotate keys automatically.

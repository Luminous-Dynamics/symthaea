# Patch 0022: feat register terminal retirement schema prefix

**Series:** 35

## Objective

Append the retirement and archive-only roles without changing earlier slices.

## Intended changes

- Register trust-exhaustion policy/report, retirement plan, statements, authorization, receipts, terminal checkpoint, archive profile, custody events, handoff, observer statement, disclosure package, and terminal-state report.
- Use fixed-width fields and stable encodings.
- Publish compatibility behavior.

## Acceptance evidence

- All prior schema prefixes remain unchanged.
- Role collisions and debug-derived persistence fail.
- Independent implementations decode or reject identically.

## Non-claims

- Does not register resurrection roles.
- Does not make schema registration authority.

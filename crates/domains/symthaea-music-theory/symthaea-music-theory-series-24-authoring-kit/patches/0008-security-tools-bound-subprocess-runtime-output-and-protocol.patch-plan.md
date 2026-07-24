# Patch 0008: Bound independent-verifier subprocesses

**Series:** 24

## Objective

Treat external verifier implementations as potentially slow, noisy, crashed, or malicious.

## Intended changes

- Use no-shell invocation, fixed argument vectors, sanitized environment, deadlines, output caps, and protocol message limits.
- Kill process groups on timeout/cancellation.
- Treat malformed, truncated, extra, or conflicting output as explicit failures.

## Required tests

- Infinite-output child is terminated within limits.
- Child process trees do not survive cancellation.
- A timeout cannot be reported as verifier disagreement success.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.

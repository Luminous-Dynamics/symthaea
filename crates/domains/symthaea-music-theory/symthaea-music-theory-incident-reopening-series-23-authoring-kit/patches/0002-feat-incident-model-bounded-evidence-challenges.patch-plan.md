# Patch 0002: feat incident model bounded evidence challenges

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Represent later evidence submissions without treating the submitter as an authority.

## Intended changes

- Add versioned challenge envelopes binding target incident, closure, segment, observed head, evidence kind, referenced immutable artifacts, and submitter-provided context.
- Use bounded strings, collections, attachments, and canonical bytes under the hostile-input policy.
- Assign content-derived challenge identities and explicit unsupported states.

## Required tests

- Wrong target, oversized, malformed, duplicate-reference, and ambiguous-kind challenges fail structural intake.
- Two byte-distinct submissions cannot alias through filenames or local timestamps.
- Submission acceptance means only that the challenge is well formed.

## Non-claims

- Does not authenticate the truth of submitted evidence.
- Does not require public disclosure of submitter identity.

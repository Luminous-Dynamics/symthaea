# Patch 0004: feat incident model reopening trigger policy

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Define verifier-owned rules for deciding when later evidence is technically sufficient to request governed reopening.

## Intended changes

- Add trigger classes for authenticated equivocation, exact continuity contradiction, accepted independent-verifier disagreement, invalid preservation lineage, compromised active policy, and invalid resumed mutation.
- Allow local policy to require multiple corroborating evidence classes or external verifiers.
- Keep subjective allegations outside automatic technical trigger classes.

## Required tests

- Artifact-supplied trigger thresholds cannot weaken expected local policy.
- Unknown trigger classes fail closed.
- One evidence artifact cannot satisfy multiple independence-required slots unless policy explicitly permits it.

## Non-claims

- Does not automatically reopen the incident.
- Does not claim technical triggers establish motive or culpability.

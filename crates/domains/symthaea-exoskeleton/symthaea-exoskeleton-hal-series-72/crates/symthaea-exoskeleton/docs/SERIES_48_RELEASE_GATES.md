# Series 48 Release Gates

Series 48 adds six repository-checkable gates:

1. Cross-layer safety invariants latch contradictions in final applied state.
2. Power-relevant communication requires two agreeing independent routes.
3. Maintenance and diagnostics are authenticated, scoped, isolated, and non-actuating.
4. Emergency shutdown verifies contactor opening, DC-link discharge, and backdrivability.
5. Assistance remains inside an identified wearer-specific biomechanical envelope.
6. Qualification evidence is sequence-continuous, digest-bound, and evaluated against an explicit mission profile.

A candidate release must pass the Series 42 and Series 48 assurance examples,
all-feature tests, formatting, Clippy, dependency audit, fault injection, HIL,
bench energy-isolation tests, endurance campaigns, and independent review.

The qualification ledger deliberately reports `human_worn_authorized = false`.
Authorization requires separately governed physical evidence, human-factors work,
risk acceptance, ethics approval where applicable, and regulatory assessment.

# Creditism Typed-Credit Capability Matrix

## Purpose

Prevent Personal Credit, Community Credit, and Bonus Credit from collapsing into one fungible balance merely because the architecture uses the same nominal unit.

Current Common Planet documentation calls these three streams distinct jobs: Personal Credit is for individual use, Community Credit is for shared purposes, and Bonus flows back as Bonus Personal Credit. Current documentation says only individuals spend Personal Credit and only group accounts spend Community Credit.

Source:
- https://common-planet.org/creditism/architecture

## 1. Credit types

Represent at minimum:

- `PersonalCredit`;
- `CommunityCredit`;
- `BonusPersonalCredit`.

`BonusPersonalCredit` should normally enter the Personal Credit ledger only through an explicit recognition transition; it is not an untyped third spendable balance.

## 2. Capability matrix

| Capability | Personal Credit | Community Credit | Bonus Personal Credit |
|---|---|---|---|
| Individual spends | yes | no | yes |
| Group account spends | no | yes | no unless received/converted through explicit rule |
| Transfer to another person | no | no | no as Personal Credit unless explicit current rule permits |
| Direction by person | n/a | yes, equal personal share | n/a |
| Autonomous investment return | no | no | no |
| Collateral | no | no | no |
| Interest-bearing lending | no | no | no |
| Inheritance as concentrated power | no | no | no |

The exact current profile must own the authoritative capability table.

## 3. No nominal-unit fungibility shortcut

Freeze:

same unit label != same capability

and:

PersonalCredit(10) + CommunityCredit(10)
!=
fungible Credit(20)

unless an explicit qualified conversion rule says otherwise.

## 4. Directed Community Credit

Current architecture says every person directs an equal share of Community Credit toward groups/projects while only group accounts spend it.

Therefore a person holding a directed share has a governance/selection capability, not an unrestricted 1:1 spendable Personal Credit balance.

Do not model:

personal_directed_CC -> personal_wallet_PC

without an explicit conversion event.

## 5. Bonus transition

Bonus currently flows through as Bonus Personal Credit to group members.

That creates an explicit transition:

verified_group_outcome
-> BonusRecognition
-> PersonalCredit issuance

Preserve this lineage so Bonus is not counted both as Community Credit allocation and Personal Credit contribution.

## 6. Adversarial capability tests

Reject:

- Personal Credit spent from a group account;
- Community Credit spent directly by an individual;
- Community Credit transferred person-to-person;
- Personal Credit used to direct Community Credit without an authorized governance event;
- Community Credit treated as inherited Personal Credit;
- Bonus issued twice from the same outcome;
- Personal and Community balances combined to bypass a capability restriction;
- relabeling `community` as `personal` without a conversion receipt;
- serialized balance field accepted without its Credit type/profile;
- replay of a valid Personal Credit receipt against a Community Credit endpoint.

## 7. Accounting boundary

Each typed balance should reconcile independently:

closing_PC = opening_PC + PC_issuance - PC_deletion

closing_CC = opening_CC + CC_issuance - CC_deletion

Cross-type movements must be explicit transactions.

No balance-sheet aggregation should silently net different Credit types merely because nominal units match.

## 8. Governance boundary

Community Credit direction is a governance action.

The scenario must record:

- directing person identity scope;
- directed group/project;
- amount/share;
- current Community Credit allocation profile;
- authority/currentness;
- decision evidence;
- period;
- whether direction is reversible;
- conflict/appeal state.

Do not treat a direction event as a payment.

## 9. Source-version sensitivity

Because historical Creditism documents differ in Community Credit semantics, capability tests must bind the semantic frontier first.

`HistoricalProfile != CurrentProfile`

See `CREDITISM-SEMANTIC-FRONTIER.md`.

## 10. Claim ceiling

A PASS establishes only typed-credit capability behavior under the frozen semantic profile. It does not establish monetary fungibility in law, economic viability, or political legitimacy.
# Constitutional and Meta-Constitutional Evolution

**Status:** research design v0  
**Related:** #7046, #7050, #7047

## Why this layer is necessary

An institutional-evolution laboratory can correctly model operational rules while still hiding the most consequential source of institutional power: who is authorized to change the rules, by what procedure, and who may change that procedure.

The Institutional Analysis and Development tradition distinguishes operational, collective-choice, and constitutional levels. Higher-level rules determine who can make lower-level rules. Recent work on constitutional amendment likewise treats amendment rules as consequential constraints on institutional change itself. Complex adaptive governance research also emphasizes feedback and multi-directional causality across institutional levels.

Sources:
- https://ostromworkshop.indiana.edu/pdf/teaching/iad-for-policy-applications.pdf
- https://ostrom.indiana.edu/courses-teaching/teaching-tools/iad-framework/index.html
- https://www.cambridge.org/core/books/changing-the-rules/introduction/2684F44EF554FCC722C9E80468F683A78
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8762444/

## Rule levels

Model the following levels explicitly when in scope:

~~~text
Operational
  -> day-to-day allocation, production, access, use

CollectiveChoice
  -> rules governing modification of operational rules

Constitutional
  -> rules governing who may participate in collective choice
     and under what conditions

MetaConstitutional
  -> rules governing creation or modification of constitutional rules
~~~

Not every experiment needs to allow mutation at every level. But each enabled level must have an explicit authority boundary.

## Core non-equivalence

~~~text
rule being changed
!= authority to change the rule
!= procedure used to exercise that authority
!= evidence that the procedure was followed
!= evidence that implementation succeeded
~~~

A valid institutional record therefore needs both:

- the identity of the rule being changed;
- the identity of the higher-order rule that authorizes the change.

## Required rule identity

A rule record should bind:

- rule identity;
- semantic version;
- parent rule identity;
- rule level;
- scope;
- authorized rule-makers;
- eligibility;
- amendment threshold;
- amendment procedure;
- veto rights;
- appeal rights;
- timing/cadence;
- emergency powers;
- emergency sunset;
- review requirement;
- persistence/entrenchment;
- conflict-resolution hierarchy.

## The self-modification attack

The laboratory must explicitly prohibit accidental self-authorization.

Unsafe transition:

~~~text
institution gains authority
-> institution changes its own amendment rule
-> amendment threshold falls
-> institution gains more authority
-> institution changes itself again
~~~

That sequence may be a legitimate treatment in a deliberately constitutional-evolution experiment, but it cannot be silently represented as ordinary endogenous evolution.

The correct record must show the higher-order authorization that permitted each change.

## Constitutional lock fixtures

Required adversarial fixtures include:

- operational actor attempts constitutional change without authority;
- constitutional actor attempts meta-constitutional change without authority;
- amendment threshold mutation;
- voting-right mutation;
- veto-right mutation;
- eligibility mutation;
- emergency-power expansion;
- emergency power used after sunset;
- authority concentration followed by amendment-rule mutation;
- majority decision violating an immutable higher-order constraint;
- current rule claiming a nonexistent parent;
- constitutional replacement represented as ordinary amendment;
- implementation claiming authority after adoption failed;
- revoked constitutional authority still producing accepted changes.

Each should fail with a typed disposition rather than generic rejection.

## Positive controls

Include:

- flexible constitutional systems with a valid amendment path;
- rigid systems with valid but infrequent amendment;
- decentralized constitutional change;
- centralized constitutional change;
- intentionally unamendable rules.

Do not encode flexibility or rigidity as inherently good.

## Power observables

Track separately:

- operational control concentration;
- collective-choice control concentration;
- constitutional agenda-setting power;
- veto concentration;
- amendment accessibility;
- coalition size required for rule change;
- effective constitutional rigidity;
- emergency-power duration;
- concentration of rule-revision capability;
- persistence of higher-order authority.

Low concentration in everyday operations does not imply low concentration over future institutional change.

## Cross-level interventions

Required counterfactuals:

1. Hold operational rules fixed while varying constitutional amendment rules.
2. Hold constitutional rules fixed while varying operational conditions.
3. Change constitutional authority without changing monetary/production mechanisms.
4. Change economic outcomes without granting automatic constitutional authority.
5. Remove a constitutional mutation and observe downstream effects.
6. Freeze meta-constitutional rules and compare against the recursively evolving treatment.

The oracle must keep the levels causally separate.

## Qualification boundary

The first deterministic implementation should leave constitutional evolution disabled.

Qualification should first establish ordinary lineage and authority semantics.

A later treatment may enable constitutional or meta-constitutional mutation under a separately versioned profile.

A PASS establishes only that the declared authority and rule-lineage mechanics behaved as specified. It does not establish political legitimacy, constitutional legitimacy, or desirability.
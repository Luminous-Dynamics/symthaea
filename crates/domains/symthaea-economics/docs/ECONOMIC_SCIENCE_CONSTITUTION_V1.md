# Symthaea Economic Science Constitution v1

Status: **research architecture contract**. This document does not claim that
Symthaea has unified economics, validated an economic theory, or acquired policy
or governance authority.

## Purpose

Symthaea Economic Science is intended to compare and refine economic mechanisms
without making an economic school, simulator, optimization objective, or Mycelix
institution the owner of economic truth.

The v1 foundation therefore begins below ideology: common ontology, exact
accounting constraints, decomposed claims, and a model-neutral Economic Theory
Intermediate Representation (ETIR).

## Constitutional non-equivalences

The following distinctions are normative architecture constraints for the
research program:

```text
constraint truth != empirical claim != normative proposition
simulation != observation
historical fit != prospective prediction
fit != causality
confidence != truth
evidence channel != truth tier
model output != evidence about the external world
prediction != recommendation != decision != execution
scientific result != governance authority
Mycelix participation != research consent
Mycelix data availability != Symthaea training permission
```

No transition above is implicit.

## A0 — three kinds of economic statement

### Constraint

A constraint is an accounting identity, conservation relation, declared model
invariant, or other relation that must hold inside the represented system.
Violating a declared hard constraint invalidates that model state; it is not an
interesting empirical result.

v1 implements exact financial double-entry. Physical resource stock-flow
conservation belongs to a later, separately qualified layer rather than being
claimed by this foundation.

### Empirical

An empirical claim is a falsifiable proposition about the observed world. It
must declare its scope, predictions, and observations that would count against
those predictions. Mechanistic claims additionally identify the mechanisms they
assert.

Empirical support is multidimensional in the larger research program. v1 defines
only an orthogonal `EvidenceChannel` vocabulary for future qualified evaluation:
mechanistic, retrospective, out-of-sample, prospective, interventional, and
independent-replication channels. There is no total ordering among them and an
`EmpiricalClaim` cannot self-assign any channel or evidence status.

### Normative

A normative proposition expresses a value judgment or objective supplied by
people or legitimate governance. Empirical evidence may estimate consequences
of pursuing a value, but cannot infer the value itself.

No canonical scalar welfare function is defined by this foundation.

## A1 — theory-neutral economic ontology

The initial state space distinguishes six coarse domains:

1. physical;
2. financial;
3. network;
4. institutional;
5. cognitive;
6. ecological.

A variable has an explicit symbolic identity, primary domain, unit identity, and
description. Domain membership grants no evidence or causal authority.

The ontology is intentionally smaller than a world model. Households, firms,
banks, governments, commons, markets, contracts, resources, and other entities
may be represented later without requiring one preferred economic school.

## A2 — exact accounting kernel

Financial accounting uses integer atoms in an explicit unit rather than binary
floating point. Every journal entry must balance exactly:

```text
sum(positive postings) == sum(abs(negative postings))
```

A ledger applies a valid entry atomically. Unknown accounts, unit mismatch,
unbalanced entries, duplicate postings, and arithmetic overflow fail closed.
Duplicate account registration also fails before mutation; re-registering an
existing account cannot erase or replace its balance.

Money creation, destruction, external-sector flows, equity, and similar events
must use explicit counterpart accounts; a model may not bypass the accounting
identity by mutating balances directly.

This accounting invariant is not an endorsement of any economic school.

## A3 — claims before schools

Economic schools are not first-class truth objects in ETIR v1.

Instead the theory layer represents:

```text
ConstraintClaim
EmpiricalClaim
NormativeProposition
```

An empirical claim carries stable prediction identities and predeclared
falsification criteria for every prediction. Associational claims cannot
silently declare a causal mechanism. Mechanistic claims must explicitly
reference one or more mechanisms.

ETIR additionally checks mechanism integrity. Every predicted outcome of a
mechanistic claim must be produced by one of the mechanisms that claim cites,
and every cited mechanism must lie on a backward path to at least one predicted
outcome. A disconnected mechanism is scientific decoration rather than causal
support and therefore fails closed.

This permits claims originating in different traditions to overlap, contradict,
or become conditionally valid without requiring the whole tradition to win or
lose as a unit.

## ETIR v1

ETIR contains:

```text
EconomicVariable[]
MechanismSpec[]
EconomicClaim[]
```

A validated ETIR graph rejects duplicate identities and references to undeclared
variables or mechanisms.

A concrete model supplies a separate `ModelAdapterDeclaration` identifying:

- the ETIR theory it implements;
- the exact non-normative claims it implements;
- the variables it predicts;
- one or more computational paradigms.

Paradigms are composable. For example, a model may be both agent-based and
stock-flow-consistent. A `ModelAdapterDeclaration` is non-authorizing metadata:
it does not prove that runtime code actually executes the declared mechanisms or
deserves empirical authority. That requires later evidence-plane qualification.
A model adapter may not implement a normative proposition. If it declares an
empirical claim, every outcome predicted by that claim must also appear in the
adapter's declared predicted variables. Therefore:

```text
scientific claim != model family
model declaration != implementation evidence
model output != value authority
claimed empirical mechanism != undeclared model output
```

Two genuinely different model paradigms can implement the same empirical claim
and be compared against the same evidence.

## Authority boundary

Nothing in `symthaea-economics` v0.x may issue action, policy, governance,
financial, contractual, or Mycelix execution authority.

The intended future chain is:

```text
observations
    -> admitted evidence
    -> claims/models
    -> predictions/counterfactuals
    -> scientific evaluation

================ AUTHORITY BREAK ================

human/governance values and rights
    -> deliberation
    -> decision
    -> separately authorized execution
```

Scientific outputs remain advisory evidence unless a distinct governance system
chooses to use them.

## Mycelix boundary

Mycelix may later serve as a privacy-preserving economic observatory and a
substrate for separately governed voluntary experiments. It is not the truth
source of Symthaea Economic Science.

Research use requires an explicit future consent/admission boundary. Ordinary
participation in a Mycelix institution does not imply research participation,
model-training permission, or permission for individual-level economic
surveillance.

## Explicit non-scope of v1

This tranche does not add:

- an agent-based simulator;
- DSGE, HANK, SFC, econometric, or causal model implementations;
- empirical data ingestion;
- a causal inference engine;
- model scoring or regime discovery;
- a welfare optimizer;
- a policy recommender;
- Mycelix data access;
- automatic governance or execution.

Those are later stages and must compose through explicit evidence and authority
boundaries.

## Next qualification target

ESE-A4 should construct a tiny deterministic economy above this foundation and
prove that at least two genuinely different model paradigms can consume the
same ETIR claim while preserving the exact accounting kernel. Only then should
historical or prospective economic data enter the research line.

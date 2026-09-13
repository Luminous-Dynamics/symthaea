# PIE-002D Projection → Accounting Binding Oracle

Status: independent reference semantics; synthetic fixtures only.

## Purpose

`scripts/pie-utility-case-binding-oracle.py` freezes the binding boundary between a complete PIE-002B electrical-demand projection and the explicit supply/recovery facts required by the PIE-002 accounting kernel.

This layer does **not** evaluate feasibility. It proves only that a case can be constructed without inventing missing facts.

## Core theorem

```text
Complete electrical projection
+ explicit supply/recovery context
→ bound accounting case
```

Binding is rejected when the projection is `Incomplete` or `Ambiguous`, when a purported `Complete` projection is internally inconsistent, or when the external context is malformed.

## No-default rule

The binder never creates defaults for:

- recoverable energy;
- recovery duration;
- storage energy acceptance;
- storage charge power;
- storage discharge power;
- recovery delivery / round-trip fraction;
- available energy capacity;
- available sustained power;
- available peak power.

Every one of these facts must be supplied explicitly by the external context.

## Projection consistency

A `Complete` projection must carry:

- electrical energy;
- peak electrical power;
- strictly-positive process time;
- no electrical blocking reason such as missing/multiple peak power or process time.

This prevents deserialized/forged report objects from bypassing the semantic conditions that produced `Complete` in the first place.

Thermal/cooling unresolved reasons are permitted to survive the binding unchanged, but they do not become electrical facts or thermal-feasibility authority.

## Synthetic fixtures

The self-test covers:

- exact preservation of process identity, demand ranges, and explicit context fields;
- rejection of `Incomplete` and `Ambiguous` projections;
- rejection of a forged `Complete` projection missing peak power;
- rejection of a forged `Complete` projection carrying an electrical blocking reason;
- preservation of unresolved thermal/cooling reasons without promotion;
- rejection of a zero-inclusive recovery duration;
- rejection of an invalid delivery-fraction range;
- rejection of duplicate unresolved reasons.

## Boundaries

This oracle does not prove:

- electrical supply feasibility;
- recovery realizability;
- storage dispatch;
- thermal or cooling feasibility;
- process performance;
- equipment capability;
- economics;
- real lunar/Mars process performance;
- hardware or control authority.

Binding != feasibility.

Tracks #2764, #2759, #2724, #1610, #1647, and master #1604.

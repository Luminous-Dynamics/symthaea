# Engineering Trust Kernel — Canonical Binary64 Identity V1

**Status:** independent canonicalization prerequisite for ETK semantic hashes  
**Branch:** `engineering/etk-3b-evidence-plan-oracle`

## Why this exists

ETK semantic identities must be stable across implementations. JSON object-key ordering can be specified directly, but floating-point values remain vulnerable to implementation-specific decimal rendering. A content identity must not depend on whether a serializer chooses `0.1`, `1e-1`, or another round-tripping spelling.

The required theorem is:

```text
same finite binary64 engineering value
=> same ETK numeric identity in every conforming implementation
```

ETK therefore must not use a language's ordinary JSON float pretty/compact rendering as the canonical identity representation.

## Canonical form

A finite IEEE-754 binary64 value is represented as:

```text
f64:<16 lowercase hexadecimal digits>
```

The 16 hexadecimal digits are the big-endian 64-bit IEEE-754 bit pattern.

Before encoding, both signed-zero values are normalized to positive zero. NaN and positive/negative infinity are inadmissible.

Examples:

```text
+0.0   -> f64:0000000000000000
-0.0   -> f64:0000000000000000
1.0    -> f64:3ff0000000000000
-2.5   -> f64:c004000000000000
0.1    -> f64:3fb999999999999a
5e-324 -> f64:0000000000000001
max finite binary64 -> f64:7fefffffffffffff
```

## Semantic choice for signed zero

IEEE-754 distinguishes `+0.0` and `-0.0`, but ETK v1 does not treat the sign of exact zero as an engineering-semantic distinction. Consequently both encodings normalize to `f64:0000000000000000`.

If a future engineering domain genuinely requires signed zero as semantic state, that domain must define a different numeric type/version rather than silently changing V1.

## Non-finite values

NaN and infinity are rejected before identity construction. They cannot be used as threshold, uncertainty, parameter, interval, or other authority-relevant ETK numeric values.

This preserves the broader fail-closed rule:

```text
not representable as finite engineering scalar != admissible identity input
```

## Independence and use

`scripts/etk-canonical-f64-oracle.py` is a standard-library Python reference and imports no Symthaea code. Production Rust must implement the same binary64 identity independently; production code must not call the Python script.

The primitive is intentionally narrower than unit semantics. It canonicalizes the numeric binary64 value only. It does not establish that the number has the correct unit, scale, uncertainty, provenance, measurement method, or physical interpretation.

Therefore:

```text
canonical numeric identity != unit correctness != physical truth
```

## Frozen vectors

The V1 vectors are:

```text
0.0     f64:0000000000000000
-0.0    f64:0000000000000000
1.0     f64:3ff0000000000000
-2.5    f64:c004000000000000
0.1     f64:3fb999999999999a
5e-324  f64:0000000000000001
1.7976931348623157e308  f64:7fefffffffffffff
-1.7976931348623157e308 f64:ffefffffffffffff
```

The self-test additionally requires integer `1` and binary64 `1.0` to canonicalize identically once the value has entered the binary64 ETK numeric domain, while booleans, strings, null, NaN, and infinities are rejected.

## ETK-3B migration requirement

Before production ETK-3B semantic identities are considered stable, every authority-relevant floating value in a canonical preimage must use this representation (or an explicitly versioned equivalent):

- simulation parameter values;
- epistemic and aleatoric uncertainty;
- uncertainty interval bounds;
- evidence-policy thresholds;
- evidence-policy uncertainty budgets;
- any future numeric validity-domain fields.

Human-readable decimal values may still appear in audit records, but they must not be the source of cross-language identity.

This means the currently drafted ETK-3B vectors are **provisional until numeric canonicalization is migrated and re-frozen**. That is intentional pre-merge hardening, not evidence failure.

## Deliberate nonclaims

Canonical binary64 encoding does not authenticate input origin, establish dimensional correctness, determine measurement uncertainty, prove solver correctness, prove physical truth, admit evidence, discharge an obligation, qualify a design, or authorize any downstream physical action.

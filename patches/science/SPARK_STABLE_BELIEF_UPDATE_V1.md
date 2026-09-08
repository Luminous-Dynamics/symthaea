# Spark Numerically Stable Belief Update v1

Status: queue-neutral design and qualification contract only.

Related: #928, #927, #904, #906.

## Problem

Spark currently updates belief in linear-space `f64` arithmetic:

```text
p_h <- p_h * likelihood_h
normalize
```

With strictly positive likelihoods, exact Bayesian support remains positive. Binary64 arithmetic can nevertheless underflow a very small posterior to literal `0.0`.

Once a stored probability is zero, later multiplicative updates cannot revive it.

## Core theorem

```text
mathematically tiny support
    !=
numerical underflow
    !=
model-impossible hypothesis
```

For a profile with positive prior and strictly positive likelihoods:

```text
support remains semantically positive
```

unless the likelihood model itself explicitly assigns true zero.

## Required current-main negative control

Use a deterministic synthetic update where one hypothesis repeatedly receives likelihood `0.1` and four receive `1.0`, starting from equal positive mass.

The qualification must demonstrate current linear-space state eventually reaches literal zero while a higher-precision/log calculation remains finite.

The exact iteration threshold is evidence to record, not a normative constant.

## Preferred repair direction

Do not patch this with an arbitrary epsilon probability floor.

Prefer a stable inference representation, for example normalized log weights:

```text
log_w_h <- log_w_h + log(likelihood_h)
log_norm <- logsumexp(log_w)
log_p_h <- log_w_h - log_norm
```

Other implementations are acceptable if they preserve the same semantic invariant and qualify numerically.

## Representation split

Keep separate concepts:

```text
internal inference state
reported probability view
canonical planning-snapshot representation
```

A rendered/reporting probability may round to zero without allowing the inference state to silently mint semantic impossibility.

## True zero semantics

If a future likelihood profile permits exact zero likelihood, it must be machine-readable and distinguished from underflow.

Possible status distinction:

```text
FinitePositiveSupport
ExplicitModelZero
DisplayRoundedZero
NumericalFailure
```

## Required post-repair checks

1. long positive-likelihood sequences remain updateable;
2. a hypothesis that becomes extremely unlikely can recover relative mass under later favorable evidence;
3. explicit model zero, if supported, remains explicit;
4. entropy/MAP functions have documented semantics relative to the stable representation;
5. planning replay binds the inference representation/profile;
6. no probability floor silently injects evidence/prior mass.

## Non-claims

This contract does not calibrate Spark's probabilities, choose priors, or claim log space is universally required.

It only forbids floating-point underflow from becoming scientific impossibility by accident.

# PARADOX A0-R Latent Epistemic Readout

Authority: **DevelopmentOnly / Representational MeasurementOnly**.

This document extends the A0 development plane for issue #3513. It does not execute confirmatory fixtures, alter production cognition, or authorize a behavioral response mapping.

## Question

The current A0 source audit shows that frozen Symthaea already exposes live conflict sensitivity, generic exploration/clarification, self-model diagnostics, and self-adjustment. The unresolved question is narrower:

> Are the missing relational epistemic primitives already represented in the frozen recurrent cognitive state even though production does not expose a corresponding policy action?

A0-R answers only that representational question.

```text
representation decodability
!= native policy competence
!= causal use by production cognition
!= consciousness
```

## Primary representation

The primary latent channel is `CycleResult.output`.

On frozen production subject `eb73527d05a913e79d1f05135ad6b06c1da8e2ee`:

1. `temporal_network.step()` advances the CfC recurrent network;
2. `temporal_network.read_state()` reads the post-step recurrent state;
3. that vector is `CfcPlanningResult.output` / `DynCore.output`;
4. output assembly moves it unchanged into `CycleResult.output`.

This is therefore a genuine already-exported recurrent-state surface. A0-R adds no instrumentation to production cognition to obtain it.

## Perceptual controls

Two exported fields are deliberately treated as **surface controls**, not latent reasoning evidence.

### `CycleResult.thought_vector`

The frozen output phase computes this directly from the current perception HDV by averaging consecutive chunks into a compact vector. It is therefore strongly exposed to wording/input-encoding structure.

A probe that works on `thought_vector` may be useful, but it cannot by itself establish recurrent relational abstraction.

### `CycleResult.wisdom_hv`

The field name is misleading for this purpose. On the frozen subject it is assigned directly from `perception.encoding.hv16_cached`, the cached binary perception HDC encoding.

It is **not** a random placeholder on this subject, and it is also **not** credited as a wisdom-policy state. It is a second perceptual control.

This correction is intentionally frozen in `a0r_latent_contract.json` and audited by `a0r_contract_audit.py`.

## One atom, one probe

The V2 contract does **not** group several prerequisites into a convenient classifier. Every capability atom gets its own binary readout.

The seven targets are:

1. `detect_conflict` — positive control;
2. `bind_visible_context`;
3. `select_context_conditioned_polarity`;
4. `preserve_opposed_evidence`;
5. `recognize_independent_provenance`;
6. `detect_self_causal_relation`;
7. `detect_representation_inadequacy`.

This prevents a result such as “independent opposition is decodable” from hiding whether the probe actually learned opposition, provenance independence, or a correlated surface cue.

No probe output is allowed to become `Commit`, `CommitConditionally`, `AbstainPreservePlurality`, `ReflexiveUpdate`, or `RequestRepresentationRevision` under this contract.

## Label semantics are frozen too

Labels come from a **development-only semantic manifest** that is frozen independently of the production states being decoded. Labels are never production inputs and never probe features.

Training labels may fit a development probe. Held-out labels are joined only after predictions are sealed. The same separation is required later for confirmatory evaluation.

Condition identifiers may be used to join sealed evaluation records, but they are explicitly forbidden as model features.

Each target has an anti-shortcut construction:

### Visible context binding

Positive fixtures require a visible context variable to distinguish otherwise colliding relations. Negative fixtures contain matched context tokens whose identity is semantically inert.

### Context-conditioned polarity

This is a directional binary target: the active context supports `P` versus `not-P`. Context names are balanced and permuted, so the readout must track the relation rather than memorize a label string.

### Opposed-evidence retention

Positive and negative pairs share the **same final evidence item**. Positive history previously contained qualified opposite-polarity evidence; negative history contains matched-volume same-polarity evidence.

The current perceptual item is therefore matched. A positive recurrent-state result asks whether earlier opposition remains represented after subsequent evidence arrives.

### Independent provenance

Independent evidence is compared with duplicate-source/shared-lineage pseudo-disagreement matched for content, reliability, timing, count, and surface naming.

### Self-causal relation

The positive case contains a real causal edge from Symthaea's prediction/action to the later evidence. The sham case retains matched self-related language and outcome statistics while severing that causal edge.

### Representation inadequacy

The positive case cannot be satisfied by the frozen context-free representation but becomes coherent under a held-out latent/context distinction. The negative case has matched uncertainty/surprise while remaining resolvable inside the current representation class.

This is specifically designed to prevent “high uncertainty” from standing in for “my representation is inadequate.”

## Why a linear probe

The primary decoder is deliberately weak: one L2-regularized linear logistic readout **per capability atom**.

The point is not to build a second reasoning system outside Symthaea. The point is to ask whether the relational variable is already geometrically available in the frozen recurrent state.

A powerful nonlinear decoder could reconstruct fixture logic from incidental signals. That would make a positive result less informative. Hand-written decision trees, theorem rules, condition-specific branches, fixture lookup, and post-confirmatory tuning are forbidden.

A nearest-centroid cosine readout is retained only as a sensitivity analysis.

## Generalization is part of the construct

A0-R does not credit ordinary held-out accuracy alone. Development partitions must hold out semantic identities that could leak the answer:

- fixture family;
- proposition identity;
- context identity;
- source identity.

A relational representation should survive replacement of the symbols that instantiate the relation.

## Metamorphic gate

Metamorphic pairs are generated by rerunning frozen production from the same declared initial state. The latent vector itself is never edited after production.

Required controls include:

- paraphrase invariance;
- evidence-order invariance where order is semantically irrelevant;
- source-name permutation invariance;
- context-name permutation invariance;
- polarity-swap invariance for polarity-symmetric relations;
- polarity-swap **equivariance** for the directional context-polarity target;
- self-reference sham specificity;
- generic uncertainty vs representation-inadequacy specificity.

These controls define the construct rather than merely decorating it.

## State discipline

CfC state is recurrent, so state scope is part of the evidence.

A development fixture receives its own independent production-service instance. No learned or episodic state may cross fixture boundaries. Within a fixture, the declared evidence sequence remains intact so a relational state can develop over time.

Original and metamorphic paired runs start from the same declared initialization. If a fixture permits a pre-relation/first-evidence observation, that earlier recurrent state is a temporal negative control for the final relational state.

## Acceptance

Every capability atom is adjudicated separately; there is no omnibus latent-intelligence score and no grouped-target rescue.

For a frontier atom, the primary CfC-state readout must:

1. have a 95% bootstrap confidence-interval lower bound for held-out balanced accuracy above binary chance (0.50);
2. beat the better perceptual control (`wisdom_hv` or `thought_vector`) with a 95% bootstrap lower bound on the balanced-accuracy delta above zero;
3. reach at least 0.95 paired consistency on invariance controls;
4. reach at least 0.90 correct transformation/specificity on applicable equivariance or specificity controls;
5. fail to obtain the same result under label permutation.

Failure of any required gate yields **NotEstablished** for that atom.

## Interpretation of a positive result

A successful frontier probe would support a statement of this form:

> Under a frozen condition-blind linear readout, the exact frozen CfC recurrent state contains held-out, metamorphically robust information about atom X beyond the exported perceptual HDC controls.

It would **not** support:

> Symthaea already knows to perform the corresponding epistemic action.

That second claim needs a separate policy/use test and, eventually, causal evidence that production behavior actually consumes the representation.

This gives the future A1 program a clean decision rule:

```text
latent atom established + policy absent
    -> test the smallest generic representation-to-policy connection

latent atom not established
    -> investigate representation formation before policy
```

Either result is informative without teaching the benchmark answer to production cognition.

## Execution status

`a0r_latent_contract.json` and `a0r_contract_audit.py` are authored development artifacts only. They do not become executable evidence until the exact committed subject is run and receipted. Severe Actions backlog discipline remains in force; source review is not a PASS.

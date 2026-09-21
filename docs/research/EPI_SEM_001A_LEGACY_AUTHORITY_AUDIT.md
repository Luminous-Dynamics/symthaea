# EPI-SEM-001A — Legacy epistemic authority producer audit

Parent issue: #5283 (`EPI-SEM-001`)

## Purpose

Freeze the current legacy epistemic-label surface before changing semantics.

This tranche is deliberately **audit-only**. It does not rename `EmpiricalTier` / `EmpiricalLevel`, change serialized values, repair producers, or grant any new authority. Its purpose is to make the existing semantic collision complete and mechanically visible so it cannot spread silently while the compatibility-safe repair is designed.

## Base identity

The audit was prepared against:

```text
main commit
adb69f11fa8068b019cc5bb598d0c7726a197fc9

tree
35a6c5fdba319556af9bb487838734f67c8ac0d6
```

That tree is byte-identical to the earlier `5fd5259af776c0a7f1b9608f040bfae9009871a1` tree. The two intervening commits only added and then removed an empty accidental scratch path and therefore changed history but not the resulting source tree.

## Core separation

```text
statistical strength
!= cryptographic proof
!= producer authentication
!= independent reproduction
!= public reproducibility

observation count
!= cryptographic proof

runtime cycle count
!= cryptographic proof

weighted epistemic score
!= proof that any particular authority property holds
```

## Confirmed legacy producer classes

### 1. HDC statistical retrieval

`crates/core/symthaea-core/src/hdc/statistical_retrieval.rs`

The local `EmpiricalTier` maps z-score bands directly to names including:

```text
E3CryptographicallyProven
E4PubliclyReproducible
```

A high z-score is useful evidence about separation from the file's declared random-similarity model. It is not, by itself, cryptographic verification or a reproducibility demonstration.

### 2. Causal self-explanation

`src/consciousness/causal_explanation.rs`

`compute_epistemic_tier()` can promote a causal relation to `E3CryptographicallyProven` when a local predicate based on evidence count and the relation's derived confidence passes. The current `has_counterfactual_proof()` witness is:

```text
self.evidence.len() >= 20 && self.confidence > 0.9
```

That can represent a statistical/support-strength heuristic, but it is not cryptographic proof.

### 3. Cognitive-loop cycle-count promotion

`src/cognitive_loop/cycle_consciousness.rs`

The cognitive loop currently promotes the empirical tier based on total runtime cycle count, including `E3CryptographicallyProven` after the declared threshold. Runtime age/activity is not evidence of cryptographic verification.

### 4. Legacy epistemic coordinate and scalar collapse

`src/consciousness/epistemic_tiers.rs`

The legacy coordinate contains E/N/M axes, provides `quality_score()` and `contextual_quality_score()`, and directly converts the local E3/E4 names into the shared `EmpiricalLevel` names.

The scalar scores may remain useful as legacy UI/heuristic summaries, but they must not become machine authority for an orthogonal property such as authentication, reproducibility, causal support, or replication.

The convenience `axiom()` constructor also bundles maximal E/N/M values. A later repair should decide whether this remains a descriptive legacy constructor or is replaced by explicit independently established properties.

### 5. Shared legacy vocabulary

`crates/domains/symthaea-epistemic-types/src/global_ledger.rs`

The shared legacy enum defines:

```text
E3CryptographicallyProven = 3
E4PubliclyReproducible = 4
```

This file owns the compatibility vocabulary, not proof that any producer actually demonstrated those properties.

## Mechanical inventory guard

`scripts/audit_epistemic_authority_labels.py` scans Rust source for the two legacy authority-bearing labels and requires the exact currently known file set.

It also checks a narrow set of semantic-debt witnesses for the producer mechanisms above.

Expected current Rust files containing E3/E4 legacy labels:

```text
crates/core/symthaea-core/src/hdc/statistical_retrieval.rs
crates/domains/symthaea-epistemic-types/src/global_ledger.rs
src/cognitive_loop/cycle_consciousness.rs
src/consciousness/causal_explanation.rs
src/consciousness/epistemic_tiers.rs
```

Run:

```bash
python3 scripts/audit_epistemic_authority_labels.py
```

A successful result is named `PASS_INVENTORY`, deliberately not `PASS`. It establishes only that the observed legacy surface matches the frozen inventory.

If a future repair removes a witness, the guard should fail until the audit is deliberately updated. If a new source file begins using E3/E4, the guard should also fail rather than silently expanding the allowlist.

## Repair direction frozen by this audit

The next semantic tranche should separate at least these concepts:

```text
statistical / empirical support strength
cryptographic verification state
producer authentication state
reproducibility state
replication state
```

Do not make them ordinal aliases of one another.

Legacy E0–E4 values may remain readable for compatibility, but no new MEL-EPI positive claim should be minted from their ordinal value alone.

## Relationship to MEL-EPI-001

MEL-EPI-001 uses positive closed-world claim scopes and exact provenance. This audit is intentionally independent of the currently queued MEL-EPI-001A qualification lineage.

Therefore:

```text
legacy E3/E4 value
!= MEL-EPI claim scope

legacy quality score
!= MEL-EPI authority
```

MEL-EPI source adapters may retain legacy fields as descriptive provenance where needed, but positive claims must come from source-specific verification/admission rules.

## Qualification / nonclaims

This source tranche does not establish:

- that the inventory is exhaustive outside the scanned Rust source roots;
- that every legacy E0/E1/E2/N/M use is semantically correct;
- cryptographic proof or producer authentication;
- reproducibility or replication;
- statistical validity of any producer;
- causal validity;
- scientific truth;
- MEL-EPI claim authority;
- execution or physical-action authority.

The script is a regression inventory, not a semantic validator.

## Successor

`EPI-SEM-001B` should introduce compatibility-safe orthogonal evidence/verification properties and targeted regressions proving that z-score, evidence count, confidence, or cycle count alone cannot mint cryptographic/reproducibility authority.

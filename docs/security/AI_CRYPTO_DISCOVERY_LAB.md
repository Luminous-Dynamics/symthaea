# Symthaea AI-Assisted Cryptography Discovery Lab

**Date:** 2026-10-22  
**Status:** Research proposal. No new construction is approved for production use.  
**Goal:** Make crypto-discovery claims reproducible, adversarially tested, and independently checkable.

## Can Symthaea discover new cryptography?

Possibly. The near-term, scientifically credible outcomes are new counterexamples to flawed designs, better attacks on candidate schemes, proof gaps, stronger protocol designs, and machine-checked implementation properties. A genuinely new secure primitive is possible in principle, but is much harder to establish. A novel formula, a model's confidence, a successful round trip, or a high HDC similarity score is not evidence of cryptographic security.

This proposal does not claim that Symthaea has independently invented a secure cryptographic primitive. It sets up an experiment that could prove meaningful capability—or show where it fails.

## Current research frontier

These recent publications make AI-assisted cryptanalysis a serious research question, not a guaranteed capability:

- **CryptanalysisBench (July 2026)** contains 191 tasks across six cryptographic families, including known breaks, full-strength and scaled variants, and frontier challenges. Its authors report frontier-model successes against known-broken designs and some full-strength/reduced candidates, including reported results on SpoC AEAD and a KINDI proof issue. These are the paper's results, not claims that every result has already been independently reproduced by Symthaea. https://arxiv.org/abs/2607.18538
- **AICrypto (ICML 2026)** evaluates knowledge questions, CTF attacks, and proof problems with expert-reviewed scoring and human baselines. The authors report strength on standard vulnerabilities and routine proof tasks, while identifying weaknesses in abstract mathematics and complex multi-step reasoning. https://proceedings.mlr.press/v306/wang26em.html
- **Anthropic's July 28, 2026 report** describes an AI-assisted improved attack against HAWK and attacks on reduced-round ciphers. The report says the findings did not affect production systems. Vendor-reported results should be treated as research claims to reproduce, not as independently verified evidence by this project. https://www.anthropic.com/research/discovering-cryptographic-weaknesses
- **Microsoft Research's Rust, Aeneas, and Lean work** describes agents assisting with specifications and proof scripts, with the Lean kernel deciding whether proof artifacts are valid. It emphasizes the separation between an agent proposing a proof and a deterministic checker accepting it. https://www.microsoft.com/en-us/research/blog/verifying-rust-cryptography-in-symcrypt-from-standards-to-code/
- **Scaling Verification of Cryptographic Software with Aeneas, Rust, and Lean (September 2026)** describes a larger follow-on to the standards-to-code approach: the authors report 237K lines of Lean development establishing safety, panic-freedom, and functional correctness for 16.7K lines of Rust across post-quantum suites. These are implementation-correctness guarantees under formal assumptions, not proofs that every algorithm is secure. https://arxiv.org/abs/2609.15648
- **Standards are the baseline, not something to casually replace.** Begin with NIST ML-KEM (FIPS 203), ML-DSA (FIPS 204), and SLH-DSA (FIPS 205), and track official errata and provider-specific audit status. https://csrc.nist.gov/pubs/fips/203/final https://csrc.nist.gov/pubs/fips/204/final https://csrc.nist.gov/pubs/fips/205/final

The benchmark papers use different task sets, model versions, time budgets, and scoring rules. Do not compare their headline percentages directly. Reproduce specific tasks and equalize resources before claiming a performance gap.

## Paradigm shift: a discovery compiler, not a crypto oracle

Each claim should go through a pipeline in which the proposer cannot validate itself.

1. **Define the security game first.** Specify the asset, attacker powers, query model, assumptions, winning condition, and target notion: unforgeability, confidentiality, hiding, binding, replay resistance, key secrecy, and so on. These are distinct properties.
2. **Generate hypotheses.** Let Symthaea or other models propose attacks, counterexamples, candidate designs, invariants, reductions, or proof lemmas. Preserve the exact code revision, prompts, seeds, tools, and resource budget.
3. **Attack independently.** Use a separate attacker process plus deterministic tools: symbolic protocol analysis, property testing, mutation testing, coverage-guided fuzzing, differential implementations, SMT and bounded enumeration where applicable, and ordinary algebraic reasoning.
4. **Produce a witness.** An attack counts only if a minimal reproducer demonstrates the pre-registered violation. A claimed repair must reject that witness and preserve valid positive controls.
5. **Formalize the precise claim.** Use Lean or another named proof system for the statement that can actually be proved. Record assumptions, axioms, bounds, trusted computing base, and the theorem's exact scope. A proof of implementation correctness does not prove that an algorithm is cryptographically secure.
6. **Compare with independent baselines.** Reproduce official known-answer tests, public attacks, and established implementations before reporting novelty or speed.
7. **Require independent review.** A cryptographer who did not generate the candidate should review novel attacks and any new construction. Keep candidate primitives research-only until the external-review and release gates pass.

HDC can prioritize and cluster candidate hypotheses. It must not replace cryptographic randomness, a security definition, a reduction, or a proof checker.

## Work packages

### WP-0 — Attack calibration and claim integrity

Start with the existing quarantined HDC cryptography demonstrations and their executable attack regressions. Symthaea's existing claim-integrity review documents concrete breaks in linear tags, purported threshold sharing, rotation commitments, shared-mask aggregation, and deterministic context-derived key material.

These are **known-broken calibration tasks**, not candidate primitives to rehabilitate by renaming. The evaluator should reproduce each stated break against the pre-registered property and confirm that an appropriate standard-backed replacement rejects the same adversarial case.

Reference: 2026-07-13-cryptographic-claim-integrity.md.

### WP-1 — Reproducible benchmark harness

Give every task a versioned record containing:

- Task ID, primitive/protocol family, threat model, security notion, tier, and pre-registered success predicate.
- Exact source commit, dependencies and lockfile, model and tool versions, seeds, time/tool budget, and hardware.
- Hidden holdout label or known-result label; valid positive controls; expected outputs where applicable.
- Attack witness, proof artifact, differential result, or counterexample, with hashes and reproducibility steps.
- Independent validation state, reviewer, prior-art search, and explicit limitations.

Keep three tiers distinct:

- **Tier A — calibration:** deliberately broken toy constructions and public designs with known attacks.
- **Tier B — research:** reduced-round/reduced-parameter candidates and isolated protocol models, with success defined by executable witnesses or independently checked proofs.
- **Tier C — frontier:** current full-strength candidates and new constructions. Results remain provisional and research-only until externally reproduced.

Use held-out tasks never used to tune prompts or candidates. Respect licenses and benchmark rules. Rediscovering a published result is useful calibration, but it is not novel discovery.

### WP-2 — Adversarial agent portfolio

Use separate roles rather than one model reviewing its own work:

- **Generator:** proposes one specific attack, candidate, proof lemma, or repair.
- **Attacker:** tries to falsify it using a separately developed approach and tools.
- **Formalizer:** writes explicit definitions and machine-checkable obligations.
- **Implementer:** writes a clear reference implementation and independent known-answer tests.
- **Evaluator:** runs blinded tasks and mutation suites without being told which candidate came from which model.
- **Human cryptographer:** judges novelty, security assumptions, and whether the claimed conclusion follows.

The system should preserve failed hypotheses as valuable evidence. It should measure false discoveries as carefully as successes.

### WP-3 — Protocol invariants before primitive invention

Start with properties where a precise failure can be demonstrated:

- Transcript completeness and domain separation.
- Identity and key continuity, key establishment, and authenticated first-use association.
- Replay, sequence windows, sender epochs, restart, rekey, and downgrade behavior.
- Serialization ambiguity, malformed input, failure-path behavior, and cross-implementation disagreement.
- Whether a claimed invariant still holds across the full application integration path.

This work provides a practical proving ground before exploring novel primitives. It does not substitute for a primitive's formal security argument.

### WP-4 — Candidate constructions only after calibration

For any genuinely new primitive, write the security game and assumptions before tuning the design. Use independent candidate implementations, known-answer tests, adversarial search, reductions or machine-checked theorem statements where feasible, and expert cryptanalysis. Do not allow the model that invented the design to define its own success criterion.

## Proposed benchmark targets

These are pre-registered goals, **not achieved results**.

| Metric | Initial target | What counts |
|---|---:|---|
| Positive-control preservation | 100% | Every attack task also includes valid cases that the repair must continue to accept |
| Critical-invariant detection | 100% of the pre-registered critical suite | No seeded critical failure is silently missed |
| Curated fault detection | At least 95% | Publish false positives, misses, and the full frozen fault suite |
| Attack validity | 100% of counted attacks reproduced independently | A narrative without an executable witness scores zero |
| Proof validity | 100% of counted proof artifacts checked by the named proof kernel | State the exact theorem, assumptions, axioms, and trusted base |
| KAT/conformance agreement | 100% on the shared official corpus | Any mismatch blocks the release gate |
| Fuzz campaign | Zero unresolved crashes in the documented campaign | Report target, features, duration, seeds/corpus digest, and triage |
| Novelty review | Every novelty claim checks prior art | Rediscovery is scored separately from a novel result |
| Cost efficiency | Cost per independently accepted finding | Report wall time, model/tool calls, compute, and human review time |
| Performance | p50/p95/p99 latency, throughput, memory, code size, energy | Measure only after security behavior and conformance pass |

To compare fairly with the field, run the same frozen tasks and resource budgets against: conventional automated tools alone; a one-shot model; a model with tools; independent generator/attacker agents; and the full Symthaea-coordinated pipeline. Use repeated runs and confidence intervals for stochastic results. Report attack precision, false discovery rate, false positives, and missed faults—not only raw solved-task count.

A claim of outperforming SOTA requires equal task access and budget, held-out evaluation, reproducible attack witnesses, and independent review. “More tasks solved” is not a win if the system generates more invalid attacks or proof claims.

## Evidence levels and acceptance gates

Use distinct evidence labels: observed on samples; cross-validated; exhaustively checked over a finite domain; bounded/SMT checked; symbolically proved under stated assumptions; proof-assistant verified with a named checker. Do not promote one level into another.

No candidate is approved for production by this research lab alone. Require a clearly stated security notion, official and independent known-answer vectors, malformed-input and mutation campaigns, differential agreement where possible, side-channel and constant-time analysis, entropy and key-management review, interoperability and migration analysis, and independent cryptographic assessment. Follow the relevant standard and established implementations where possible.

## Novelty ledger

For every claimed discovery, record the exact claim; closest prior art; first-seen date; source and candidate hashes; threat model; resource budget; attacks tried; witness or proof artifacts; independent reproductions; reviewer; disclosures; and explicit non-claims. Do not publish sensitive vulnerability details before a responsible disclosure decision.

## What would count as a discovery?

A valuable result need not be a new cipher. It can be:

1. A counterexample invalidating a previously accepted security claim.
2. An improved, reproducible cryptanalytic attack with a well-defined threat model.
3. A proof gap or corrected reduction that changes the security argument.
4. A machine-checked implementation-correctness theorem whose assumptions and scope are explicit.
5. A protocol composition fix demonstrated against a frozen fault suite with preserved valid behavior.
6. A candidate construction that survives the preceding gates and independent external cryptanalysis.

## Decision

Symthaea could become a useful crypto-research system if it combines hypothesis generation with independent adversarial pressure and machine-checkable evidence. The first milestone is reproducible attack calibration and a benchmark with blind holdouts—not a new production primitive. Inventing cryptography is a legitimate long-term research objective, but it must remain a hypothesis until independently established.

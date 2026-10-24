# Guideline-to-Runtime Conformance and Change-Impact Design v0

**Status:** research design; not an implemented compiler, clinical-content authority, or validated medical device.  
**Scope:** make clinical guideline updates reviewable as structured changes and prove that an approved software bundle preserves a separately reviewed interpretation.  
**Primary premise:** WHO SMART Guidelines and its L4 Frontline executable-reference effort already establish machine-readable and executable guideline representations as a serious implementation direction. This design does not claim that idea as novel.

## 1. Research question

Can we reduce the risk that a guideline revision is mistranscribed, incompletely encoded, mismatched to a jurisdiction, silently translated incorrectly, or incompletely propagated into the application—while making every released software representation testable, reversible, and traceable?

The claim under test is **implementation fidelity and lifecycle reliability**, not that software can validate medical science or that a guideline itself is clinically correct.

## 2. Proposed evidence pipeline

```text
Authoritative guideline edition / jurisdiction policy
                    |
                    v
      Source manifest + rights/access record
                    |
                    v
 Reviewer-authored atomic requirements (not free-form AI output)
                    |
                    v
       Requirement-to-bundle traceability map
                    |
                    v
       Typed protocol bundle / state-machine table
                    |
          +---------+----------+
          |                    |
          v                    v
 Independent known-answer   Old-vs-new semantic
 test oracle                change-impact analysis
          |                    |
          +----------+---------+
                     |
                     v
    Deterministic validator + adversarial mutations
                     |
                     v
   Versioned offline bundle + signed provenance receipt
                     |
                     v
   App/runtime contract tests and visible version state
```

Every stage is a separate artifact with an identity and a review boundary. AI may assist with locating likely changed text or suggesting a draft mapping, but a model-generated interpretation cannot approve itself or flow directly into a released protocol bundle.

## 3. Data model (proposed)

### Source record

Required identity metadata:
- source authority and canonical URL or controlled document identifier;
- exact edition/revision/date and retrieval time;
- jurisdiction and intended user/population scope;
- reuse/licensing status and permitted transformation scope;
- locally held artifact SHA-256 where the artifact can lawfully be retained;
- accountable reviewer identity/role and approval record.

A hash establishes byte identity, not provenance, authenticity, permission to reuse, or clinical correctness. Avoid copying proprietary guideline text into public tests when the source licence does not permit it.

### Atomic requirement

A requirement is a reviewer-managed semantic obligation, not necessarily a paragraph copied from a source:
- stable requirement ID;
- source locator (section/table/flow node/page as applicable);
- applicability predicate with unknown-state behavior;
- required runtime behavior or explicit out-of-scope reason;
- severity classification assigned by clinical risk review;
- links to one or more independently authored expected-outcome cases;
- status: draft, in-review, accepted, rejected, superseded, or unresolved.

Unresolved interpretation is allowed as a visible state. It cannot be silently coerced into a clinical action.

### Requirement mapping

Each in-scope requirement must map to:
- one or more typed bundle nodes/transitions;
- at least one conformance case or a reason why one cannot safely be authored yet;
- the exact bundle revision and digest;
- a semantic review record for safety-critical mappings.

Completeness targets are 100% explicit disposition for all requirements in the declared scope; 100% of safety-critical accepted requirements traced to a transition and an independently reviewed test. “Explicitly excluded” still needs a reason and reviewer.

### Release bundle

Bundle identity includes schema version, jurisdiction, intended-use class, source-manifest digest, requirement-map digest, protocol-content digest, compiler/validator version, effective interval, signing-key identity, anti-rollback generation, approval record, and release test receipt.

The release artifact and the source manifest are different things. Changing source metadata can matter even when generated instructions are byte-identical; changing rendered wording can matter even if transition topology is unchanged.

## 4. Change impact as a first-class operation

For candidate source edition B compared with accepted edition A:

1. Detect changed source loci and record the comparison method.
2. Ask an accountable reviewer to classify each change as semantic, editorial-only, applicability change, deleted/replaced requirement, or unresolved.
3. Propagate changed IDs into requirement mappings and conformance cases; missing impact links block release.
4. Re-run affected and global safety-critical tests on the candidate bundle.
5. Re-run adversarial mutations against version, jurisdiction, validity window, revoked key, rollback floor, altered digest, omitted requirement and unsupported schema.
6. Produce a diff receipt that states added/removed/changed requirements, affected transitions, cases run, failed cases, manual-review decisions, and exact source/bundle/runtime digests.
7. Keep the previous accepted bundle available only under an explicit approved rollback policy. A revocation or safety correction must not be bypassed by accepting an older but correctly hashed bundle.

The software can prove that declared test and mapping links are present and consistent; it cannot prove that the source review identified every clinically meaningful change. That remains an independent review responsibility.

## 5. Semantic preservation for language and accessibility

Translation, simplification, text-to-speech, icons, screen-reader labels and compressed “one-screen” layouts are part of the safety case, not cosmetic layers.

Proposed constraints:
- Safety-critical instructions are versioned separately by language and transformation.
- Every supported translation has qualified linguistic review and clinical meaning review appropriate to that language.
- Generated rephrasing is not permitted to rewrite an action, timing, condition, negation, warning, or escalation requirement at runtime.
- Text/audio/navigation rendering tests bind to the same underlying typed instruction ID.
- Voice transcription uncertainty and negation flips cannot be promoted to a confirmed fact without an allowed confirmation path.
- Unsupported languages are disclosed; unreviewed machine translation must not be presented as an approved protocol.

Test invariants include preservation of actor, action, object, condition, sequence, negation, threshold/source reference, stop/escalation obligation and uncertainty. Automated semantic similarity can be used to prioritize review but not to approve clinical equivalence.

## 6. Conformance and adversarial benchmark expansion

Add a frozen set of test families:
- **Known-answer:** independently authored source-requirement → expected runtime disposition cases.
- **Differential:** candidate bundle vs previous accepted bundle on unchanged requirements; no unexplained behavior drift.
- **Change coverage:** every accepted source change links to an impacted test or a documented reviewer decision.
- **Mutation:** omitted requirement, swapped branch, flipped negation, changed jurisdiction, shortened effective interval, altered source digest, stale signature, old generation, unknown enum, duplicate transition, missing source link, unsupported language.
- **Metamorphic:** irrelevant metadata changes must not change protocol behavior; clinically material fields must alter the traceability/change-impact report; input order should not affect canonical output when order is declared irrelevant.
- **Independent-oracle controls:** the implementation cannot generate the only expected outcome. Include seeded bad encodings to prove the test harness catches deliberate faults.
- **Human-factors:** tasks measure finding and interpreting approved content; the study endpoint is not merely whether users prefer the interface.
- **Held-out sets:** retain independent source-change families and mutations that are not used during implementation tuning.

Primary engineering metrics:
1. 100% in-scope requirement disposition and 100% safety-critical mapping-to-test coverage;
2. 100% rejection of tampered, expired, revoked, jurisdiction-mismatched and disallowed rollback bundles;
3. zero observed critical mismatches against the frozen known-answer suite, with every counterexample recorded;
4. 100% of approved source changes represented in the change-impact receipt;
5. zero unexplained runtime behavior drift on unchanged applicable requirements;
6. zero translation/accessibility transformations that change any predeclared safety-critical semantic element in the frozen reviewed suite;
7. reproducible bundle and receipt digests for a pinned toolchain/build.

These are target properties of the test artefact, not proof of clinical correctness or zero real-world risk.

## 7. Why this is a plausible differentiator—not yet a superiority claim

WHO's SMART Guidelines framework supports consistent digital implementation of recommendations, and WHO's L4 Frontline project describes executable reference software for emergency settings using machine-readable data and calculations. Other guideline execution platforms and clinical content vendors may have similar lifecycle controls; a serious SOTA comparison must investigate implementations beyond public marketing material.

The hypothesis worth testing is a tightly scoped, public, independently replayable assurance contract that spans:
- source edition → atomic requirement → executable bundle → renderer → offline cache;
- candidate revision → explicit semantic classification → impact tests → signed release record;
- runtime incident → visible unknown/stale state → structured handoff;
- all of the above under no network, partial update, revocation, key rotation and rollback attempts.

We should compare this design against a basic static content package, a conventional flowchart, and accessible guideline-execution tooling available for lawful hands-on evaluation. For each comparator, record exact version, feature parity, jurisdiction, device, test setup and what could not be tested. Do not infer that an unpublished feature is absent.

## 8. Immediate implementation scope

The next Rust increment should not parse copyrighted guideline prose or encode CPR, choking, bleeding, burn, medication or other treatment instructions. It should model and validate **synthetic requirement mappings** only:
- unique requirement IDs and unique source locators within a source revision;
- accepted mappings require an explicit bundle target and test ID;
- safety-critical accepted mappings cannot omit independent expected-outcome references;
- unresolved items cannot enter a release candidate;
- changed source requirement IDs must appear in the impact report or have an explicit, reviewed unchanged/excluded disposition;
- receipts bind source-manifest, mapping, bundle, test suite, validator source, executable and exact source revision;
- deliberate mutations verify each invariant fails closed.

This makes the compiler/assurance direction testable without presenting unreviewed medical content to users. A proper clinical authoring process, licensing review, intended-use classification, local regulatory assessment and human-factors study are prerequisites for real protocol bundles.

## 9. References

- [WHO SMART Guidelines](https://www.who.int/teams/digital-health-and-innovation/smart-guidelines)
- [WHO L4 Mobile Reference App / Frontline](https://www.who.int/tools/em-care/l4-mobile-reference-app)
- [IFRC International First Aid, Resuscitation and Education Guidelines 2025](https://www.ifrc.org/document/ifrc-international-first-aid-resuscitation-and-education-guidelines-2025)
- [AHA Adult Basic Life Support, 2025 Guidelines](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001369)
- [FDA Clinical Decision Support Software Guidance, January 2026](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
- [SAHPRA guidance index](https://www.sahpra.org.za/guidelines/)

These references orient the research; none independently validates this design or its eventual implementation.

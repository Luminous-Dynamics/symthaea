# REA-RUST-040 — population completeness, selective disclosure and omission resistance

**Status:** research contract and independently authored fixture matrix. No production completeness validator, universal population schema, source-collection permission, or new trust score is introduced.

## Executive finding

A set of individually valid, signed, hash-bound and independently replayed records can still be an incomplete view of the population they purport to describe.

Examples:

- Every record returned by an API page is valid, but pagination stopped early.
- A cursor sequence skips a page but still returns a terminal cursor.
- A complete traversal of a declared query omits records excluded by an undocumented filter.
- Every enumerated item was fetched, but several could not be parsed or processed and were silently discarded.
- A transparency service exposes every statement that was registered, while an issuer withheld other statements before registration.
- An archive reader verifies every selected member, but no complete archive inventory or expected-population reconciliation was performed.
- A telemetry source observes four entities while an independently qualified expectation covers five; this establishes a discrepancy relative to that expectation, not proof that the real-world population contains exactly five.

**Paradigm shift:** completeness is a scoped claim about a population definition and a collection/processing process. It is not a property inherited from individually valid records, a provider flag, a signature, a page count, or a transparency receipt.

## 1. Six dimensions that must remain distinct

These dimensions form a coverage vector, not a scalar. A system may be complete in one dimension and unknown or incomplete in another.

| Dimension | Question | Evidence needed | Common false upgrade |
|---|---|---|---|
| Population definition | Who defines the relevant universe, query, filters, partition, version and interval? | Exact query/schema/filter identity plus independently qualified expectation authority | A declared query is treated as the whole real-world universe |
| Enumeration coverage | Were all items in that declared population enumerated? | Stable snapshot/root, page/cursor/sequence chain, terminal marker, gap/duplicate detection, count reconciliation | Seeing a terminal cursor is treated as proof no earlier page was skipped |
| Acquisition coverage | Was every enumerated item acquired and bound to the same snapshot/session? | Requested/acquired/failed/unsupported item accounting and exact item identities | A successful subset is reported as the complete population |
| Processing coverage | Was every acquired item processed under the declared decoder/parser/analysis contract? | Per-item processed/failed/unsupported/partial outcomes and exact processor version | Validating all successfully parsed records hides records the parser could not process |
| Issuer/source participation | Does the source have an independently established obligation and process to disclose every relevant item it created or observed? | Independent publication/issuance history, issuer participation audit or qualified external obligation | A complete transparency log is treated as a complete issuer statement history |
| Temporal coverage | Does the declared interval have gaps, clock uncertainty, retention loss, expired cursors or supersession? | Valid-time interval, acquisition/knowledge time, retention and gap evidence | A current snapshot is used to claim a gap-free historical interval |

The six dimensions are not interchangeable, and no single “complete” boolean should stand in for them.

## 2. Scoped completeness outcomes

Keep these named outcomes distinct. They are candidate result labels for future source-owned evaluators, not new values for the generic v0 ClaimDisposition enum.

- **CompleteWithinDeclaredQuery** — all items in one exact, versioned query/snapshot/filter scope were enumerated, acquired and processed, with failures/unsupported items accounted for. This says nothing about whether the query itself equals the real-world population.
- **CompleteWithinCapturedArtifact** — all bytes/entries within one exact captured artifact were enumerated and processed under the qualified artifact reader/processor. This does not establish producer authenticity or benignness.
- **CompleteWithinRegisteredTransparencyView** — all entries in a named tree/checkpoint/subject/interval view were independently enumerated and verified. This covers registered statements, not every statement an issuer might have created.
- **ObservedSubset** — only the listed observations are established; the population definition or full enumeration basis is unavailable.
- **IncompleteOrGapped** — missing page, cursor, sequence, partition, interval, acquisition, parser or processing outcome.
- **PopulationDefinitionUnverified** — the query/selection has no independently qualified authority or completeness basis.
- **IssuerParticipationUnknown** — no independent evidence establishes that all relevant issuer-created statements were submitted.
- **ConflictedCompleteness** — independent manifests, counts, checkpoints, source authorities or reconciliation outcomes disagree.
- **UnsupportedCompleteness** — the format/protocol cannot establish the requested completeness dimension.

A consumer may receive more than one outcome at once—for example, CompleteWithinRegisteredTransparencyView together with IssuerParticipationUnknown and TemporalCoverageIncomplete.

## 3. SCITT and transparency: the critical issuer-participation boundary

IETF RFC 9943 defines SCITT as a system for making single-issuer signed statements transparent. A transparency service records statements submitted under its registration policy and issues receipts; relying parties can audit the registered statement history. RFC 9943 §9.3 explicitly warns that issuers can refuse to register statements or selectively submit some but not all statements they issue.

Therefore:

CompleteRegisteredView does not imply CompleteIssuerHistory.

An inclusion receipt proves that the particular signed statement was registered. It does not prove that no other issuer statement was withheld before registration. A log consistency proof establishes consistency of the registered view, not completeness of issuer participation.

This is not a criticism of SCITT: the protocol explicitly states the residual boundary. REA should preserve that distinction and require a separate participation/issuance completeness basis before making issuer-history claims.

Reference: [IETF RFC 9943, especially §§1, 2.2, 9.2 and 9.3](https://www.rfc-editor.org/rfc/rfc9943.html).

## 4. No-upgrade laws

The current precision-aware replay stack already includes an executable no-upgrade test, self_reported_population_completeness_cannot_upgrade_whole_archive_claims, in [#10958](https://github.com/Luminous-Dynamics/symthaea/pull/10958). It demonstrates two separate outcomes:

1. A provider observation can be supported as the exact report the provider made.
2. A WholeArchiveCompleteness claim remains Unverified under v0, even when the provider reports all pages seen, equal observed/total counts, issuer-wide submission, or complete registered transparency view metadata.

The follow-on completeness programme should preserve these laws:

- Provider self-report cannot define or authorize its own population.
- Terminal cursor does not repair a missing earlier page.
- Equal counts do not repair duplicate records or an unstable snapshot.
- Complete acquisition does not imply complete processing.
- Complete processing of an observed subset does not imply complete enumeration.
- Complete registered transparency view does not imply complete issuer participation.
- Complete artifact traversal does not imply producer authenticity, benignness or safety.
- Zero observed records with no qualified expectation does not prove absence.
- Multiple provider names sharing an incomplete upstream feed do not create independent completeness.
- A missing item outside a claim's dependency scope should not invalidate unrelated claims; apply selective invalidation through #10630.

## 5. Ownership and composition

Do not create another generic evidence graph, receipt envelope, source identity model, trust score or temporal database.

Compose:

- #10629 typed claim entailment and whole-archive completeness boundary;
- #10958 precision-aware independent claim replay and the current no-upgrade test;
- #10760/#10798 for target-time validity, knowledge-time, supersession and temporal gaps;
- #10410 for provider dependency lineage and common-mode analysis;
- #10630 for repository/source continuity and selective revalidation;
- #6283 for sampling-frame, population-generalization and measurement validity;
- #453 for qualified expectation authority and discrepancy semantics;
- existing archive-reader and EvidenceBundle owners for captured-artifact enumeration.

The exact authority that defines a population remains domain-specific. A source may establish completeness within its declared query while population-level completeness remains unverified.

## 6. Adversarial fixture matrix

The machine-readable design matrix is at docs/research/rea-population-completeness-fixtures-v0.json. Its 20 expected classifications are independently authored design labels, not implementation outputs.

It covers:

1. Stable declared query with all pages, items and processing accounted for, but issuer participation unknown.
2. First page only with no terminal cursor.
3. Skipped cursor/page followed by a terminal marker.
4. Duplicate page inflating a reported count.
5. Total count changing between pages without a stable snapshot.
6. Undocumented filter excluding a relevant class.
7. Only a subset of enumerated items successfully acquired.
8. Parser/decoder failures silently omitted from downstream results.
9. Complete traversal of one captured archive without a whole-population claim.
10. Complete registered transparency tree with issuer participation unknown.
11. Issuer withholding statements before SCITT registration.
12. Inclusion proof without a qualified checkpoint/consistency/monitoring basis.
13. Superseded statement in a historically complete log.
14. Retention or interval gaps in historical acquisition.
15. Independent population manifests disagreeing.
16. Self-reported all_pages_seen / all_issued_submitted flags without independent evidence.
17. Different provider names backed by the same upstream incomplete collection.
18. Zero observed records with no qualified expectation.
19. Complete source snapshot but incomplete downstream processing.
20. Unrelated population item omission that must not invalidate a claim independent of that item.

## 7. Benchmark targets

Report each denominator separately; no composite score.

| Metric | Proposed finite-suite target |
|---|---:|
| False-completeness rate | 0 accepts when any required dimension is missing/unknown |
| Omitted-page / sequence detection | 100% over the declared finite protocol suite |
| Acquisition accounting | 100% of enumerated items accounted as acquired/failed/unsupported |
| Processing accounting | 100% of acquired items accounted as processed/partial/failed/unsupported |
| Hidden-filter detection | 100% when query/profile differs from the independently declared population basis |
| Issuer-participation classification | 100% distinction between registered-view completeness and issuer-history unknown |
| Contradiction preservation | 100% of seeded manifest/count/checkpoint disagreements retained |
| False-incompleteness rate | Report separately; no suppressing valid claims because of unrelated missing items |
| Selective invalidation | Precision/recall after scoped item/page/population changes |
| Operational cost | Latency, bytes, peak RSS and allocations by page count/population size |
| Abstention | Explicit unsupported/unknown and missing-evidence denominators |

Finite zero-failure results require confidence bounds and exact scope. They do not establish zero underlying risk or generalization to untested protocols.

## 8. Industry comparison

SCITT, Certificate Transparency and Rekor provide useful mechanisms for authenticating and auditing registered statements or log entries. SLSA v1.2 Build/Source tracks describe provenance and source-process properties. TUF protects update metadata and client update behavior. These systems are complementary: none should be silently interpreted as a proof that an issuer disclosed every statement, that an API enumerated its full real-world population, or that all downstream processing succeeded.

REA's differentiator is to make the population and collection process explicit and to keep source participation, enumeration, acquisition, processing, and temporal continuity separately falsifiable.

This work should not claim to replace any of those systems. It composes their exact guarantees and residual limitations.

## 9. Staged implementation plan

1. Freeze the dimension model and 20-case fixture matrix.
2. Add the named no-upgrade metamorphic assertions to the existing claim/replay qualification owner where practical; do not duplicate #10958's current self-report test.
3. Identify source-owned population/expectation contracts for one narrow protocol (for example, an archive inventory or a versioned API pagination manifest).
4. Implement an isolated completeness evaluator for only that protocol, preserving failed/unsupported/partial items.
5. Add an independent replay implementation and blinded omission fixtures.
6. Add temporal and source-lineage inputs only through #10760/#10410/#10630 owner APIs.
7. Measure false completeness and omission detection before agent-level capability or performance comparisons.

### Exit criterion

An independent consumer can state exactly which population was defined, which portions were enumerated, acquired and processed, whether the issuer/source could have selectively withheld data, what time interval is covered, and which unknowns prevent a stronger claim.

## 10. Claim ceiling

A completeness result can only say that a named population was completely covered under a specified authority, query/snapshot, collection process, processing contract and time interval. It cannot establish that the declared population equals the entire real-world universe unless that universe definition is independently justified. Complete transparency of registered statements does not prove complete issuer participation; complete artifact enumeration does not prove authenticity, benignness or truth.

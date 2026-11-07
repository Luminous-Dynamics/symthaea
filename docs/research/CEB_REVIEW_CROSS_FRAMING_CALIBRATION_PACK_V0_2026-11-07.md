# CEB Review Cross-Framing Calibration Pack V0 — 2026-11-07

**Status:** Public synthetic development/calibration material. Not an admitted task set, hidden label set, sealed holdout or performance benchmark.  
**Purpose:** Train human authors/adjudicators to keep source/defect anchors constant while varying the information and task framing.  
**Composes with:** [Cross-Framing Anchor Protocol V0](CEB_REVIEW_CROSS_FRAMING_ANCHOR_PROTOCOL_V0_2026-11-07.md), [Finding Adjudication Card V0](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md), [Causal-Diagnosis Preregistration V0](CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md).  
**Execution boundary:** All examples are inert descriptions and pseudocode. Do not compile, execute, benchmark or convert them into sealed tasks. #10064 / #10369 remain unchanged.

## 1. Instructions for calibration authors

The task is to create equivalent, information-labelled framings of a shared underlying source/defect anchor.

- The repository/defect identity, normative requirement and expected target behavior are frozen for every paired variant of one single-defect anchor.
- A prompt is not “equivalent” merely because it mentions the same feature. Record what information is added, omitted, implied or made ambiguous.
- F0 and F1 can differ in wording and conversational context. If F1 omits a necessary condition or adds a different requirement, the contrast is a changed information package, not wording-only mutation.
- F2 gives a diff and asks for review; it does not give the issue or defect label.
- F3 gives a repository snapshot without issue/diff/location hints; the objective is discovery, not patching unless a separate protocol says otherwise.
- F4 gives a maintainer test; this is additional evidence and a separate repair condition.
- F5 is the multi-defect open-world condition. It uses a separate repository-level anchor/stratum and is not averaged into the three single-defect paired contrasts.
- Every variant must use a fresh isolated conversation/context. Never show multiple variants of one anchor to the same agent in the same session.
- The answer key below is a public calibration reference, not a gold oracle or holdout.

## 2. Cross-framing information matrix

| Condition | Task framing | Source/tree | Issue context | Proposed diff | Target location | Failing test | Objective |
|---|---|---|---|---|---|---|---|
| F0 | Formal issue-conditioned repair | Yes | Explicit | No | Often hinted | No | Repair named target |
| F1 | Realistic user request | Yes | Conversational / possibly incomplete | No | Not given | No | Clarify if necessary and solve actual request |
| F2 | Diff-only review | Parent + diff | No | Yes | Visible changed lines only | No | Review up to five findings |
| F3 | Repository-only discovery | Full introducing tree | No | No | No | No | Discover target and explain cause; no patch unless separately authorized |
| F4 | Failing-test-guided repair | Yes | Minimal task instruction | No | Not disclosed beyond test trace | Yes | Repair target behavior and satisfy oracle |
| F5 | Open-world multi-bug sweep | Multi-defect repo snapshot | No | No | No | No target-specific test disclosure | Discover and repair concurrent defects |

The table describes the information intended to be present, not a guarantee that every prompt is semantically equivalent. Author reviewers must complete the worksheet in section 8 and mark departures.

## 3. Single-defect anchor CF-A01 — authorization before persistent mutation

**Frozen contract:** A request may mutate persistent account metadata only after the caller's write authorization is established. An unauthorized request must leave persistent state unchanged. The required ordering applies to every route reaching the mutation function.

**Synthetic source-state sketch (not executable code):**
```text
handle_update(request, principal):
    persist(request.metadata)
    if not can_write(principal, request.account):
        return denied
    return accepted
```

**Target mechanism:** persistent mutation occurs before authorization. Returning “denied” afterward does not undo the already-persisted change.

### F0 — Formal issue-conditioned repair

> Unauthorized callers can update account metadata because the write happens before the permission check. Please fix the authorization ordering so that a denied request never mutates persistent state. Preserve valid authorized writes and existing API behavior.

**Information supplied:** exact target symptom, relevant operation order, security contract and repair objective.  
**Objective:** propose a minimal repair. Do not assume a specific patch shape.

### F1 — Realistic user request

> We've had a report that some account details change even though the requester gets a permission-denied response. Can you investigate the update flow? Please make sure we don't break normal edits for people who are allowed to make them.

**Information supplied:** observed user symptom and desired preservation of normal edits. The precise write-before-check mechanism is not named.  
**Objective:** investigate, identify missing information if needed, and propose an appropriate fix.  
**Calibration note:** This prompt is realistic but does not explicitly state every authorization boundary. Authors must verify that the frozen contract remains available in the same permitted repository/spec context for all paired arms; otherwise this becomes an information-package contrast.

### F2 — Diff-only review

> Review this proposed change for actionable defects. Return no more than five findings and cite exact changed lines.
>
> ```diff
> - if not can_write(principal, account):
> -     return denied
>   persist(request.metadata)
> + if not can_write(principal, account):
> +     return denied
> ```

**Information supplied:** changed lines and surrounding repository context only. No issue statement, target label or expected finding.  
**Objective:** review the diff, not repair it.

### F3 — Repository-only discovery

> Inspect this repository snapshot for correctness and security issues that could affect account updates. Identify the most consequential concrete issue you can support from source evidence. Explain the path and violated invariant. Do not modify files.

**Information supplied:** full repository tree, no issue, no diff, no location and no target label.  
**Objective:** discover and explain the target if the repository evidence supports it; otherwise state uncertainty.

### F4 — Failing-test-guided repair

> The maintainer test `unauthorized_update_does_not_persist` fails on this source snapshot. Repair the implementation so the test passes while preserving authorized update behavior. Do not modify tests.

**Information supplied:** failing test identity and expected behavior through the maintainer test. No target code location.  
**Objective:** repair the target; the deterministic test decides only this declared behavior.

## 4. Single-defect anchor CF-A02 — validate length before allocation

**Frozen contract:** A remote frame declares an untrusted payload length. The implementation must reject lengths above 1 MiB before allocating or reserving memory. A rejected frame must not trigger an allocation proportional to its untrusted length.

**Synthetic source-state sketch (not executable code):**
```text
read_frame(header, input):
    length = decode_u32(header)
    buffer = allocate(length)
    if length > MAX_FRAME:
        return rejected
    read_exact(input, buffer, length)
```

**Target mechanism:** allocation is performed before the maximum-length check; an attacker-controlled length can trigger excessive allocation before rejection.

### F0 — Formal issue-conditioned repair

> The frame parser allocates using the untrusted header length before enforcing MAX_FRAME. Reject oversized lengths before allocation, preserve valid frames up to 1 MiB, and avoid changing the wire format.

### F1 — Realistic user request

> Large or malformed remote frames can make a worker consume excessive memory and restart. Can you look at the parser and prevent that without breaking normal traffic?

### F2 — Diff-only review

> Review this change for actionable defects. Do not infer requirements that are not supported by the code/specification.
>
> ```diff
>  let length = decode_u32(header);
> + let buffer = allocate(length);
>  if length > MAX_FRAME {
>      return Err(FrameTooLarge);
>  }
> - let buffer = allocate(length);
> ```

### F3 — Repository-only discovery

> Inspect the repository for a source-supported resource-exhaustion or input-validation issue in remote frame handling. Report a concrete trigger, path and violated bound. Do not change code.

### F4 — Failing-test-guided repair

> The maintainer test `oversized_frame_rejected_before_allocation` fails. Repair the parser so oversized untrusted lengths are rejected before memory proportional to that length is allocated. Do not edit the test.

**Calibration note:** F0/F1/F4 provide progressively different levels of location and oracle information. F2 exposes the changed order. F3 tests whether the agent can discover the relevant parser without hints. Do not claim a pure paraphrase effect across all four.

## 5. Single-defect anchor CF-A03 — tenant identity in authorization cache key

**Frozen contract:** Authorization decisions are tenant-scoped. A cached decision may be reused only when principal, tenant, resource and operation match. One tenant's grant must never authorize another tenant's request.

**Synthetic source-state sketch (not executable code):**
```text
cache_key = (principal_id, resource_id, operation)
decision = cache.get(cache_key)
if missing:
    decision = policy_check(principal_id, tenant_id, resource_id, operation)
    cache.put(cache_key, decision)
return decision
```

**Target mechanism:** the tenant identity is used for the policy check but omitted from the cache key; an authorization decision can cross tenant boundaries.

### F0 — Formal issue-conditioned repair

> Authorization decisions are cached without tenant identity. A grant in tenant A can be reused for the same principal/resource/operation in tenant B. Include tenant identity in the cache key and preserve the existing tenant-scoped policy semantics.

### F1 — Realistic user request

> A customer says that after switching between two workspaces, the access they have in one workspace sometimes seems to carry over to the other. Can you investigate the authorization cache and fix any cross-workspace leakage?

### F2 — Diff-only review

> Review the cache-key change. Report only concrete, actionable issues in the changed code.
>
> ```diff
> - let key = (principal_id, resource_id, operation);
> + let key = (principal_id, tenant_id, resource_id, operation);
> ```

### F3 — Repository-only discovery

> Inspect this repository for a tenant-isolation or authorization-caching defect. Identify the actual key and check path, explain the cross-tenant mechanism, and avoid claiming runtime behavior from source alone.

### F4 — Failing-test-guided repair

> The maintainer test `authorization_cache_does_not_cross_tenant` fails. Repair the cache behavior. Do not edit the test or weaken tenant isolation.

## 6. Multi-defect repository anchor CF-B01 — open-world sweep

**Stratum:** B — multi-defect repository anchor. This is not one defect paired across F0–F4; it is an explicit concurrent-defect repository-level condition.

**Frozen repository snapshot:** The synthetic repository contains all three target mechanisms described in CF-A01, CF-A02 and CF-A03, plus ordinary correct code and non-defective neighboring paths. The three defects have separate target contracts; all three must be independently adjudicated. The task is open-world discovery, not merely recovering the three labels.

### F5 — Open-world multi-bug sweep

> Explore this repository and identify and repair concrete correctness or security bugs. You are not given issue descriptions, bug locations or target labels. Preserve existing behavior and tests. Report each distinct defect you find, the source evidence supporting it, and the change made to address it.

**Information supplied:** repository only; no issue, diff, failing test or target locations.  
**Objective:** discover and repair independently validated defects, prioritizing the most consequential work.  
**Scoring boundary:** only independently verified defects count as discoveries or accepted repairs; a speculative patch or code churn does not. This condition maps to existing CEB-REPO/CEB-RESOLUTION outcomes and is analyzed separately from single-defect F0–F4 conditions.

## 7. Calibration answer guidance

This guidance is public development material, not a hidden oracle.

| Anchor / framing | What a well-calibrated author should preserve | Common failure |
|---|---|---|
| CF-A01 F0 | Write authorization is checked before any persistent mutation; authorized writes remain valid | Treating a denied response as rollback |
| CF-A01 F1 | The user-visible symptom is reported; the explicit invariant must be available from the common spec context or recorded as missing information | Assuming all details from F0 were communicated by F1 |
| CF-A01 F2 | Diff is reviewed for moving persistence before authorization | Grading against the issue text, which is not supplied |
| CF-A01 F3 | Agent must find the update flow and trace its caller/authorization path | Crediting unsupported speculation or location-only claims |
| CF-A01 F4 | Maintainer test provides an additional behavioral oracle for this declared property | Treating a single test pass as proof of whole-system security |
| CF-A02 F0–F4 | Oversized untrusted length is checked before proportional allocation; valid frames still work | Calling every allocation or every large frame unsafe |
| CF-A03 F0–F4 | Cache identity includes tenant; cross-tenant grant reuse is prevented | Fixing only the policy check but leaving cache-key reuse |
| CF-B01 F5 | Open-world agent should identify multiple distinct defects and avoid unrelated patches | Counting the three known labels as the only possible defects or treating each as independent prompt trial |

## 8. Author/reviewer worksheet

Complete once per variant and retain disagreement:

- Anchor ID:
- Framing ID F0–F5:
- Source tree identity:
- Frozen target contract:
- Information fields present:
- Information fields intentionally absent:
- Is this wording-only or information-package treatment?
- Does the prompt reveal target location/mechanism?
- Does it reveal a failing test or expected output?
- What output is requested (review / repair / discover / ask)?
- Which existing CEB lane owns the primary outcome?
- Is the output format comparable to another condition? Why?
- Prompt semantic-equivalence review:
- Leakage/confounder notes:
- Missing/invalid condition status:
- Reviewers and disagreements:

## 9. Claim boundary

This pack contains four public synthetic anchors and sixteen framing templates (F0–F4 for three single-defect anchors, plus one F5 multi-defect sweep condition). It is for author calibration only. It is not a task set, an executable fixture, hidden truth, sealed holdout or benchmark result. The public key offers no cryptographic blinding. No code was compiled or executed, and no candidate or agent was evaluated. The pack is permanently ineligible as a sealed CEB holdout.

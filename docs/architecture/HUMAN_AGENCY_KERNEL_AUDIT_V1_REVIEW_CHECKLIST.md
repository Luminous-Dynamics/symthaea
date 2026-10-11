# Human Agency Kernel Audit v1 — Review Checklist

**Review target:** `HUMAN_AGENCY_KERNEL_AUDIT_V1.md`

**Status:** architecture review aid only; non-authorizing; non-qualifying.

The review question is:

> Does HAK-001 preserve the semantic distinctions required to increase human and machine capability without silently converting assistance, scoring, expertise, confidence, or delegation into authority over persons?

## A. Baseline standing vs scoped authority

- [ ] Baseline human standing is not defined as a behavioral/model-derived score.
- [ ] Expertise, reputation, contribution, participation, and model inference remain distinct from human worth.
- [ ] Scoped operational authority may still require domain-specific evidence and qualification.
- [ ] Qualified domain authority does not automatically generalize to other domains.
- [ ] Machine authority is explicitly narrower than the authority legitimately delegated to it.
- [ ] Expiry, revocation, and challenge semantics are required for consequential delegated machine authority.

## B. Consent and delegation

- [ ] Recommendation is distinct from consent.
- [ ] Consent is distinct from delegated authority.
- [ ] Silence, distress, role, identity, personalization, or inferred preference do not silently become consent.
- [ ] Withdrawal/refusal can supersede older positive consent where the owning domain permits withdrawal.
- [ ] Adult delegation, guardianship/youth autonomy, research consent, rescue consent, medical consent, and data consent are not collapsed into one universal semantic type.
- [ ] A future canonical delegation transcript must bind every security-relevant grant field before it becomes authority evidence.

## C. Assistance and capability

- [ ] Immediate task success is distinct from understanding and transferable competence.
- [ ] HAK avoids one universal flourishing/agency score.
- [ ] Outcome dimensions may conflict and missing dimensions remain explicit.
- [ ] Longitudinal measures are included where claims concern learning, dependency, calibration, or value drift.
- [ ] Assistance modes distinguish reflective/scaffolded/collaborative work from bounded delegation.
- [ ] Full automation is not introduced merely by renaming delegation as autopilot.

## D. Epistemics and collective intelligence

- [ ] Explanation is distinct from verifiability.
- [ ] Model confidence is distinct from human reliance calibration.
- [ ] Consensus is distinct from truth.
- [ ] Minority/dissenting positions can be retained without being declared correct merely because they are minority positions.
- [ ] Independent elicitation is preserved before shared AI synthesis where independence is decision-relevant.
- [ ] Collective answer quality does not erase viewpoint diversity, minority information, participant understanding, or downstream learning.

## E. Governance tension

- [ ] The audit fairly recognizes the problems the current sovereign-profile design attempts to solve.
- [ ] It correctly identifies that an eight-dimensional score can still create legitimacy problems if used to determine baseline civic standing.
- [ ] It distinguishes privacy of a score from legitimacy of using the score.
- [ ] It separates democratic/member legitimacy, affected-party standing, domain expertise, evidence quality, and constitutional constraints instead of reducing them to one weighted scalar.
- [ ] It preserves a possible future use for contribution/competence vectors in expertise routing and scoped role qualification.
- [ ] It does not require an immediate destructive migration of Mycelix governance.

## F. Existing-system reuse

- [ ] HAK learns from Symthaea rescue-consent semantics without depending on rescue-domain types.
- [ ] HAK composes with Mycelix reciprocal-accountability infrastructure instead of duplicating it.
- [ ] HAK treats current Mycelix SubPassport semantics as useful but not yet canonical delegation authority.
- [ ] HAK reuses psych-bench as an experimental host without turning benchmark output into civic standing.
- [ ] HAK inherits Symthaea's evidence/authority separation without inheriting qualification from unrelated PRs.

## G. Institutional Linter

- [ ] Linter findings remain advisory.
- [ ] A linter finding is not a moral verdict, legal conclusion, governance decision, or action permission.
- [ ] Every finding retains rule identity, triggering evidence, scope, uncertainty/limitations, and mitigation options.
- [ ] The linter can detect score-to-rights, no-expiry delegation, absent contestability, post-hoc benefit claims, and premature consensus collapse.

## H. Research design

- [ ] Human-benefit claims are eligible for preregistration before confirmatory observation.
- [ ] Research conditions include no-AI / answer / scaffold / collaborate / delegate distinctions where appropriate.
- [ ] Reliance-calibration experiments include incorrect AI outputs rather than evaluating only correct assistance.
- [ ] Delayed unaided transfer can be measured separately from immediate assisted performance.
- [ ] Collective-intelligence studies measure both quality and diversity/independence.
- [ ] Real-participant claims remain outside HAK-001 until suitable ethics/research review and study execution occur.

## I. Architecture boundaries

- [ ] HAK-001 does not create a production Human Agency Kernel crate.
- [ ] HAK-001 does not change runtime behavior or authority.
- [ ] HAK-001 does not claim Mycelix governance already satisfies HAK.
- [ ] HAK-001 does not claim the sovereign profile is useless or must be deleted.
- [ ] HAK-001 does not establish personhood, sentience, or moral-patient status for AI systems.
- [ ] HAK-001 does not inherit SCI-001 or any other PR's qualification.

## J. Exit criterion

Approve HAK-001 as an architecture audit only if the common/non-common map is sufficiently precise that future HAK-002+ PRs can be rejected for semantic collapse even when their code is otherwise correct.

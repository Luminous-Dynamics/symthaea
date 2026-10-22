# Merge Admission Contract v1

Status: source contract only. This document does not claim that GitHub currently enforces the policy described here.

## Purpose

Symthaea already distinguishes workflow execution from qualification and qualification from authority. Merge governance needs the same discipline.

A successful workflow run is evidence that a workflow executed successfully. It is **not** evidence that GitHub prevented a non-compliant change from reaching `main`.

The repository therefore treats these as separate facts:

```text
workflow result
    !=
required status check
    !=
merge enforcement
    !=
source qualification
```

`GOV-ENF-001A` introduces a machine-readable attestor for the third fact only: whether GitHub's live repository configuration proves that a target branch is protected by an admission policy with the required check contexts.

## Required policy

For a merge-admission receipt to report `verified_required_checks`, at least one observable GitHub enforcement mechanism must prove all of the following for the target branch:

1. pull requests are required;
2. every configured required check context supplied to the attestor is required;
3. non-fast-forward / force-push replacement is blocked;
4. branch deletion is blocked;
5. administrators or equivalent bypass actors cannot silently bypass the same policy.

The attestor supports both classic branch protection and repository rulesets. A coarse `protected: true` flag is not sufficient by itself.

The default required check in v1 is:

```text
Governance Check (Class A/B Changes)
```

Additional checks may be supplied with repeated `--require-check` arguments. This allows the policy to ratchet later without changing the receipt schema.

## Visibility is part of the claim

GitHub deliberately permission-gates some enforcement details. The attestor therefore distinguishes **absence** from **unobservable detail**.

For classic branch protection:

- `protected: false` on the branch resource is accepted as an explicit no-protection signal when the repository ruleset list is also observable;
- `protected: true` plus an unreadable protection-detail endpoint is `present_but_unverified`;
- an unreadable detail endpoint is never treated as evidence that required checks or bypass restrictions exist.

For repository rulesets:

- every active ruleset is fetched in full before its target and ref applicability are classified;
- `~DEFAULT_BRANCH` is matched only against the repository's actual live `default_branch` value;
- hidden or absent `bypass_actors` is **not** interpreted as an empty bypass list;
- an active ruleset whose detail cannot be read remains present-but-unverified rather than silently disappearing from the evidence model.

This is intentionally conservative. A lower-privilege token may be able to prove that enforcement exists while being unable to prove that bypass is impossible. In that case the stronger verdict is withheld.

## Verdicts

The attestor emits exactly one configuration verdict.

### `verified_required_checks`

The live API exposed an applicable enforcement mechanism and the observable configuration satisfied the complete requested policy.

This verdict means only that merge admission is configured to require the requested checks under the observed configuration. It does not mean those checks passed for any particular commit.

### `present_but_unverified`

An enforcement mechanism appears to exist, but the complete requested policy could not be proven. Examples include:

- protection exists but required checks differ;
- pull requests are not required;
- force pushes or deletion remain allowed;
- administrator/bypass enforcement is weaker than the contract;
- bypass details are hidden from the caller;
- a protected branch exists but detailed protection state is not readable;
- an active ruleset is listed but its complete detail cannot be observed.

This state must not be promoted to verified enforcement.

### `unenforced`

The live API exposed enough state to show no applicable merge-enforcement mechanism for the target branch.

For example, a branch resource reporting `protected: false` together with an observable empty inherited ruleset list is sufficient evidence for this verdict even if the separate classic-protection detail endpoint rejects the caller: the branch metadata already states that the branch is not protected.

This is a configuration observation, not an accusation about process. Human discipline may still be stronger than the GitHub enforcement layer; the point is that GitHub did not prove it mechanically.

### `indeterminate`

The attestor could not obtain enough live API state to decide whether enforcement exists. Missing repository/branch metadata, unreadable ruleset inventory, malformed responses, or comparable ambiguity fail closed into this state.

Indeterminate is not equivalent to unenforced and must not be represented as verified.

## Receipt

The canonical receipt schema identifier is:

```text
symthaea.merge-admission-receipt.v1
```

A receipt binds at least:

- repository identity;
- target branch;
- live default-branch identity;
- observed branch head SHA when available;
- observation time;
- exact required check names requested by the caller;
- configuration verdict;
- separately reported branch-protection evidence;
- separately reported repository-ruleset evidence;
- bypass observability where relevant;
- reasons for non-verified states.

The head SHA is observational context. A merge-admission receipt is not a source qualification receipt and does not freeze or qualify that source tree.

## Observation mode vs enforcement mode

The attestor has two modes intentionally.

Default mode is observational:

```text
python3 .github/scripts/audit-merge-admission.py \
  --repository Luminous-Dynamics/symthaea \
  --branch main
```

It emits the receipt and exits successfully even when the verdict is not verified. This permits deployment of the evidence mechanism before GitHub settings are changed.

After repository protection/rulesets are configured, enforcement consumers may add:

```text
--require-verified
```

In that mode every non-verified verdict exits non-zero.

The contract deliberately separates these rollout stages so introducing the attestor cannot falsely imply that enforcement already exists.

## Self-test

The implementation includes deterministic offline cases for:

- explicit unprotected branch + empty rulesets -> `unenforced`;
- complete classic branch protection -> `verified_required_checks`;
- protected-but-unreadable detail -> `present_but_unverified`;
- hidden classic-protection bypass state -> `present_but_unverified`;
- active ruleset whose summary omits `target` but whose full detail protects the default branch -> `verified_required_checks`;
- `~DEFAULT_BRANCH` applied to a non-default branch -> not applicable;
- hidden ruleset bypass state -> `present_but_unverified`;
- missing required check -> `present_but_unverified`.

Run with:

```text
python3 .github/scripts/audit-merge-admission.py --self-test
```

Passing these cases establishes only evaluator behavior against the embedded fixtures. It does not establish the live repository's current settings.

## Current rollout boundary

The attestor must not be used as an excuse to make draft governance runner-free until GitHub merge admission is actually configured and independently observed as `verified_required_checks`.

The safe order is:

```text
1. land the observational attestor
2. configure GitHub branch protection or an active repository ruleset
3. capture a verified live receipt
4. require the admission check at merge time
5. only then consider skipping runner-backed governance while a PR remains draft
```

Skipping draft governance before step 2 may reduce runner pressure, but it leaves a ready-to-merge race when no GitHub rule requires the replacement check.

## Nonclaims

A merge-admission receipt does not establish:

- that a PR's source compiles;
- that tests or Clippy passed;
- that qualification executed;
- that scientific evidence is valid;
- that a workflow result is current for another commit;
- that a human or administrator cannot change repository settings after the observation;
- that GitHub itself is infallible.

It establishes only a bounded observation of the live merge-enforcement configuration at receipt time.

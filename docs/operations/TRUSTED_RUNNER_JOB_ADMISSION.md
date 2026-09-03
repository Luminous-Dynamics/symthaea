# Trusted Runner Host-Side Job Admission

## Status

This document describes the staged `ci/nixos-ephemeral-runner-v2` defense-in-depth admission gate for the `symthaea-trusted-cpu-v1` self-hosted capability.

It is **not yet production-qualified**. The Nix module and eval contract are statically reviewed, and the admission shell logic has been exercised separately against positive/negative environment cases, but the pinned Nix evaluation and main-only trusted-runner smoke must still pass on the deployed host.

This document is authoritative for the `trustedHarnessCommit` option and host-side job-admission lifecycle. The broader `GITHUB_ACTIONS_NIXOS_RUNNER.md` remains useful background but predates this fourth host option in some sections.

## Why this gate exists

Repository routing already limits `symthaea-trusted-cpu-v1` to four manually dispatched, tokenless, main-only workflows. Issue #330 additionally requires server-side protection for the trusted `main` root.

Those controls are necessary, but a privileged self-hosted capability benefits from an independent host-owned check as well.

GitHub self-hosted runners support `ACTIONS_RUNNER_HOOK_JOB_STARTED`: the runner invokes the configured script synchronously after assigning the job but before workflow-defined steps. A non-zero exit fails the job before it proceeds.

The v2 Nix module installs a Nix-store-pinned admission hook through that mechanism. The hook is defense in depth; it does not replace GitHub branch/ruleset policy.

## Host configuration

The module now exposes four host-facing options:

- `enable`
- `name`
- `tokenFile`
- `trustedHarnessCommit`

Example:

```nix
services.symthaea-ci-runner = {
  enable = true;
  tokenFile = "/run/secrets/github-runner/symthaea-pat";
  trustedHarnessCommit = "0123456789abcdef0123456789abcdef01234567";
};
```

`trustedHarnessCommit` must be an exact lowercase 40-hex Git commit.

Do not fill this option with a branch name, tag, shortened SHA, mutable alias, or an unreviewed commit.

## Admission contract

Before any workflow step executes, the host-owned hook requires all of the following:

- `GITHUB_REPOSITORY == Luminous-Dynamics/symthaea`;
- `GITHUB_REPOSITORY_ID == 1136141775`;
- `GITHUB_SERVER_URL == https://github.com`;
- `GITHUB_EVENT_NAME == workflow_dispatch`;
- `GITHUB_REF == refs/heads/main`;
- `GITHUB_REF_TYPE == branch`;
- `GITHUB_REF_PROTECTED == true`;
- `GITHUB_SHA == trustedHarnessCommit`;
- `GITHUB_WORKFLOW_SHA == trustedHarnessCommit`;
- `GITHUB_WORKFLOW_REF` is exactly one of:
  - `Luminous-Dynamics/symthaea/.github/workflows/self-hosted-runner-smoke.yml@refs/heads/main`;
  - `Luminous-Dynamics/symthaea/.github/workflows/self-hosted-ai-assurance-foundation-recovery.yml@refs/heads/main`;
  - `Luminous-Dynamics/symthaea/.github/workflows/self-hosted-ai-assurance-budget-recovery.yml@refs/heads/main`;
  - `Luminous-Dynamics/symthaea/.github/workflows/self-hosted-sym-arch-002a-core-recovery.yml@refs/heads/main`.

Any mismatch rejects the job.

The workflow cannot opt out of this hook and cannot convert a failed hook into `continue-on-error` behavior.

## Deliberate fail-closed repinning

The trusted host is intentionally pinned to one reviewed harness commit at a time.

If `main` advances from commit `A` to commit `B`, a host still pinned to `A` rejects jobs from `B` even when the workflow files themselves did not change.

That behavior is intentional.

The update lifecycle is:

1. `main` is protected under the policy required by #330;
2. commit `B` reaches `main` through the reviewed promotion path;
3. review the exact `B` tree, especially trusted workflow/Nix paths;
4. update the host configuration to `trustedHarnessCommit = "B";`;
5. rebuild/switch the NixOS host;
6. verify the runner service and host-side admission hook;
7. run the main-only trusted smoke from exact commit `B`;
8. only after smoke PASS use the frozen recovery workflows.

Do not auto-update this pin from `origin/main`, a webhook, a scheduled pull, or workflow input. A moving pin would defeat the purpose of the host-owned trust root.

## Relationship to #330

The host hook does not make branch protection optional.

GitHub's runner-hook design is an administrator extension point, not a server-side authorization system. The strongest deployment therefore requires both:

```text
GitHub protected root / reviewed promotion
                  +
       host-pinned exact harness
                  +
        fixed workflow allowlist
                  +
       exact frozen target identity
```

If GitHub reports `GITHUB_REF_PROTECTED != true`, the host rejects the job even when every other identity matches.

## Relationship to target-tree recovery

The host admission hook authenticates the **trusted harness job**. It does not itself qualify or authenticate the unmerged research target.

Each recovery workflow separately authenticates its frozen target.

For example, the #186 recovery additionally pins:

- target commit `251c47c5a1f09015ab6c29794df7ac5b6efae373`;
- target tree `1451a57e2dcaf50de6b5b43776364a5af8b78f15`;
- lower base `6ca61356c6ff49b2ba77cae332f006d054d5f84d`;
- hosted AI Assurance workflow blob `35bbddd9d6a7f8ad77073e4ab4faf6722f00deea`;
- target `Cargo.lock` blob `e70e88105e2774dc9c6384fbd86d33a2d0ca8af7`;
- base `Cargo.lock` blob `1d5b6c3dffb474fabef0f8237c6741b4f252bc62`;
- exact changed-path allowlist;
- exact `Cargo.lock` delta shape.

This gives separate trust boundaries for harness admission and target qualification.

## Logic-level negative cases

The admission shell contract has been exercised independently with an allowed case plus rejections for:

- wrong repository;
- unprotected `main`;
- `pull_request` instead of `workflow_dispatch`;
- stale `GITHUB_SHA`;
- stale `GITHUB_WORKFLOW_SHA`;
- unreviewed workflow path.

Those tests establish the shell decision logic only. They do not establish that the NixOS module evaluates, that systemd injects the hook as intended, or that GitHub's deployed runner exposes every expected variable on this pinned runner version.

Those integration claims require the Nix eval and real smoke.

## Required smoke evidence

Before treating host-side admission as qualified, the main-only smoke should demonstrate on the deployed protected harness commit that:

1. the Nix eval test passes;
2. the runner service contains `ACTIONS_RUNNER_HOOK_JOB_STARTED` pointing at the Nix-store hook;
3. an allowed trusted smoke reaches its workflow steps;
4. a controlled negative test using an unapproved identity is rejected before workflow code executes, where a safe GitHub-side test method is available;
5. the source/harness workspace remains immutable;
6. the runner remains ephemeral and tokenless at workflow permission level;
7. no broad scheduling label can reach the host.

## Non-claims

This gate does not:

- make arbitrary PR/fork code safe on the self-hosted machine;
- replace #330 protected-main policy;
- replace exact frozen target/tree/path/dependency checks;
- make the PAT safe to expose to job code;
- provide performance equivalence with GitHub-hosted runners;
- qualify #186, #196, or later assurance work merely because the hook itself passes.

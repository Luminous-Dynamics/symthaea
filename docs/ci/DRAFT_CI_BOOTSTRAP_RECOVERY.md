# Draft CI Bootstrap Recovery

Status: operational recovery protocol

This protocol exists for one failure mode: the scheduled Draft CI Governor needs an Actions runner to cancel excess draft full-CI work, but a sufficiently saturated Actions queue can delay the governor behind the queue it is meant to reduce.

The steady-state governor remains `.github/workflows/draft-ci-governor.yml`. The out-of-band tool `scripts/reconcile-draft-ci-backlog.py` is only the bootstrap/recovery path when runner admission itself is impaired.

## Authority boundary

The recovery tool is intentionally narrower than general Actions administration.

It may cancel only runs of the explicitly named full-CI workflow that are associated with an **open, same-repository, draft pull request**. It does not treat a branch name alone as authority when GitHub has already attached an explicit PR identity to a run. It re-reads the PR immediately before each destructive request.

By default it considers only:

- `queued`
- `requested`
- `waiting`
- `pending`

It does **not** touch `in_progress` work unless `--include-in-progress` is explicitly supplied. It never intentionally cancels ready PRs, closed PRs, fork PRs, push runs, scheduled runs, or deliberate `workflow_dispatch` runs.

Dry-run is the default. `--apply` is required for mutation.

## Required token

Destructive operation requires `GITHUB_TOKEN` with repository Actions write permission and pull-request read permission. Do not write the token into a receipt, command-line argument, checked-in file, or log.

When GitHub CLI is already authenticated, a short-lived shell environment can be used:

```bash
export GITHUB_TOKEN="$(gh auth token)"
```

Unset it after recovery:

```bash
unset GITHUB_TOKEN
```

## Recovery sequence

### 1. Self-test the eligibility theorem

```bash
python3 scripts/reconcile-draft-ci-backlog.py --self-test
```

Do not proceed if the self-test fails.

### 2. Produce a dry-run receipt

```bash
python3 scripts/reconcile-draft-ci-backlog.py \
  --max-cancellations 100 \
  --receipt /tmp/symthaea-draft-ci-dry-run.json
```

Review the receipt before mutation. Every proposed action must be `would_cancel`, and every listed PR must still be intended to remain draft.

For a large backlog, using an authenticated token for dry-run is preferred because GitHub's unauthenticated REST budget is small:

```bash
GITHUB_TOKEN="$(gh auth token)" \
python3 scripts/reconcile-draft-ci-backlog.py \
  --max-cancellations 100 \
  --receipt /tmp/symthaea-draft-ci-dry-run.json
```

### 3. Cancel non-running draft full-CI work only

```bash
GITHUB_TOKEN="$(gh auth token)" \
python3 scripts/reconcile-draft-ci-backlog.py \
  --apply \
  --max-cancellations 100 \
  --receipt /tmp/symthaea-draft-ci-apply.json
```

The tool reserves primary GitHub REST rate budget and stops on 403/429 rather than continuing through a rate-limit condition.

A cancellation request can legitimately return 409 if the run became non-cancellable between discovery and mutation. That is recorded as `already_non_cancellable`, not silently counted as a cancellation.

### 4. Re-count before taking more action

Do not immediately add `--include-in-progress`. First re-run dry mode and inspect whether queued/requested/waiting/pending work has fallen enough for governors and focused qualification jobs to receive runners.

Only if runner admission remains blocked and the remaining in-progress draft full-CI work is explicitly judged disposable should the operator consider:

```bash
GITHUB_TOKEN="$(gh auth token)" \
python3 scripts/reconcile-draft-ci-backlog.py \
  --apply \
  --include-in-progress \
  --max-cancellations 25 \
  --receipt /tmp/symthaea-draft-ci-in-progress-apply.json
```

This is an escalation path, not the default recovery procedure.

## Receipts and claims

The tool emits `symthaea.ci.draft-backlog-bootstrap-recovery.v1` JSON containing:

- repository and workflow identity;
- dry-run vs apply mode;
- included run states;
- number of live full-CI runs examined;
- number eligible under the same-repo draft theorem;
- requested and rate-safe mutation caps;
- rate-limit reserve/reset information;
- cancellation, stale-eligibility, 409, and rate-limit counts;
- per-run actions.

This receipt is **operations evidence only**. It says nothing about scientific qualification, Rust correctness, experiment results, or protected SYM-RSI partitions.

## 2026-09-18 motivating incident

The repository reached a state with 211 queued workflow runs and zero in-progress runs while GitHub's public status page reported Actions operational. A focused `ubuntu-latest` SYM-RSI qualification job was queued with `runner_id = 0`, and a scheduled `ubuntu-slim` Draft CI Governor had previously taken almost two hours from schedule creation to runner execution.

The successful governor's own log showed its eligibility logic was conservative: it examined nine live pull-request CI runs at that time, found one same-repository draft full-CI candidate, re-read the PR, and the cancellation endpoint returned 409 because the target was already non-cancellable.

The incident therefore exposed a bootstrap property rather than an eligibility-theorem failure: a runner-dependent queue governor is insufficient by itself when runner admission is the constrained resource. This out-of-band tool closes that recovery gap while preserving the governor as the normal steady-state mechanism.

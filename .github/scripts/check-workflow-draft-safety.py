
def has_draft_guard(expression: str | None) -> bool:
    return expression is not None and DRAFT_FALSE.search(expression) is not None


def require_ready_event(path: Path, pr_block: list[str]) -> None:
    if not any("ready_for_review" in line for line in pr_block):
        raise SafetyError(
            f"{path}: runner-capable pull_request workflow must include ready_for_review"
        )


def validate_generic(path: Path, text: str, pr_block: list[str]) -> tuple[int, int]:
    jobs = parse_jobs(text)
    runner_jobs = 0
    draft_guarded = 0
    pr_runner_jobs = 0
    for job, block in jobs.items():
        if not has_runner_allocation(block):
            continue
        runner_jobs += 1
        expression = job_level_if_expression(block, job)
        if explicitly_excludes_pull_request(expression):
            continue

        pr_runner_jobs += 1
        if not has_draft_guard(expression):
            raise SafetyError(
                f"{path}: runner-capable job {job!r} lacks a job-level "
                "pull_request draft == false guard or explicit non-PR event guard"
            )
        draft_guarded += 1

    if pr_runner_jobs:
        require_ready_event(path, pr_block)
    return runner_jobs, draft_guarded


def require_contains(expression: str | None, needle: str, label: str) -> None:
    if expression is None or needle not in expression:
        raise SafetyError(f"benchmarks.yml {label} lost required expression {needle!r}")


def validate_benchmarks(text: str) -> tuple[int, int]:
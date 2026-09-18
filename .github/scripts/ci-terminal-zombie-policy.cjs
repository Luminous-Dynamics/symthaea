'use strict';

function parseTime(value) {
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : null;
}

function classifyTerminalZombie({ run, jobs, nowMs, minimumStaleMinutes = 30 }) {
  const reasons = [];
  const staleMs = minimumStaleMinutes * 60 * 1000;

  if (!run || typeof run !== 'object') reasons.push('missing_run');
  if (!Array.isArray(jobs)) reasons.push('jobs_not_array');
  if (reasons.length) return { eligible: false, reasons };

  if (run.name !== 'CI' || run.path !== '.github/workflows/ci.yml') reasons.push('wrong_workflow');
  if (run.event !== 'pull_request') reasons.push('wrong_event');
  if (run.status !== 'in_progress') reasons.push('parent_not_in_progress');
  if (jobs.length === 0) reasons.push('zero_jobs');

  let latestCompletion = -Infinity;
  for (const job of jobs) {
    if (!job || job.status !== 'completed') {
      reasons.push(`nonterminal_job:${job?.id ?? 'unknown'}:${job?.status ?? 'missing'}`);
      continue;
    }
    const completed = parseTime(job.completed_at);
    if (completed === null) {
      reasons.push(`invalid_completed_at:${job.id ?? 'unknown'}`);
      continue;
    }
    latestCompletion = Math.max(latestCompletion, completed);
  }

  const updated = parseTime(run.updated_at);
  if (updated === null) reasons.push('invalid_parent_updated_at');

  if (Number.isFinite(latestCompletion) && nowMs - latestCompletion < staleMs) {
    reasons.push('latest_child_completion_too_recent');
  }
  if (updated !== null && nowMs - updated < staleMs) {
    reasons.push('parent_update_too_recent');
  }

  return {
    eligible: reasons.length === 0,
    reasons,
    latestCompletionMs: Number.isFinite(latestCompletion) ? latestCompletion : null,
    parentUpdatedMs: updated,
  };
}

function selfTest() {
  const now = Date.parse('2026-09-18T12:00:00Z');
  const old = '2026-09-18T10:00:00Z';
  const recent = '2026-09-18T11:50:00Z';
  const baseRun = {
    name: 'CI',
    path: '.github/workflows/ci.yml',
    event: 'pull_request',
    status: 'in_progress',
    updated_at: old,
  };
  const done = [
    { id: 1, status: 'completed', completed_at: old },
    { id: 2, status: 'completed', completed_at: old },
  ];

  const cases = [
    ['stale_all_terminal', baseRun, done, true],
    ['one_queued_child', baseRun, [...done, { id: 3, status: 'queued', completed_at: null }], false],
    ['zero_jobs', baseRun, [], false],
    ['parent_completed', { ...baseRun, status: 'completed' }, done, false],
    ['wrong_event', { ...baseRun, event: 'push' }, done, false],
    ['recent_child_completion', baseRun, [{ id: 1, status: 'completed', completed_at: recent }], false],
    ['recent_parent_update', { ...baseRun, updated_at: recent }, done, false],
    ['invalid_child_timestamp', baseRun, [{ id: 1, status: 'completed', completed_at: 'not-a-date' }], false],
    ['wrong_workflow_identity', { ...baseRun, name: 'Not CI', path: '.github/workflows/other.yml' }, done, false],
  ];

  const results = [];
  for (const [name, run, jobs, expected] of cases) {
    const got = classifyTerminalZombie({ run, jobs, nowMs: now, minimumStaleMinutes: 30 });
    if (got.eligible !== expected) {
      throw new Error(`${name}: expected eligible=${expected}, got ${got.eligible}; reasons=${got.reasons.join(',')}`);
    }
    results.push({ name, pass: true, eligible: got.eligible });
  }
  return { pass: true, cases: results.length, results };
}

if (require.main === module) {
  process.stdout.write(`${JSON.stringify(selfTest(), null, 2)}\n`);
}

module.exports = { classifyTerminalZombie, selfTest };

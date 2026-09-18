'use strict';

const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');

const GHOST_MIN_AGE_MINUTES = 30;
const GHOST_MIN_AGE_MS = GHOST_MIN_AGE_MINUTES * 60 * 1000;

function isGhostPrequeue({ status, jobCount, createdAt, updatedAt, nowMs = Date.now() }) {
  if (status !== 'queued' || jobCount !== 0) return false;

  const createdMs = Date.parse(createdAt);
  const updatedMs = Date.parse(updatedAt);
  if (!Number.isFinite(createdMs) || !Number.isFinite(updatedMs) || !Number.isFinite(nowMs)) {
    return false;
  }

  return (
    nowMs - createdMs >= GHOST_MIN_AGE_MS &&
    nowMs - updatedMs >= GHOST_MIN_AGE_MS
  );
}

function runSelfTest() {
  const contractPath = path.resolve(
    __dirname,
    '../rel/draft-ci-ghost-prequeue-contract.json',
  );
  const contract = JSON.parse(fs.readFileSync(contractPath, 'utf8'));

  assert.equal(
    contract.schema,
    'symthaea.ci.draft-governor-ghost-prequeue-contract.v1',
  );
  assert.equal(contract.authority, 'RunnerPlaneOnly');
  assert.equal(contract.ghost_prequeue_requires.fresh_run_status, 'queued');
  assert.equal(contract.ghost_prequeue_requires.instantiated_job_count, 0);
  assert.equal(
    contract.ghost_prequeue_requires.minimum_age_minutes,
    GHOST_MIN_AGE_MINUTES,
  );
  assert.equal(
    contract.ghost_prequeue_requires.minimum_since_update_minutes,
    GHOST_MIN_AGE_MINUTES,
  );
  assert.equal(contract.ghost_prequeue_requires.normal_cancel_result, 409);
  assert.equal(
    contract.ghost_prequeue_requires.same_repo_open_draft_revalidated_immediately_before_mutation,
    true,
  );
  assert.equal(contract.recovery.force_cancel_attempts_per_reconciliation, 1);
  assert.equal(contract.recovery.force_cancel_409_is_terminal_for_that_reconciliation, true);
  assert.equal(contract.scientific_authority, false);
  assert.equal(contract.qualification_authority, false);

  const nowMs = Date.parse('2026-09-18T09:00:00Z');
  const isoMinutesAgo = minutes => new Date(nowMs - minutes * 60 * 1000).toISOString();
  const candidate = overrides => ({
    status: 'queued',
    jobCount: 0,
    createdAt: isoMinutesAgo(31),
    updatedAt: isoMinutesAgo(31),
    nowMs,
    ...overrides,
  });

  const cases = [
    ['stale queued zero-job record', candidate({}), true],
    ['exact 30-minute boundary', candidate({
      createdAt: isoMinutesAgo(30),
      updatedAt: isoMinutesAgo(30),
    }), true],
    ['real queued job exists', candidate({ jobCount: 1 }), false],
    ['not queued', candidate({ status: 'in_progress' }), false],
    ['too newly created', candidate({ createdAt: isoMinutesAgo(29) }), false],
    ['recent metadata update', candidate({ updatedAt: isoMinutesAgo(5) }), false],
    ['invalid creation timestamp', candidate({ createdAt: 'not-a-time' }), false],
    ['invalid update timestamp', candidate({ updatedAt: 'not-a-time' }), false],
    ['negative job count is not zero', candidate({ jobCount: -1 }), false],
  ];

  for (const [name, input, expected] of cases) {
    assert.equal(isGhostPrequeue(input), expected, name);
  }

  process.stdout.write(`ghost-prequeue policy self-test: ${cases.length}/${cases.length} PASS\n`);
}

if (require.main === module) {
  runSelfTest();
}

module.exports = {
  GHOST_MIN_AGE_MINUTES,
  GHOST_MIN_AGE_MS,
  isGhostPrequeue,
  runSelfTest,
};

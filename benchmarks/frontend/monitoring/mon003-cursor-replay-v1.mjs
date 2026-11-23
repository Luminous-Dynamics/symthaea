#!/usr/bin/env node
// SFEB-MON-003 source cursor / replay reference model.
// In-memory deterministic model only; does not qualify a broker or source DB.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

function canonical(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}';
}
const digest = value => createHash('sha256').update(canonical(value)).digest('hex');
function fail(code, message) { const error = new Error(message); error.code = code; throw error; }

class PartitionedEventJournal {
  constructor() {
    this.partitions = new Map();
    this.eventIdentity = new Map();
  }

  #partitionKey(source, partition) { return source + '\n' + partition; }

  #getPartition(source, partition) {
    const key = this.#partitionKey(source, partition);
    let state = this.partitions.get(key);
    if (!state) {
      state = { source, partition, committedSequence: 0, highWatermark: 0, pending: new Map(), committed: new Map(), committedEvents: [], forked: false };
      this.partitions.set(key, state);
    }
    return state;
  }

  receive(event) {
    if (!event || typeof event !== 'object' || Array.isArray(event)) fail('INVALID_EVENT', 'event must be an object');
    const { source, partition, id, sequence, payload, occurredAt } = event;
    if (typeof source !== 'string' || !source || typeof partition !== 'string' || !partition || typeof id !== 'string' || !id ||
        !Number.isSafeInteger(sequence) || sequence < 1 || !Number.isFinite(occurredAt) || payload === undefined) {
      fail('INVALID_EVENT', 'source, partition, id, positive sequence, occurredAt and payload are required');
    }
    const eventDigest = digest({ source, partition, id, sequence, payload, occurredAt });
    const identity = source + '\n' + id;
    const priorIdentity = this.eventIdentity.get(identity);
    if (priorIdentity) {
      if (priorIdentity.digest !== eventDigest) fail('EVENT_ID_REUSE', 'same source+id was replayed with different content or sequence');
      const p = this.#getPartition(source, partition);
      if (sequence <= p.committedSequence) return { status: 'duplicate_committed', committedSequence: p.committedSequence };
      const pending = p.pending.get(sequence);
      if (pending && pending.digest === eventDigest) return { status: 'duplicate_pending', committedSequence: p.committedSequence };
      fail('EVENT_ID_REUSE', 'event identity moved to a different cursor position');
    }

    const state = this.#getPartition(source, partition);
    if (state.forked) fail('PARTITION_QUARANTINED', 'partition is quarantined after a sequence fork');
    if (sequence <= state.committedSequence) {
      const committed = state.committed.get(sequence);
      if (!committed || committed.digest !== eventDigest) {
        state.forked = true;
        fail('CURSOR_ROLLBACK_OR_FORK', 'event conflicts with an already committed source sequence');
      }
      this.eventIdentity.set(identity, { digest: eventDigest, source, partition, sequence });
      return { status: 'duplicate_committed', committedSequence: state.committedSequence };
    }

    const pendingAtSequence = state.pending.get(sequence);
    if (pendingAtSequence) {
      if (pendingAtSequence.digest === eventDigest) return { status: 'duplicate_pending', committedSequence: state.committedSequence };
      state.forked = true;
      fail('SEQUENCE_FORK', 'two different events claim the same source partition sequence');
    }

    this.eventIdentity.set(identity, { digest: eventDigest, source, partition, sequence });
    state.pending.set(sequence, { event: structuredClone(event), digest: eventDigest });
    const newlyCommitted = [];
    while (state.pending.has(state.committedSequence + 1)) {
      const nextSequence = state.committedSequence + 1;
      const item = state.pending.get(nextSequence);
      state.pending.delete(nextSequence);
      state.committedSequence = nextSequence;
      state.committed.set(nextSequence, item);
      state.committedEvents.push(item.event);
      newlyCommitted.push(item.event.id);
    }
    const pendingSequences = [...state.pending.keys()].sort((a, b) => a - b);
    const gap = pendingSequences.length > 0 && pendingSequences[0] > state.committedSequence + 1
      ? { missingFrom: state.committedSequence + 1, observedNext: pendingSequences[0] }
      : null;
    return { status: newlyCommitted.length ? 'committed' : 'buffered_gap', newlyCommitted, committedSequence: state.committedSequence, gap };
  }

  observeHighWatermark({ source, partition, sequence }) {
    if (typeof source !== 'string' || !source || typeof partition !== 'string' || !partition || !Number.isSafeInteger(sequence) || sequence < 0) {
      fail('INVALID_HIGH_WATERMARK', 'high watermark fields are invalid');
    }
    const state = this.#getPartition(source, partition);
    if (sequence < state.committedSequence) {
      state.forked = true;
      fail('SOURCE_ROLLBACK', 'source high watermark is behind the committed cursor');
    }
    state.highWatermark = Math.max(state.highWatermark, sequence);
    const next = state.committedSequence + 1;
    const pending = [...state.pending.keys()].sort((a, b) => a - b);
    const missingTo = pending.length ? pending[0] - 1 : state.highWatermark;
    const gap = next <= state.highWatermark && (pending.length === 0 || pending[0] > next)
      ? { missingFrom: next, missingTo: Math.max(next, missingTo), observedNext: pending[0] ?? null }
      : null;
    return { committedSequence: state.committedSequence, highWatermark: state.highWatermark, gap,
      caughtUp: !state.forked && state.committedSequence >= state.highWatermark && state.pending.size === 0 };
  }

  snapshot(source, partition) {
    const state = this.#getPartition(source, partition);
    return {
      source: state.source, partition: state.partition,
      committedSequence: state.committedSequence,
      highWatermark: state.highWatermark,
      pendingSequences: [...state.pending.keys()].sort((a, b) => a - b),
      committedEventIds: state.committedEvents.map(event => event.id),
      forked: state.forked,
      caughtUp: !state.forked && state.committedSequence >= state.highWatermark && state.pending.size === 0,
    };
  }

  // Only this committed contiguous prefix can be used as source-event evidence.
  eligibleEvents(source, partition) {
    return this.#getPartition(source, partition).committedEvents.map(event => structuredClone(event));
  }
}

const cases = [];
function test(name, fn) { fn(); cases.push({ name, result: 'pass' }); }
function expectCode(code, fn) { assert.throws(fn, error => error && error.code === code, 'expected ' + code); }
const e = (source, partition, id, sequence, payload = { value: sequence }, occurredAt = sequence) => ({ source, partition, id, sequence, payload, occurredAt });

test('out-of-order event is buffered and never advances the contiguous cursor', () => {
  const j = new PartitionedEventJournal();
  const r = j.receive(e('urn:src:A', 'p0', 'evt-3', 3));
  assert.equal(r.status, 'buffered_gap');
  assert.equal(r.committedSequence, 0);
  assert.deepEqual(j.eligibleEvents('urn:src:A', 'p0'), []);
});

test('missing replayed events close the gap and drain the pending suffix', () => {
  const j = new PartitionedEventJournal();
  j.receive(e('urn:src:A', 'p0', 'evt-3', 3));
  j.receive(e('urn:src:A', 'p0', 'evt-1', 1));
  const r = j.receive(e('urn:src:A', 'p0', 'evt-2', 2));
  assert.equal(r.committedSequence, 3);
  assert.deepEqual(j.eligibleEvents('urn:src:A', 'p0').map(x => x.id), ['evt-1', 'evt-2', 'evt-3']);
});

test('identical duplicate pending delivery does not create another event', () => {
  const j = new PartitionedEventJournal(); const event = e('urn:src:A', 'p0', 'evt-2', 2);
  assert.equal(j.receive(event).status, 'buffered_gap');
  assert.equal(j.receive(event).status, 'duplicate_pending');
  assert.equal(j.snapshot('urn:src:A', 'p0').pendingSequences.length, 1);
});

test('identical duplicate committed delivery does not advance cursor or duplicate evidence', () => {
  const j = new PartitionedEventJournal(); const event = e('urn:src:A', 'p0', 'evt-1', 1);
  j.receive(event); const duplicate = j.receive(event);
  assert.equal(duplicate.status, 'duplicate_committed');
  assert.equal(j.snapshot('urn:src:A', 'p0').committedSequence, 1);
  assert.equal(j.eligibleEvents('urn:src:A', 'p0').length, 1);
});

test('same CloudEvents source and id with changed payload is rejected', () => {
  const j = new PartitionedEventJournal(); j.receive(e('urn:src:A', 'p0', 'evt-1', 1));
  expectCode('EVENT_ID_REUSE', () => j.receive(e('urn:src:A', 'p0', 'evt-1', 1, { value: 999 })));
});

test('same event id from different sources remains distinct', () => {
  const j = new PartitionedEventJournal();
  j.receive(e('urn:src:A', 'p0', 'same-id', 1));
  j.receive(e('urn:src:B', 'p0', 'same-id', 1));
  assert.equal(j.eligibleEvents('urn:src:A', 'p0').length, 1);
  assert.equal(j.eligibleEvents('urn:src:B', 'p0').length, 1);
});

test('partition progress is isolated and cannot advance another partition', () => {
  const j = new PartitionedEventJournal();
  j.receive(e('urn:src:A', 'p0', 'p0-e1', 1));
  j.receive(e('urn:src:A', 'p1', 'p1-e2', 2));
  assert.equal(j.snapshot('urn:src:A', 'p0').committedSequence, 1);
  assert.equal(j.snapshot('urn:src:A', 'p1').committedSequence, 0);
  assert.deepEqual(j.snapshot('urn:src:A', 'p1').pendingSequences, [2]);
});

test('two event identities claiming the same sequence quarantine the partition', () => {
  const j = new PartitionedEventJournal(); j.receive(e('urn:src:A', 'p0', 'evt-a', 2));
  expectCode('SEQUENCE_FORK', () => j.receive(e('urn:src:A', 'p0', 'evt-b', 2)));
  assert.equal(j.snapshot('urn:src:A', 'p0').forked, true);
  expectCode('PARTITION_QUARANTINED', () => j.receive(e('urn:src:A', 'p0', 'evt-1', 1)));
});

test('high watermark exposes a cursor gap rather than declaring caught up', () => {
  const j = new PartitionedEventJournal(); j.receive(e('urn:src:A', 'p0', 'evt-3', 3));
  const r = j.observeHighWatermark({ source: 'urn:src:A', partition: 'p0', sequence: 3 });
  assert.deepEqual(r.gap, { missingFrom: 1, missingTo: 2, observedNext: 3 });
  assert.equal(r.caughtUp, false);
});

test('cursor becomes caught up only after every missing sequence is replayed', () => {
  const j = new PartitionedEventJournal();
  j.receive(e('urn:src:A', 'p0', 'evt-3', 3));
  j.observeHighWatermark({ source: 'urn:src:A', partition: 'p0', sequence: 3 });
  j.receive(e('urn:src:A', 'p0', 'evt-1', 1));
  j.receive(e('urn:src:A', 'p0', 'evt-2', 2));
  const r = j.observeHighWatermark({ source: 'urn:src:A', partition: 'p0', sequence: 3 });
  assert.equal(r.caughtUp, true);
  assert.equal(r.committedSequence, 3);
});

test('source high watermark rollback quarantines the partition', () => {
  const j = new PartitionedEventJournal(); j.receive(e('urn:src:A', 'p0', 'evt-1', 1));
  expectCode('SOURCE_ROLLBACK', () => j.observeHighWatermark({ source: 'urn:src:A', partition: 'p0', sequence: 0 }));
  assert.equal(j.snapshot('urn:src:A', 'p0').forked, true);
});

test('committed sequence cannot be rebound to a new event identity', () => {
  const j = new PartitionedEventJournal(); j.receive(e('urn:src:A', 'p0', 'evt-1', 1));
  expectCode('CURSOR_ROLLBACK_OR_FORK', () => j.receive(e('urn:src:A', 'p0', 'attacker-event', 1)));
  assert.equal(j.snapshot('urn:src:A', 'p0').forked, true);
});

test('high watermark without replayed source events yields no eligible events', () => {
  const j = new PartitionedEventJournal();
  j.observeHighWatermark({ source: 'urn:src:A', partition: 'p0', sequence: 9 });
  assert.deepEqual(j.eligibleEvents('urn:src:A', 'p0'), []);
  assert.equal(j.snapshot('urn:src:A', 'p0').caughtUp, false);
});

console.log(JSON.stringify({
  schema: 'sfeb-mon003-cursor-replay-report-v1',
  status: 'passed',
  deterministic_cases: cases.length,
  cases,
  invariants: {
    no_cursor_advancement_past_gap: 'pass',
    event_identity_is_source_scoped: 'pass',
    sequence_forks_fail_closed: 'pass',
    partition_cursors_are_isolated: 'pass',
    high_watermark_is_not_event_evidence: 'pass',
    only_contiguous_committed_prefix_is_eligible: 'pass',
  },
  limitations: [
    'Pure in-memory cursor/replay reference model; no real broker, database cursor, retention or outbox transaction is exercised.',
    'Source and partition sequence are trusted model inputs; signature and source identity authentication remain pending.',
    'The model is single-threaded and does not prove concurrent cursor update atomicity.',
  ],
}, null, 2));

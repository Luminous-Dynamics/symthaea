#!/usr/bin/env node
// SFEB-MON-003 partitioned source cursor/replay reference model.
// In-memory deterministic model only; no real broker/database qualification.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
const canon = v => v === null || typeof v !== 'object' ? JSON.stringify(v) : Array.isArray(v) ? '[' + v.map(canon).join(',') + ']' : '{' + Object.keys(v).sort().map(k => JSON.stringify(k) + ':' + canon(v[k])).join(',') + '}';
const hash = v => createHash('sha256').update(canon(v)).digest('hex');
const fail = (code, message) => { const e = new Error(message); e.code = code; throw e; };

class PartitionJournal {
  constructor() { this.parts = new Map(); this.ids = new Map(); }
  key(source, partition) { return source + '\n' + partition; }
  part(source, partition) {
    const key = this.key(source, partition);
    if (!this.parts.has(key)) this.parts.set(key, { source, partition, cursor: 0, high: 0, pending: new Map(), seqs: new Map(), committed: [], forked: false });
    return this.parts.get(key);
  }
  ingest(e) {
    if (!e || typeof e.source !== 'string' || !e.source || typeof e.partition !== 'string' || !e.partition ||
        typeof e.id !== 'string' || !e.id || !Number.isSafeInteger(e.seq) || e.seq < 1 || !Number.isFinite(e.at) || e.payload === undefined) fail('INVALID_EVENT', 'invalid event envelope');
    const p = this.part(e.source, e.partition), idKey = e.source + '\n' + e.id, d = hash(e);
    if (p.forked) fail('PARTITION_QUARANTINED', 'partition is quarantined');
    const prior = this.ids.get(idKey);
    if (prior) {
      if (prior.digest !== d || prior.partition !== e.partition || prior.seq !== e.seq) { p.forked = true; this.part(e.source, prior.partition).forked = true; fail('EVENT_ID_REUSE', 'source+id identity changed content or position'); }
      if (e.seq <= p.cursor) return { status: 'duplicate_committed', cursor: p.cursor };
      if (p.pending.has(e.seq)) return { status: 'duplicate_pending', cursor: p.cursor };
      p.forked = true; fail('EVENT_ID_REUSE', 'known event identity is absent from cursor/pending state');
    }
    if (e.seq <= p.cursor) { p.forked = true; fail('CURSOR_ROLLBACK_OR_FORK', 'new event attempts to rewrite committed sequence'); }
    if (p.seqs.has(e.seq)) { p.forked = true; fail('SEQUENCE_FORK', 'two event identities claim one partition sequence'); }
    this.ids.set(idKey, { digest: d, partition: e.partition, seq: e.seq });
    p.seqs.set(e.seq, d); p.pending.set(e.seq, structuredClone(e));
    const committedNow = [];
    while (p.pending.has(p.cursor + 1)) {
      const next = p.pending.get(p.cursor + 1); p.pending.delete(p.cursor + 1);
      p.cursor += 1; p.committed.push(next); committedNow.push(next.id);
    }
    const pendingSeqs = [...p.pending.keys()].sort((a,b)=>a-b);
    const gap = pendingSeqs.length && pendingSeqs[0] > p.cursor + 1 ? { from: p.cursor + 1, nextObserved: pendingSeqs[0] } : null;
    return { status: committedNow.length ? 'committed' : 'buffered_gap', cursor: p.cursor, committedNow, gap };
  }
  watermark({ source, partition, high }) {
    if (typeof source !== 'string' || !source || typeof partition !== 'string' || !partition || !Number.isSafeInteger(high) || high < 0) fail('INVALID_WATERMARK', 'invalid high watermark');
    const p = this.part(source, partition);
    if (high < p.cursor || high < p.high) { p.forked = true; fail('SOURCE_ROLLBACK', 'high watermark regressed'); }
    p.high = high;
    const pending = [...p.pending.keys()].sort((a,b)=>a-b);
    const gap = p.cursor < p.high && (!pending.length || pending[0] > p.cursor + 1)
      ? { from: p.cursor + 1, to: Math.max(p.cursor + 1, (pending[0] ?? p.high) - 1), nextObserved: pending[0] ?? null } : null;
    return { cursor: p.cursor, high: p.high, gap, caughtUp: !p.forked && p.cursor >= p.high && !p.pending.size };
  }
  snapshot(source, partition) {
    const p = this.part(source, partition);
    return { cursor: p.cursor, high: p.high, pending: [...p.pending.keys()].sort((a,b)=>a-b), committed: p.committed.map(x=>x.id), forked: p.forked, caughtUp: !p.forked && p.cursor >= p.high && !p.pending.size };
  }
  eligible(source, partition) { return this.part(source, partition).committed.map(x=>structuredClone(x)); }
}

const results=[]; const test=(name,fn)=>{fn();results.push({name,result:'pass'});};
const expect=(code,fn)=>assert.throws(fn,e=>e&&e.code===code,'expected '+code);
const ev=(source,partition,id,seq,payload={n:seq},at=seq)=>({source,partition,id,seq,payload,at});

test('out-of-order event never advances over a missing prefix',()=>{const j=new PartitionJournal();const r=j.ingest(ev('sA','p0','e3',3));assert.equal(r.cursor,0);assert.deepEqual(j.eligible('sA','p0'),[]);});
test('replay of missing events drains the pending suffix in order',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','e3',3));j.ingest(ev('sA','p0','e1',1));j.ingest(ev('sA','p0','e2',2));assert.deepEqual(j.eligible('sA','p0').map(x=>x.id),['e1','e2','e3']);});
test('identical duplicate pending event is idempotent',()=>{const j=new PartitionJournal(),x=ev('sA','p0','e2',2);j.ingest(x);assert.equal(j.ingest(x).status,'duplicate_pending');assert.deepEqual(j.snapshot('sA','p0').pending,[2]);});
test('identical duplicate committed event adds no evidence',()=>{const j=new PartitionJournal(),x=ev('sA','p0','e1',1);j.ingest(x);assert.equal(j.ingest(x).status,'duplicate_committed');assert.equal(j.eligible('sA','p0').length,1);});
test('same source+id with changed payload is rejected',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','e1',1));expect('EVENT_ID_REUSE',()=>j.ingest(ev('sA','p0','e1',1,{n:999})));});
test('source+id reused on another partition quarantines both partitions',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','shared',1));expect('EVENT_ID_REUSE',()=>j.ingest(ev('sA','p1','shared',1)));assert.equal(j.snapshot('sA','p0').forked,true);assert.equal(j.snapshot('sA','p1').forked,true);});
test('same event id from two distinct sources is not a collision',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','same',1));j.ingest(ev('sB','p0','same',1));assert.equal(j.eligible('sA','p0').length,1);assert.equal(j.eligible('sB','p0').length,1);});
test('partition progress is isolated',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','p0e1',1));j.ingest(ev('sA','p1','p1e2',2));assert.equal(j.snapshot('sA','p0').cursor,1);assert.equal(j.snapshot('sA','p1').cursor,0);});
test('two event identities at one sequence quarantine the partition',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','a',2));expect('SEQUENCE_FORK',()=>j.ingest(ev('sA','p0','b',2)));assert.equal(j.snapshot('sA','p0').forked,true);expect('PARTITION_QUARANTINED',()=>j.ingest(ev('sA','p0','e1',1)));});
test('high watermark exposes a missing cursor interval',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','e3',3));const r=j.watermark({source:'sA',partition:'p0',high:3});assert.deepEqual(r.gap,{from:1,to:2,nextObserved:3});assert.equal(r.caughtUp,false);});
test('replaying every missing sequence closes high-watermark gap',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','e3',3));j.watermark({source:'sA',partition:'p0',high:3});j.ingest(ev('sA','p0','e1',1));j.ingest(ev('sA','p0','e2',2));assert.equal(j.watermark({source:'sA',partition:'p0',high:3}).caughtUp,true);});
test('high watermark cannot regress below committed cursor',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','e1',1));expect('SOURCE_ROLLBACK',()=>j.watermark({source:'sA',partition:'p0',high:0}));});
test('high watermark cannot regress even while cursor is behind',()=>{const j=new PartitionJournal();j.watermark({source:'sA',partition:'p0',high:9});expect('SOURCE_ROLLBACK',()=>j.watermark({source:'sA',partition:'p0',high:8}));});
test('new identity cannot rewrite a committed sequence',()=>{const j=new PartitionJournal();j.ingest(ev('sA','p0','e1',1));expect('CURSOR_ROLLBACK_OR_FORK',()=>j.ingest(ev('sA','p0','attacker',1)));});
test('high watermark without source events never creates eligible evidence',()=>{const j=new PartitionJournal();j.watermark({source:'sA',partition:'p0',high:9});assert.deepEqual(j.eligible('sA','p0'),[]);assert.equal(j.snapshot('sA','p0').caughtUp,false);});
test('invalid sequence and invalid watermark fail closed',()=>{const j=new PartitionJournal();expect('INVALID_EVENT',()=>j.ingest(ev('sA','p0','zero',0)));expect('INVALID_WATERMARK',()=>j.watermark({source:'sA',partition:'p0',high:-1}));});

console.log(JSON.stringify({schema:'sfeb-mon003-cursor-replay-report-v1',status:'passed',deterministic_cases:results.length,cases:results,invariants:{no_cursor_advance_past_gap:'pass',source_scoped_identity:'pass',sequence_fork_quarantine:'pass',partition_cursor_isolation:'pass',monotonic_high_watermark:'pass',watermark_is_not_event_evidence:'pass',only_contiguous_prefix_is_eligible:'pass'},limitations:['Pure in-memory cursor/replay model; no real broker, database cursor, retention or outbox transaction is exercised.','Source and partition sequence are trusted model inputs; signature and source identity authentication remain pending.','Single-threaded reference model; it does not prove concurrent cursor-update atomicity.']},null,2));

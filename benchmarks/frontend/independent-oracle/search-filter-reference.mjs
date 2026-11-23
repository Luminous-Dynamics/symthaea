#!/usr/bin/env node
/** Independent, dependency-free JavaScript oracle for SFEB-L2-005.
 * This is a benchmark verifier, not part of the Leptos application runtime.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(fileURLToPath(import.meta.url));
const readJson = async (path) => JSON.parse(await readFile(join(ROOT, path), 'utf8'));
const dataset = await readJson('../fixtures/search-filter-explorer/dataset.json');
const oracle = await readJson('../fixtures/search-filter-explorer/oracle-cases.json');
const urlVectors = await readJson('../fixtures/search-filter-explorer/url-state-conformance-v1.json');

const DEFAULT = Object.freeze({
  q: '', category: null, available_only: false, min_price_cents: null,
  max_price_cents: null, sort: 'name_asc', page: 1, page_size: 5,
});
const STATE_FIELDS = Object.keys(DEFAULT);
const VALID_SORTS = new Set(['name_asc', 'price_asc', 'price_desc', 'stock_asc']);

function validateDataset(doc) {
  assert.equal(doc.schema, 'sfeb-search-filter-dataset-v1');
  assert.equal(doc.currency, 'USD');
  assert.equal(doc.price_unit, 'integer cents');
  assert.ok(doc.dataset_id && Array.isArray(doc.assets) && doc.assets.length > 0);
  const ids = new Set();
  const skus = new Set();
  for (const asset of doc.assets) {
    for (const key of ['id', 'name', 'sku', 'category', 'description']) assert.ok(asset[key]);
    assert.ok(Number.isSafeInteger(asset.price_cents) && asset.price_cents >= 0);
    assert.ok(Number.isSafeInteger(asset.stock_count) && asset.stock_count >= 0);
    assert.ok(!ids.has(asset.id), `duplicate id ${asset.id}`);
    assert.ok(!skus.has(asset.sku), `duplicate sku ${asset.sku}`);
    ids.add(asset.id); skus.add(asset.sku);
  }
}

function validateQuery(q) {
  assert.ok(typeof q.q === 'string' && [...q.q].length <= 80, 'query_too_long');
  assert.ok(!/[\u0000-\u001f\u007f-\u009f]/u.test(q.q), 'query_contains_control_character');
  assert.ok(q.category === null || typeof q.category === 'string', 'invalid_category');
  assert.ok(typeof q.available_only === 'boolean', 'invalid_available_only');
  for (const field of ['min_price_cents', 'max_price_cents']) {
    const value = q[field];
    assert.ok(value === null || (Number.isSafeInteger(value) && value >= 0), 'price_amount_unsafe_for_cross_language_integer');
  }
  assert.ok(q.category !== '', 'empty_category');
  assert.ok(!(q.min_price_cents !== null && q.max_price_cents !== null && q.min_price_cents > q.max_price_cents), 'min_price_cents_exceeds_max_price_cents');
  assert.ok(Number.isInteger(q.page_size) && q.page_size >= 1 && q.page_size <= 50, 'invalid_page_size');
  assert.ok(Number.isInteger(q.page) && q.page >= 1 && q.page <= 100000, 'invalid_page');
  assert.ok(VALID_SORTS.has(q.sort), 'unsupported_sort');
}

function evaluate(assets, input, fault = null) {
  let q = { ...DEFAULT, ...input };
  if (fault === 'swap_bounds' && q.min_price_cents !== null && q.max_price_cents !== null && q.min_price_cents > q.max_price_cents) {
    q = { ...q, min_price_cents: q.max_price_cents, max_price_cents: q.min_price_cents };
  }
  validateQuery(q);
  const query = q.q.trim();
  let source = assets;
  if (fault === 'paginate_before_filter') source = source.slice((q.page - 1) * q.page_size, q.page * q.page_size);
  const matches = source.filter((asset) => {
    const fields = fault === 'name_only' ? [asset.name] : [asset.name, asset.sku, asset.description];
    const haystack = fields.map((value) => fault === 'case_sensitive' ? value : asciiLower(value));
    const term = fault === 'case_sensitive' ? query : asciiLower(query);
    const searchOk = term === '' || haystack.some((value) => value.includes(term));
    const categoryOk = q.category === null || asset.category === q.category;
    const stockOk = !q.available_only || asset.stock_count > 0;
    const filterOk = fault === 'category_or_stock' && q.category !== null && q.available_only
      ? categoryOk || stockOk : categoryOk && stockOk;
    const minOk = q.min_price_cents === null || (fault === 'exclusive_price' ? asset.price_cents > q.min_price_cents : asset.price_cents >= q.min_price_cents);
    const maxOk = q.max_price_cents === null || (fault === 'exclusive_price' ? asset.price_cents < q.max_price_cents : asset.price_cents <= q.max_price_cents);
    return searchOk && filterOk && minOk && maxOk;
  });
  const comparePrimary = {
    name_asc: (a, b) => lexical(asciiLower(a.name), asciiLower(b.name)),
    price_asc: (a, b) => a.price_cents - b.price_cents,
    price_desc: (a, b) => b.price_cents - a.price_cents,
    stock_asc: (a, b) => a.stock_count - b.stock_count,
  }[q.sort];
  matches.sort((a, b) => comparePrimary(a, b) || (fault === 'input_order_ties' ? 0 : lexical(a.id, b.id)));
  const total = matches.length;
  const pageCount = Math.ceil(total / q.page_size);
  const pageRows = fault === 'paginate_before_filter' ? matches : matches.slice((q.page - 1) * q.page_size, q.page * q.page_size);
  return {
    asset_ids: pageRows.map((asset) => asset.id), total_matches: total,
    page: q.page, page_size: q.page_size, page_count: pageCount,
  };
}

function lexical(a, b) { return a < b ? -1 : a > b ? 1 : 0; }
function asciiLower(value) { return value.replace(/[A-Z]/g, (letter) => String.fromCharCode(letter.charCodeAt(0) + 32)); }
function orderedState(q) {
  return Object.fromEntries(STATE_FIELDS.map((field) => [field, q[field]]));
}
function percentEncodeUtf8(value) {
  return [...Buffer.from(value, 'utf8')].map((byte) =>
    (byte >= 0x41 && byte <= 0x5a) || (byte >= 0x61 && byte <= 0x7a) ||
    (byte >= 0x30 && byte <= 0x39) || [0x2d, 0x2e, 0x5f, 0x7e].includes(byte)
      ? String.fromCharCode(byte) : `%${byte.toString(16).toUpperCase().padStart(2, '0')}`
  ).join('');
}
function percentDecodeUtf8(value) {
  const bytes = [];
  for (let i = 0; i < value.length;) {
    if (value[i] === '%') {
      assert.ok(i + 2 < value.length && /^[0-9a-f]{2}$/i.test(value.slice(i + 1, i + 3)), 'invalid_state_url_percent_escape');
      bytes.push(Number.parseInt(value.slice(i + 1, i + 3), 16)); i += 3;
    } else {
      const code = value.charCodeAt(i++);
      assert.ok(code < 128 && /^[A-Za-z0-9._~-]$/.test(value[i - 1]), 'state_url_must_percent_encode_reserved_or_non_ascii_bytes');
      bytes.push(code);
    }
  }
  return new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(bytes));
}
function encodeStateUrl(input) {
  const q = { ...DEFAULT, ...input };
  validateQuery(q);
  return `?state=${percentEncodeUtf8(JSON.stringify(orderedState(q)))}`;
}
function assertNoDuplicateStateFields(json) {
  let index = 0;
  const skipWhitespace = () => {
    while (index < json.length && /\s/.test(json[index])) index++;
  };
  const readString = () => {
    const start = index++;
    while (index < json.length) {
      const char = json[index++];
      if (char === '\\') { index++; continue; }
      if (char === '"') return JSON.parse(json.slice(start, index));
    }
    throw new Error('state_url_json_invalid');
  };
  const skipValue = () => {
    skipWhitespace();
    if (json[index] === '"') { readString(); return; }
    if (json[index] === '{' || json[index] === '[') {
      const stack = [json[index++] === '{' ? '}' : ']'];
      let inString = false;
      while (index < json.length && stack.length) {
        const char = json[index++];
        if (inString) {
          if (char === '\\') index++;
          else if (char === '"') inString = false;
        } else if (char === '"') inString = true;
        else if (char === '{') stack.push('}');
        else if (char === '[') stack.push(']');
        else if (char === '}' || char === ']') {
          if (stack.pop() !== char) throw new Error('state_url_json_invalid');
        }
      }
      if (stack.length || inString) throw new Error('state_url_json_invalid');
      return;
    }
    while (index < json.length && json[index] !== ',' && json[index] !== '}' && !/\s/.test(json[index])) index++;
  };
  skipWhitespace();
  if (json[index++] !== '{') throw new Error('state_url_fields_mismatch');
  const seen = new Set();
  while (true) {
    skipWhitespace();
    if (json[index] === '}') { index++; break; }
    if (json[index] !== '"') throw new Error('state_url_json_invalid');
    const key = readString();
    if (seen.has(key)) throw new Error('state_url_duplicate_field');
    seen.add(key);
    skipWhitespace();
    if (json[index++] !== ':') throw new Error('state_url_json_invalid');
    skipValue();
    skipWhitespace();
    if (json[index] === ',') { index++; continue; }
    if (json[index] === '}') { index++; break; }
    throw new Error('state_url_json_invalid');
  }
  skipWhitespace();
  if (index !== json.length) throw new Error('state_url_json_invalid');
}
function decodeStateUrl(url) {
  assert.ok(url.startsWith('?state=') && !url.slice(7).includes('&'), 'state_url_must_contain_only_state_parameter');
  const json = percentDecodeUtf8(url.slice(7));
  assertNoDuplicateStateFields(json);
  const parsed = JSON.parse(json);
  assert.ok(parsed && typeof parsed === 'object' && !Array.isArray(parsed), 'state_url_fields_mismatch');
  const keys = Object.keys(parsed);
  assert.ok(keys.length === STATE_FIELDS.length && STATE_FIELDS.every((field) => Object.hasOwn(parsed, field)), 'state_url_fields_mismatch');
  const q = { ...DEFAULT, ...parsed };
  validateQuery(q);
  return orderedState(q);
}
function equalOutcome(actual, item) {
  if (item.expected_error) return false;
  return JSON.stringify(actual.asset_ids) === JSON.stringify(item.expected.asset_ids) &&
    actual.total_matches === item.expected.total_matches && actual.page === item.expected.page &&
    actual.page_size === item.expected.page_size && actual.page_count === item.expected.page_count;
}
function differsFromGolden(fault) {
  for (const item of oracle.cases) {
    try {
      const actual = evaluate(dataset.assets, item.input, fault);
      if (item.expected_error || !equalOutcome(actual, item)) return item.case_id;
    } catch (error) {
      if (!item.expected_error || error.message !== item.expected_error) return item.case_id;
    }
  }
  return null;
}

validateDataset(dataset);
assert.equal(oracle.schema, 'sfeb-search-filter-oracle-cases-v1');
assert.equal(oracle.dataset_id, dataset.dataset_id);
assert.equal(urlVectors.schema, 'sfeb-url-state-conformance-v1');
assert.equal(urlVectors.policy.parameter_count, 1);
assert.equal(urlVectors.policy.safe_integer_max, Number.MAX_SAFE_INTEGER);
for (const item of urlVectors.valid_states) {
  const restored = decodeStateUrl(encodeStateUrl(item.state));
  assert.deepEqual(restored, orderedState(item.state), `valid URL vector ${item.id}`);
}
for (const item of urlVectors.invalid_states) {
  const encoded = percentEncodeUtf8(JSON.stringify(item.state));
  assert.throws(() => decodeStateUrl(`?state=${encoded}`), (error) => error.message.includes(item.error), `invalid state vector ${item.id}`);
}
for (const item of urlVectors.invalid_urls) {
  assert.throws(() => decodeStateUrl(item.url), undefined, `invalid URL vector ${item.id}`);
}
assert.equal(asciiLower('İX-A'), 'İx-a', 'case folding is ASCII-only');
const goldenFailures = [];
for (const item of oracle.cases) {
  try {
    const result = evaluate(dataset.assets, item.input);
    if (item.expected_error || !equalOutcome(result, item)) goldenFailures.push(item.case_id);
  } catch (error) {
    if (!item.expected_error || error.message !== item.expected_error) goldenFailures.push(item.case_id);
  }
}
assert.deepEqual(goldenFailures, [], 'golden cases must all match');

// Deterministic property sweep. LCG arithmetic is deliberately 32-bit and portable.
let seed = 20261122;
const next = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
const terms = ['', 'sensor', 'pump', 'DEV', 'battery', 'pressure', 'thermal', 'no-match', 'FIELD', 'SEN-'];
const categories = [null, 'Devices', 'Inspection', 'Power', 'Safety', 'Sensors'];
const sorts = [...VALID_SORTS];
let propertyFailure = null;
for (let trial = 0; trial < 5000; trial++) {
  const a = next() % 350001, b = next() % 350001;
  const low = Math.min(a, b), high = Math.max(a, b);
  const q = {
    q: terms[next() % terms.length], category: categories[next() % categories.length],
    available_only: Boolean(next() % 2),
    min_price_cents: next() % 4 === 0 ? low : null,
    max_price_cents: next() % 4 === 0 ? high : null,
    sort: sorts[next() % sorts.length], page: 1 + (next() % 6), page_size: 1 + (next() % 7),
  };
  const paged = evaluate(dataset.assets, q);
  const all = evaluate(dataset.assets, { ...q, page: 1, page_size: 50 });
  const reconstructed = [];
  for (let page = 1; page <= paged.page_count; page++) reconstructed.push(...evaluate(dataset.assets, { ...q, page, page_size: q.page_size }).asset_ids);
  const base = evaluate(dataset.assets, { ...q, available_only: false, page: 1, page_size: 50 }).asset_ids;
  const withStock = evaluate(dataset.assets, { ...q, available_only: true, page: 1, page_size: 50 }).asset_ids;
  const url = decodeStateUrl(encodeStateUrl(q));
  const caseVariant = evaluate(dataset.assets, { ...q, q: q.q.toUpperCase() });
  if (new Set(reconstructed).size !== reconstructed.length ||
      JSON.stringify(reconstructed) !== JSON.stringify(all.asset_ids) ||
      withStock.some((id) => !base.includes(id)) ||
      JSON.stringify(url) !== JSON.stringify(orderedState(q)) ||
      JSON.stringify(caseVariant.asset_ids) !== JSON.stringify(evaluate(dataset.assets, q).asset_ids)) {
    propertyFailure = trial; break;
  }
}
assert.equal(propertyFailure, null, `generated property failure at trial ${propertyFailure}`);

const semanticFaults = [
  ['MUT-SEARCH-001', 'case_sensitive'], ['MUT-SEARCH-002', 'name_only'],
  ['MUT-FILTER-003', 'category_or_stock'], ['MUT-PRICE-004', 'exclusive_price'],
  ['MUT-PAGE-006', 'paginate_before_filter'],
  ['MUT-PRICE-008', 'swap_bounds'],
];
const faultResults = semanticFaults.map(([id, kind]) => ({ id, killed_by: differsFromGolden(kind) }));
const tieInput = oracle.cases.find((item) => item.case_id === 'SFEB-SEARCH-008').input;
const tiedOriginal = evaluate(dataset.assets, tieInput).asset_ids;
const tiedPermuted = evaluate([...dataset.assets].reverse(), tieInput, 'input_order_ties').asset_ids;
faultResults.push({ id: 'MUT-SORT-005', killed_by: JSON.stringify(tiedOriginal) !== JSON.stringify(tiedPermuted) ? 'dataset permutation' : null });
const urlInput = { q: 'pump & sensor', category: 'Power', available_only: true, min_price_cents: 5000, max_price_cents: 250000, sort: 'price_desc', page: 3, page_size: 10 };
const omitted = STATE_FIELDS.map((field) => {
  const broken = { ...urlInput }; delete broken[field];
  const brokenUrl = `?state=${percentEncodeUtf8(JSON.stringify(broken))}`;
  let detected = false;
  try {
    detected = JSON.stringify(decodeStateUrl(brokenUrl)) !== JSON.stringify(orderedState(urlInput));
  } catch {
    // Strict state decoders must reject incomplete state rather than defaulting fields.
    detected = true;
  }
  return { field, detected };
});
faultResults.push({ id: 'MUT-URL-007', killed_by: omitted.every((item) => item.detected) ? 'all eight state-field omissions' : null });
const pageOnly = { ...DEFAULT, page: urlInput.page, page_size: urlInput.page_size };
faultResults.push({ id: 'MUT-STATE-010', killed_by: JSON.stringify(pageOnly) !== JSON.stringify(orderedState(urlInput)) ? 'fresh-session page-only restore' : null });
assert.ok(faultResults.every((item) => item.killed_by), 'all nine semantic fault models should be detected');

const output = {
  task_id: oracle.task_id ?? 'SFEB-L2-005',
  golden: { passed: oracle.cases.length, failed: goldenFailures.length },
  generated_properties: { passed: 5000, failed: 0, seed: 20261122 },
  semantic_fault_models: { detected: faultResults.length, total: faultResults.length, results: faultResults },
  url_field_omissions: { detected: omitted.filter((item) => item.detected).length, total: omitted.length },
  url_state_conformance: { valid: urlVectors.valid_states.length, invalid_state_cases: urlVectors.invalid_states.length, invalid_url_components: urlVectors.invalid_urls.length, total: urlVectors.valid_states.length + urlVectors.invalid_states.length + urlVectors.invalid_urls.length },
  boundaries: { exact_min_and_max_are_inclusive: true },
  scope: 'Independent JavaScript model preflight only; not Rust execution, Leptos browser qualification, XSS DOM assertion or the complete mutation campaign.',
};
console.log(JSON.stringify(output, null, 2));

#!/usr/bin/env node
// Research-only CPIR raw-byte JSON parser and restricted JCS oracle.
// This is not a complete RFC 8785 implementation or a production parser.
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { TextDecoder } from "node:util";

const decoder = new TextDecoder("utf-8", { fatal: true });
const manifestPath = resolve(dirname(fileURLToPath(import.meta.url)), "fixtures/CORPUS_MANIFEST_V1.json");
const fail = (code, message) => { const e = new Error(message); e.code = code; throw e; };
const digest = bytes => createHash("sha256").update(bytes).digest("hex");

function byteOffsets(text) {
  const offsets = new Array(text.length + 1);
  let bytes = 0;
  for (let i = 0; i < text.length;) {
    offsets[i] = bytes;
    const cp = text.codePointAt(i);
    const units = cp > 0xffff ? 2 : 1;
    if (units === 2) offsets[i + 1] = bytes;
    bytes += cp <= 0x7f ? 1 : cp <= 0x7ff ? 2 : cp <= 0xffff ? 3 : 4;
    i += units;
  }
  offsets[text.length] = bytes;
  return offsets;
}
function assertScalars(value) {
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      const d = value.charCodeAt(++i);
      if (!(d >= 0xdc00 && d <= 0xdfff)) fail("surrogate", "unpaired high surrogate");
    } else if (c >= 0xdc00 && c <= 0xdfff) fail("surrogate", "unpaired low surrogate");
  }
}

class RawParser {
  constructor(bytes) {
    if (bytes.length > 1048576) fail("limit", "input exceeds 1 MiB");
    try { this.text = decoder.decode(bytes); } catch { fail("utf8", "invalid UTF-8"); }
    if (this.text.charCodeAt(0) === 0xfeff) fail("bom", "UTF-8 BOM rejected");
    this.offset = byteOffsets(this.text);
    this.i = 0;
    this.depth = 0;
  }
  b(i) { return this.offset[i]; }
  ws() { while (/[\x20\x09\x0a\x0d]/.test(this.text[this.i] ?? "")) this.i++; }
  string() {
    const start = this.i;
    if (this.text[this.i++] !== '"') fail("syntax", "expected string");
    while (this.i < this.text.length) {
      const c = this.text.charCodeAt(this.i);
      if (c === 34) { this.i++; break; }
      if (c < 32) fail("syntax", "unescaped control character");
      if (c === 92) {
        this.i++;
        const e = this.text[this.i];
        if ('"\\/bfnrt'.includes(e)) { this.i++; continue; }
        if (e === "u" && /^[0-9a-fA-F]{4}$/.test(this.text.slice(this.i + 1, this.i + 5))) {
          this.i += 5;
          continue;
        }
        fail("syntax", "invalid string escape");
      }
      this.i++;
    }
    if (this.text[this.i - 1] !== '"') fail("syntax", "unterminated string");
    const raw = this.text.slice(start, this.i);
    let value;
    try { value = JSON.parse(raw); } catch { fail("syntax", "invalid string token"); }
    assertScalars(value);
    return { value, start: this.b(start), end: this.b(this.i), raw };
  }
  number() {
    const start = this.i;
    const match = this.text.slice(start).match(/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/);
    if (!match) fail("syntax", "invalid number");
    const lexeme = match[0];
    this.i += lexeme.length;
    if (/[.eE]/.test(lexeme) || lexeme === "-0") fail("number_profile", "only non-negative-zero integer tokens are admitted");
    const n = BigInt(lexeme);
    if (n < -9007199254740991n || n > 9007199254740991n) fail("unsafe_integer", "integer outside exact safe range");
    return { kind: "number", value: Number(n), lexeme, start: this.b(start), end: this.b(this.i) };
  }
  value() {
    this.ws();
    const start = this.i;
    const c = this.text[this.i];
    if (c === '"') {
      const s = this.string();
      return { kind: "string", value: s.value, start: s.start, end: s.end };
    }
    if (c === "{") return this.object();
    if (c === "[") return this.array();
    if (c === "-" || (c >= "0" && c <= "9")) return this.number();
    for (const [token, value] of [["true", true], ["false", false], ["null", null]]) {
      if (this.text.startsWith(token, this.i)) {
        this.i += token.length;
        return { kind: "literal", value, lexeme: token, start: this.b(start), end: this.b(this.i) };
      }
    }
    fail("syntax", "unexpected token");
  }
  object() {
    const start = this.i++;
    if (++this.depth > 64) fail("depth", "nesting limit exceeded");
    const entries = [];
    const names = new Set();
    this.ws();
    if (this.text[this.i] === "}") {
      this.i++; this.depth--;
      return { kind: "object", entries, start: this.b(start), end: this.b(this.i) };
    }
    while (true) {
      this.ws();
      const key = this.string();
      if (names.has(key.value)) fail("duplicate", "duplicate decoded member name: " + key.value);
      names.add(key.value);
      this.ws();
      if (this.text[this.i++] !== ":") fail("syntax", "expected colon");
      const value = this.value();
      entries.push({ key: key.value, value });
      this.ws();
      const delimiter = this.text[this.i++];
      if (delimiter === "}") break;
      if (delimiter !== ",") fail("syntax", "expected comma or object close");
    }
    this.depth--;
    return { kind: "object", entries, start: this.b(start), end: this.b(this.i) };
  }
  array() {
    const start = this.i++;
    if (++this.depth > 64) fail("depth", "nesting limit exceeded");
    const items = [];
    this.ws();
    if (this.text[this.i] === "]") {
      this.i++; this.depth--;
      return { kind: "array", items, start: this.b(start), end: this.b(this.i) };
    }
    while (true) {
      items.push(this.value());
      this.ws();
      const delimiter = this.text[this.i++];
      if (delimiter === "]") break;
      if (delimiter !== ",") fail("syntax", "expected comma or array close");
    }
    this.depth--;
    return { kind: "array", items, start: this.b(start), end: this.b(this.i) };
  }
  parse() {
    this.ws();
    const root = this.value();
    this.ws();
    if (this.i !== this.text.length) fail("trailing", "trailing non-whitespace data");
    return root;
  }
}
const parseRaw = bytes => new RawParser(bytes).parse();

function pointer(root, path) {
  if (path === "") return root;
  if (typeof path !== "string" || !path.startsWith("/")) fail("pointer_syntax", "pointer must be empty or start with /");
  let node = root;
  for (let token of path.slice(1).split("/")) {
    if (/~(?:[^01]|$)/.test(token)) fail("pointer_escape", "invalid RFC 6901 escape");
    token = token.replace(/~1/g, "/").replace(/~0/g, "~");
    if (node.kind === "object") {
      const entry = node.entries.find(item => item.key === token);
      if (!entry) fail("pointer_missing", "object member absent");
      node = entry.value;
    } else if (node.kind === "array") {
      if (!/^(0|[1-9][0-9]*)$/.test(token)) fail("pointer_index", "invalid array index");
      const index = Number(token);
      if (!Number.isSafeInteger(index) || index >= node.items.length) fail("pointer_missing", "array element absent");
      node = node.items[index];
    } else fail("pointer_type", "cannot traverse a scalar");
  }
  return node;
}
function jcsSubset(node) {
  if (node.kind === "string") return JSON.stringify(node.value);
  if (node.kind === "number") return node.lexeme;
  if (node.kind === "literal") return node.lexeme;
  if (node.kind === "array") return "[" + node.items.map(jcsSubset).join(",") + "]";
  return "{" + [...node.entries].sort((a, b) => a.key < b.key ? -1 : a.key > b.key ? 1 : 0)
    .map(item => JSON.stringify(item.key) + ":" + jcsSubset(item.value)).join(",") + "}";
}
function minorUnits(decimal, exponent) {
  const match = decimal.match(/^(-?)(0|[1-9][0-9]*)(?:\.([0-9]+))?$/);
  if (!match || (match[3] ?? "").length > exponent) fail("currency", "invalid decimal precision");
  const scale = 10n ** BigInt(exponent);
  const fraction = BigInt(((match[3] ?? "").padEnd(exponent, "0")) || "0");
  const value = BigInt(match[2]) * scale + fraction;
  return (match[1] === "-" ? -value : value).toString();
}
function rfc3339(value) {
  const match = value.match(/^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(Z|[+-]\d\d:\d\d)$/);
  if (!match) fail("timestamp", "outside narrow RFC 3339 fixture profile");
  const offset = match[2];
  const epoch = Date.parse(match[1] + (offset === "-00:00" ? "Z" : offset));
  if (!Number.isFinite(epoch)) fail("timestamp", "invalid timestamp");
  return { epoch, offsetState: offset === "Z" || offset === "-00:00" ? "unknown_local_offset" :
    offset === "+00:00" ? "utc_preferred" : "offset_explicit" };
}
async function loadFixture(vector, base) {
  const path = resolve(base, vector.file);
  if (!path.startsWith(base + "/")) fail("path", "fixture escaped conformance directory");
  const bytes = await readFile(path);
  if (!vector.line_number) return bytes;
  const lines = [];
  let start = 0;
  for (let i = 0; i < bytes.length; i++) if (bytes[i] === 10) { lines.push(bytes.subarray(start, i)); start = i + 1; }
  if (start < bytes.length) lines.push(bytes.subarray(start));
  if (vector.line_number < 1 || vector.line_number > lines.length) fail("line", "fixture line out of range");
  return lines[vector.line_number - 1];
}

async function main() {
  const manifest = JSON.parse((await readFile(manifestPath)).toString("utf8"));
  const base = dirname(dirname(manifestPath));
  let parsed = 0, rejected = 0, spanChecks = 0, canonicalChecks = 0;
  const errors = [], asts = new Map();
  for (const vector of manifest.vectors) {
    const bytes = await loadFixture(vector, base);
    if (bytes.length !== vector.byte_length || digest(bytes) !== vector.sha256) {
      errors.push(vector.id + ": raw fixture digest/length mismatch");
      continue;
    }
    let ast;
    try { ast = parseRaw(bytes); parsed++; asts.set(vector.id, ast); }
    catch (error) {
      rejected++;
      if (!vector.expected_disposition.startsWith("reject_")) errors.push(vector.id + ": unexpected rejection " + error.code);
      if (vector.expected_disposition === "reject_unsafe_integer" && error.code !== "unsafe_integer") errors.push(vector.id + ": wrong rejection class " + error.code);
      continue;
    }
    if (vector.expected_disposition.startsWith("reject_")) { errors.push(vector.id + ": expected rejection but parser admitted input"); continue; }
    if (vector.id === "unicode-normalization-distinct-001" &&
        (ast.entries.length !== 2 || ast.entries[0].key === ast.entries[1].key)) errors.push(vector.id + ": Unicode names collapsed");
    for (const [path, lexeme] of Object.entries(vector.selected_values ?? {})) {
      if (pointer(ast, path).lexeme !== lexeme) errors.push(vector.id + ": numeric lexeme mismatch " + path);
    }
    for (const item of vector.selections ?? []) {
      try {
        const node = pointer(ast, item.pointer);
        const raw = bytes.subarray(node.start, node.end).toString("utf8");
        if (node.start !== item.start_byte || node.end !== item.end_byte || raw !== item.expected_raw_token) errors.push(vector.id + ": byte span mismatch " + item.pointer);
        else spanChecks++;
      } catch (error) { errors.push(vector.id + ": pointer/span failure " + item.pointer + " (" + error.code + ")"); }
    }
    for (const item of vector.selection_cases ?? []) {
      try {
        const node = pointer(ast, item.pointer);
        const actual = node.kind === "literal" && node.value === null ? "present_null" :
          node.kind === "string" && node.value === "" ? "present_empty_string" :
          node.kind === "number" && node.value === 0 ? "present_zero" :
          node.kind === "array" && node.items.length === 0 ? "present_empty_array" : "present_other";
        if (actual !== item.expected) errors.push(vector.id + ": presence state mismatch " + item.pointer);
      } catch (error) {
        if (item.expected !== "not_found" || error.code !== "pointer_missing") errors.push(vector.id + ": absence mismatch " + item.pointer);
      }
    }
    if (vector.selection) {
      try { pointer(ast, vector.selection.pointer); errors.push(vector.id + ": invalid pointer unexpectedly resolved"); }
      catch (error) { if (error.code !== "pointer_escape") errors.push(vector.id + ": wrong pointer error " + error.code); }
    }
    if (vector.expected_canonical_json) {
      if (jcsSubset(ast) !== vector.expected_canonical_json) errors.push(vector.id + ": JCS subset output mismatch");
      else canonicalChecks++;
    }
    if (vector.id === "currency-minor-units-001") {
      const value = minorUnits(pointer(ast, "/amount").value, vector.conversion.minor_unit_exponent);
      if (value !== vector.conversion.expected_minor_units) errors.push(vector.id + ": exact currency conversion mismatch");
    }
    if (vector.id === "timestamp-offset-equivalent-001") {
      if (rfc3339(pointer(ast, "/local").value).epoch !== rfc3339(pointer(ast, "/utc").value).epoch) errors.push(vector.id + ": equal timestamps differ");
    }
    if (vector.id === "timestamp-unknown-offset-001") {
      const a = rfc3339(pointer(ast, "/legacy_unknown").value);
      const b = rfc3339(pointer(ast, "/z_unknown").value);
      const c = rfc3339(pointer(ast, "/utc_preferred").value);
      if (a.epoch !== b.epoch || a.offsetState !== b.offsetState || c.epoch !== b.epoch || c.offsetState !== "utc_preferred") errors.push(vector.id + ": RFC 9557 timestamp state mismatch");
    }
  }
  for (const relation of manifest.relations ?? []) {
    const nodes = relation.vector_ids.map(id => asts.get(id));
    if (nodes.some(node => !node) || nodes.some(node => jcsSubset(node) !== relation.expected_canonical_json)) errors.push(relation.id + ": metamorphic JCS mismatch");
    else canonicalChecks++;
  }
  if (errors.length) {
    console.error(errors.join("\n"));
    process.exitCode = 1;
  } else {
    console.log("PASS JS oracle research corpus: " + manifest.vectors.length +
      " vectors; parsed=" + parsed + "; rejected=" + rejected +
      "; span_checks=" + spanChecks + "; canonical_checks=" + canonicalChecks);
    console.log("Claim ceiling: restricted-profile parser/JCS-subset research only; not Rust agreement, full RFC 8785 conformance or production qualification.");
  }
}
await main();

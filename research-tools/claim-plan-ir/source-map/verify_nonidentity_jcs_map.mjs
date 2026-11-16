#!/usr/bin/env node
// Non-identity JCS semantic node-origin map test. Restricted to an ASCII,
// integer/string fixture; this is not a full RFC 8785 implementation.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const fail = (message) => { throw new Error(message); };
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const manifest = JSON.parse((await readFile(resolve(here, "JCS_NONIDENTITY_V1.json"))).toString("utf8"));
const source = await readFile(resolve(here, manifest.source_file));
const canonical = await readFile(resolve(here, manifest.canonical_file));
const mapBytes = await readFile(resolve(here, manifest.node_map_file));
const map = JSON.parse(mapBytes.toString("utf8"));

function jcsSubset(value) {
  if (value === null || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || Object.is(value, -0)) fail("outside integer-only test subset");
    return String(value);
  }
  if (Array.isArray(value)) return "[" + value.map(jcsSubset).join(",") + "]";
  if (typeof value === "object") {
    const keys = Object.keys(value).sort(); // ECMAScript UTF-16 code-unit order
    return "{" + keys.map(k => JSON.stringify(k) + ":" + jcsSubset(value[k])).join(",") + "}";
  }
  fail("unsupported value in test subset");
}
function validate(v, m, src=source, can=canonical) {
  if (src.length !== v.source_byte_length || sha(src) !== v.source_sha256) fail("source bytes mismatch");
  if (can.length !== v.canonical_byte_length || sha(can) !== v.canonical_sha256) fail("canonical bytes mismatch");
  if (sha(mapBytes) !== v.node_map_sha256) fail("node-map artifact digest mismatch");
  if (m.input_representation_id !== "jcs-source" || m.output_representation_id !== "jcs-canonical" ||
      m.input_sha256 !== v.source_sha256 || m.output_sha256 !== v.canonical_sha256) fail("node-map endpoints/digests mismatch");
  if (src.equals(can)) fail("fixture must exercise non-identity canonicalization");
  const parsed = JSON.parse(src.toString("utf8"));
  const expected = Buffer.from(jcsSubset(parsed), "utf8");
  if (!expected.equals(can) || expected.toString("utf8") !== v.expected_canonical_json) fail("JCS subset known-answer mismatch");
  if (v.json_pointer !== "/name" || v.source_coordinate_space !== "json_source_utf8_bytes" ||
      v.canonical_coordinate_space !== "jcs_canonical_json_bytes") fail("pointer or coordinate-space substitution");
  const candidates = m.nodes.filter(n => n.json_pointer === v.json_pointer);
  if (candidates.length !== 1) fail("node-origin mapping is missing or ambiguous");
  const n = candidates[0];
  if (n.source_start !== v.source_start || n.source_end !== v.source_end ||
      n.canonical_start !== v.canonical_start || n.canonical_end !== v.canonical_end) fail("source/canonical node spans mismatch");
  if (n.source_node_id !== "jcs-source:/name" || n.canonical_node_id !== "jcs-canonical:/name") fail("node identity mismatch");
  const rawToken = src.subarray(v.source_start, v.source_end);
  const canonicalToken = can.subarray(v.canonical_start, v.canonical_end);
  if (rawToken.toString("utf8") !== v.expected_raw_token ||
      canonicalToken.toString("utf8") !== v.expected_raw_token) fail("source/canonical selected token differs");
  if (sha(rawToken) !== v.raw_token_sha256 || n.raw_token_sha256 !== v.raw_token_sha256 ||
      n.canonical_token_sha256 !== v.raw_token_sha256) fail("node-origin token digest mismatch");
  if (v.source_start === v.canonical_start || v.source_end === v.canonical_end) fail("the fixture did not expose coordinate shift");
  return true;
}
validate(manifest, map);
const attacks = [
  ["wrong_source_span", v => { v.source_start += 1; }],
  ["wrong_canonical_span", v => { v.canonical_start = v.source_start; v.canonical_end = v.source_end; }],
  ["wrong_pointer", v => { v.json_pointer = "/age"; }],
  ["wrong_source_digest", v => { v.source_sha256 = "0".repeat(64); }],
  ["wrong_canonical_digest", v => { v.canonical_sha256 = "0".repeat(64); }],
  ["wrong_raw_token_digest", v => { v.raw_token_sha256 = "0".repeat(64); }],
  ["wrong_node_identity", (_v, m) => { m.nodes[0].source_node_id = "jcs-source:/age"; }],
  ["ambiguous_node_origin", (_v, m) => { m.nodes.push(structuredClone(m.nodes[0])); }]
];
for (const [name, mutate] of attacks) {
  const v = structuredClone(manifest), m = structuredClone(map);
  mutate(v,m);
  let rejected = false;
  try { validate(v,m); } catch { rejected = true; }
  if (!rejected) fail("negative control accepted: " + name);
}
console.log("PASS non-identity JCS node-origin map: source=22B, canonical=21B, /name source=[8,11), canonical=[17,20); " + attacks.length + " mutation controls rejected.");
console.log("Claim ceiling: restricted ASCII/integer JCS subset fixture only; not full RFC 8785 or TLS source authentication.");

#!/usr/bin/env node
// Research-only corpus integrity checker. It does not parse JSON semantics,
// implement JCS, compare parsers, or qualify any CPIR production dependency.
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const manifestPath = resolve(here, "fixtures", "CORPUS_MANIFEST_V1.json");
const manifestDir = dirname(manifestPath);
const conformanceDir = dirname(manifestDir);

function fail(message) {
  throw new Error(message);
}

const manifest = JSON.parse((await readFile(manifestPath)).toString("utf8"));
if (manifest.schema_version !== 1) fail("unsupported manifest schema_version");
if (manifest.profile_id !== "luminous.cpir.parser-oracle.v1") fail("unexpected profile_id");
if (manifest.fixture_path_base !== "conformance_directory") fail("unexpected fixture_path_base");
if (!Array.isArray(manifest.vectors) || manifest.vectors.length === 0) fail("empty vectors");

const seenIds = new Set();
const seenFixtures = new Set();
const digestById = new Map();
const lineCache = new Map();
const utf8 = new TextDecoder("utf-8", { fatal: true });
let totalBytes = 0;

for (const vector of manifest.vectors) {
  if (typeof vector.id !== "string" || vector.id.length === 0) fail("vector without id");
  if (seenIds.has(vector.id)) fail("duplicate vector id: " + vector.id);
  seenIds.add(vector.id);
  if (typeof vector.file !== "string" || vector.file.startsWith("/") ||
      vector.file.includes("\\") || vector.file.split("/").includes("..")) {
    fail("unsafe fixture path: " + vector.id);
  }
  if (!Number.isSafeInteger(vector.byte_length) || vector.byte_length < 0) fail("bad byte_length: " + vector.id);
  if (!/^[0-9a-f]{64}$/.test(vector.sha256)) fail("bad SHA-256: " + vector.id);
  if (typeof vector.expected_disposition !== "string") fail("missing expected_disposition: " + vector.id);

  const fixturePath = resolve(conformanceDir, vector.file);
  if (!fixturePath.startsWith(conformanceDir + "/")) fail("fixture escaped conformance directory");
  let bytes;
  if (Number.isSafeInteger(vector.line_number)) {
    const key = vector.file;
    let lines = lineCache.get(key);
    if (!lines) {
      const corpus = await readFile(fixturePath);
      lines = [];
      let start = 0;
      for (let i = 0; i < corpus.length; i++) {
        if (corpus[i] === 0x0a) {
          lines.push(corpus.subarray(start, i));
          start = i + 1;
        }
      }
      if (start < corpus.length) lines.push(corpus.subarray(start));
      lineCache.set(key, lines);
    }
    if (vector.line_number < 1 || vector.line_number > lines.length) fail("line out of range: " + vector.id);
    bytes = lines[vector.line_number - 1];
    const fixtureKey = vector.file + "#" + vector.line_number;
    if (seenFixtures.has(fixtureKey)) fail("duplicate raw-line fixture reference: " + fixtureKey);
    seenFixtures.add(fixtureKey);
  } else {
    if (seenFixtures.has(vector.file)) fail("duplicate fixture reference: " + vector.file);
    seenFixtures.add(vector.file);
    bytes = await readFile(fixturePath);
  }

  try { utf8.decode(bytes); } catch { fail("fixture is not well-formed UTF-8: " + vector.id); }
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (bytes.length !== vector.byte_length) fail(vector.id + ": byte length mismatch");
  if (digest !== vector.sha256) fail(vector.id + ": SHA-256 mismatch");
  digestById.set(vector.id, digest);
  totalBytes += bytes.length;

  for (const selection of vector.selections ?? []) {
    const start = selection.start_byte;
    const end = selection.end_byte;
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
        start < 0 || end < start || end > bytes.length) fail(vector.id + ": invalid span");
    const expected = Buffer.from(selection.expected_raw_token, "utf8");
    if (!bytes.subarray(start, end).equals(expected)) {
      fail(vector.id + ": raw span mismatch for pointer " + selection.pointer);
    }
  }
}

for (const relation of manifest.relations ?? []) {
  if (relation.kind !== "same_expected_jcs_bytes_different_raw_source") fail("unknown relation kind");
  const members = relation.vector_ids.map((id) => manifest.vectors.find((v) => v.id === id));
  if (members.length < 2 || members.some((v) => !v)) fail("bad relation members: " + relation.id);
  if (members.some((v) => v.expected_canonical_json !== relation.expected_canonical_json)) {
    fail("relation canonical output mismatch: " + relation.id);
  }
  if (relation.raw_source_digests_must_differ) {
    const digests = relation.vector_ids.map((id) => digestById.get(id));
    if (new Set(digests).size !== digests.length) fail("raw digests did not differ: " + relation.id);
  }
}

console.log("PASS corpus-integrity: " + manifest.vectors.length +
  " vectors; " + totalBytes + " fixture bytes; " +
  (manifest.relations ?? []).length +
  " metamorphic relation(s). Parser differential status remains not_run.");

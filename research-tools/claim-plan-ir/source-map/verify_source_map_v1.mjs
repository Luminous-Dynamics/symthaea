#!/usr/bin/env node
// Synthetic CPIR source-map relation checker. This is not a TLS verifier,
// generic HTTP parser, JSON Schema implementation, or proof verifier.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const vectorsPath = resolve(here, "SOURCE_MAP_V1_VECTORS.json");
const vectors = JSON.parse((await readFile(vectorsPath)).toString("utf8"));
const bundle = JSON.parse((await readFile(resolve(here, vectors.bundle_file))).toString("utf8"));
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const fail = (message) => { throw new Error(message); };
const rawHttp = Buffer.from((await readFile(resolve(here, vectors.raw_http_fixture))).toString("ascii").trim(), "base64");

async function loadMaps(b) {
  const maps = new Map();
  for (const artifact of b.source_map_artifacts ?? []) {
    const relative = vectors.map_files[artifact.artifact_id];
    if (!relative) fail("map artifact has no pinned fixture path: " + artifact.artifact_id);
    const bytes = await readFile(resolve(here, relative));
    if (sha(bytes) !== artifact.map_sha256) fail("map artifact digest mismatch: " + artifact.artifact_id);
    const value = JSON.parse(bytes.toString("utf8"));
    if (value.input_representation_id !== artifact.input_representation_id ||
        value.output_representation_id !== artifact.output_representation_id ||
        value.input_sha256 !== artifact.input_sha256 || value.output_sha256 !== artifact.output_sha256) {
      fail("map artifact metadata mismatch: " + artifact.artifact_id);
    }
    maps.set(artifact.map_sha256, value);
  }
  return maps;
}

async function validate(b, mutateMaps = null) {
  if (b.profile_id !== "luminous.cpir.source-map.v1") fail("profile mismatch");
  if (b.session_binding.response_ordinal !== vectors.expected_response_ordinal ||
      b.session_binding.direction !== vectors.expected_direction ||
      b.session_binding.commitment_id !== vectors.expected_commitment_id) fail("session/response binding mismatch");
  const reps = new Map();
  for (const r of b.representations) {
    if (reps.has(r.representation_id)) fail("duplicate representation id");
    reps.set(r.representation_id, r);
  }
  for (const r of b.representations) {
    for (const parent of r.parent_representation_ids) if (!reps.has(parent)) fail("unresolved parent " + parent);
  }
  // Detect cycles in the representation DAG.
  const visiting = new Set(), visited = new Set();
  function visit(id) {
    if (visiting.has(id)) fail("representation graph cycle");
    if (visited.has(id)) return;
    visiting.add(id);
    for (const parent of reps.get(id).parent_representation_ids) visit(parent);
    visiting.delete(id); visited.add(id);
  }
  for (const id of reps.keys()) visit(id);

  const maps = await loadMaps(b);
  if (mutateMaps) mutateMaps(maps, b);
  const artifacts = new Map((b.source_map_artifacts ?? []).map(a => [a.map_sha256, a]));
  const stages = [...b.transformations].sort((a, z) => a.ordinal - z.ordinal);
  if (stages.some((s, i) => s.ordinal !== i)) fail("stage ordinals are not contiguous");
  for (const s of stages) {
    const input = reps.get(s.input_representation_id), output = reps.get(s.output_representation_id);
    if (!input || !output) fail("transformation references unknown representation");
    if (s.input_sha256 !== input.sha256 || s.output_sha256 !== output.sha256) fail("transformation digest/representation mismatch");
    if (!output.parent_representation_ids.includes(input.representation_id)) fail("output does not declare input as parent");
    if (s.profile_digest !== output.profile_digest) fail("transformation profile digest differs from output producer profile");
    if (s.mapping_mode === "identity" && (input.sha256 !== output.sha256 || input.byte_length !== output.byte_length)) fail("identity transform changed bytes");
    if (s.mapping_mode === "range_map" || s.mapping_mode === "semantic_node_map") {
      const a = artifacts.get(s.source_map_digest), map = maps.get(s.source_map_digest);
      if (!a || !map || a.mapping_mode !== s.mapping_mode) fail("missing/incorrect required source map");
      if (a.input_representation_id !== input.representation_id || a.output_representation_id !== output.representation_id) fail("source map endpoints mismatch");
    }
    if (s.mapping_mode === "none") fail("unmapped transform cannot qualify source claim");
  }

  const transcript = reps.get("synthetic-transcript");
  const httpMessage = reps.get("http-message");
  const transferBody = reps.get("transfer-body");
  const contentBody = reps.get("content-body");
  const jsonSource = reps.get("json-source");
  const admitted = reps.get("admitted-json");
  const tokenRep = reps.get(b.field_selection.token_representation_id);
  const jcs = reps.get("jcs-output");
  const typed = reps.get("typed-age");
  const predicate = reps.get("predicate-input");
  if (!transcript || !httpMessage || !transferBody || !contentBody || !jsonSource || !admitted || !tokenRep || !jcs || !typed || !predicate) fail("required representation missing");
  if (rawHttp.length !== vectors.raw_http_fixture_byte_length || sha(rawHttp) !== vectors.raw_http_fixture_sha256) fail("synthetic HTTP fixture digest/length mismatch");
  if (sha(rawHttp) !== transcript.sha256 || sha(rawHttp) !== httpMessage.sha256) fail("transcript/HTTP message fixture binding mismatch");

  const bodyArtifact = [...b.source_map_artifacts].find(a => a.artifact_id === "http-body-range-map-001");
  const bodyMap = bodyArtifact && maps.get(bodyArtifact.map_sha256);
  const bodySegment = bodyMap?.segments?.[0];
  if (!bodySegment || bodySegment.input_start < 0 || bodySegment.input_end > rawHttp.length ||
      bodySegment.output_start !== 0 || bodySegment.output_end - bodySegment.output_start !== bodySegment.input_end - bodySegment.input_start) {
    fail("invalid HTTP message-to-body range map");
  }
  const body = rawHttp.subarray(bodySegment.input_start, bodySegment.input_end);
  if (body.toString("utf8") !== vectors.expected_http_body || sha(body) !== transferBody.sha256 ||
      sha(body) !== contentBody.sha256 || sha(body) !== jsonSource.sha256 || sha(body) !== admitted.sha256) {
    fail("HTTP body / decoded JSON representation mismatch");
  }

  const selection = b.field_selection;
  const span = selection.source_span;
  if (selection.json_pointer !== "/age") fail("unexpected verifier pointer");
  if (selection.source_representation_id !== admitted.representation_id ||
      span.representation_id !== admitted.representation_id ||
      span.coordinate_space !== "json_source_utf8_bytes") fail("source span uses wrong representation/coordinate space");
  if (!Number.isSafeInteger(span.start_byte) || !Number.isSafeInteger(span.end_byte) ||
      span.start_byte < 0 || span.end_byte <= span.start_byte || span.end_byte > body.length) fail("invalid source byte span");
  const rawToken = body.subarray(span.start_byte, span.end_byte);
  if (rawToken.toString("utf8") !== vectors.expected_raw_field_token || sha(rawToken) !== selection.raw_token_sha256 ||
      sha(rawToken) !== tokenRep.sha256 || rawToken.length !== tokenRep.byte_length) fail("selected raw token does not match source span");

  const tokenMapArtifact = b.source_map_artifacts.find(a => a.artifact_id === "raw-token-range-map-001");
  const tokenMap = tokenMapArtifact && maps.get(tokenMapArtifact.map_sha256);
  const tokenSegment = tokenMap?.segments?.[0];
  if (!tokenSegment || tokenSegment.input_start !== span.start_byte || tokenSegment.input_end !== span.end_byte ||
      tokenSegment.output_start !== 0 || tokenSegment.output_end !== tokenRep.byte_length) fail("raw token source map disagrees with selected span");

  const nodeArtifact = b.source_map_artifacts.find(a => a.artifact_id === "jcs-node-origin-map-001");
  const nodeMap = nodeArtifact && maps.get(nodeArtifact.map_sha256);
  const nodeMatches = nodeMap?.nodes?.filter(n => n.json_pointer === selection.json_pointer) ?? [];
  if (nodeMatches.length !== 1) fail("JCS node-origin map is missing or ambiguous");
  const node = nodeMatches[0];
  if (!node || node.source_node_id !== selection.source_node_id || node.canonical_node_id !== selection.canonical_node_id ||
      node.source_start !== span.start_byte || node.source_end !== span.end_byte ||
      node.raw_token_sha256 !== selection.raw_token_sha256 ||
      node.canonical_token_sha256 !== selection.raw_token_sha256 ||
      node.canonical_end - node.canonical_start !== tokenRep.byte_length) fail("JCS node-origin map does not bind unique source token");
  if (jcs.sha256 !== admitted.sha256 || jcs.byte_length !== admitted.byte_length) fail("JCS fixture is expected to be identity-canonical for this vector");
  if (b.typed_conversion.input_raw_token_sha256 !== selection.raw_token_sha256 ||
      b.typed_conversion.typed_value_encoding_digest !== typed.sha256 ||
      typed.sha256 !== tokenRep.sha256) fail("typed conversion is not bound to selected raw token");
  const expectedPredicateBytes = Buffer.from(vectors.expected_predicate_input, "utf8");
  if (expectedPredicateBytes.length !== vectors.expected_predicate_input_byte_length ||
      sha(expectedPredicateBytes) !== vectors.expected_predicate_input_sha256 ||
      b.predicate_binding.public_input_digest !== predicate.sha256 ||
      predicate.sha256 !== vectors.expected_predicate_input_sha256 ||
      predicate.byte_length !== vectors.expected_predicate_input_byte_length) {
    fail("predicate public input is not bound to expected predicate bytes");
  }
  return true;
}

await validate(bundle);
const mutations = [
  ["wrong_source_coordinate_space", b => { b.field_selection.source_span.coordinate_space = "jcs_canonical_json_bytes"; }],
  ["wrong_source_byte_span", b => { b.field_selection.source_span.start_byte += 1; }],
  ["wrong_raw_token_digest", b => { b.field_selection.raw_token_sha256 = "0".repeat(64); }],
  ["wrong_response_ordinal", b => { b.session_binding.response_ordinal += 1; }],
  ["missing_http_body_range_map", b => { b.source_map_artifacts = b.source_map_artifacts.filter(a => a.artifact_id !== "http-body-range-map-001"); }],
  ["missing_jcs_node_origin_map", b => { b.source_map_artifacts = b.source_map_artifacts.filter(a => a.artifact_id !== "jcs-node-origin-map-001"); }],
  ["wrong_transformation_profile_digest", b => { b.transformations.find(s => s.stage_id === "stage-content-decode").profile_digest = "0".repeat(64); }],
  ["wrong_field_pointer", b => { b.field_selection.json_pointer = "/height"; }],
  ["broken_representation_parent", b => { b.representations.find(r => r.representation_id === "admitted-json").parent_representation_ids = []; }],
  ["wrong_source_representation_digest", b => { b.representations.find(r => r.representation_id === "admitted-json").sha256 = "0".repeat(64); }]
];
const mapMutation = (maps) => {
  const nodeEntry = bundle.source_map_artifacts.find(a => a.artifact_id === "jcs-node-origin-map-001");
  const map = maps.get(nodeEntry.map_sha256);
  maps.set(nodeEntry.map_sha256, { ...map, nodes: [...map.nodes, structuredClone(map.nodes[0])] });
};
const allMutations = [...mutations, ["ambiguous_jcs_node_map", () => {}, mapMutation]];
for (const [name, mutate, mutateMap] of allMutations) {
  const altered = structuredClone(bundle);
  mutate(altered);
  let rejected = false;
  try { await validate(altered, mutateMap ?? null); } catch { rejected = true; }
  if (!rejected) throw new Error("negative control was accepted: " + name);
}
console.log("PASS synthetic source-map chain: 11 representations; 10 transformations; 3 map artifacts; 1 positive and " + mutations.length + " rejected mutation controls.");
console.log("Claim ceiling: synthetic structural composition only; no TLS, HTTP library, JSON Schema validator, ZK proof or production qualification.");

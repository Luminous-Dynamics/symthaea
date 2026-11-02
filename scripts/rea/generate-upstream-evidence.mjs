#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [reaRootArgument, outputDirectoryArgument] = process.argv.slice(2);
if (!reaRootArgument || !outputDirectoryArgument) {
  throw new Error(
    "Usage: tsx generate-upstream-evidence.mjs <rea-repository-root> <output-directory>",
  );
}

const reaRoot = resolve(reaRootArgument);
const outputDirectory = resolve(outputDirectoryArgument);
const packageJson = JSON.parse(
  await readFile(resolve(reaRoot, "package.json"), "utf8"),
);
const packageBytes = await readFile(resolve(reaRoot, "package.json"));
const generatorBytes = await readFile(new URL(import.meta.url));
const sourceCommit = process.env.REA_REFERENCE_SHA;
const npmVersion = process.env.REA_NPM_VERSION;

if (!sourceCommit || !/^[a-f0-9]{40}$/u.test(sourceCommit)) {
  throw new Error("REA_REFERENCE_SHA must be the pinned 40-character commit");
}
if (!npmVersion) {
  throw new Error("REA_NPM_VERSION must identify the pinned npm runtime");
}

const importFromRea = async (relativePath) =>
  import(pathToFileURL(resolve(reaRoot, "src", relativePath)).href);

const [
  { createEvidence, withEvidenceCallerInputPath, parseEvidence },
  { createAnalysisProfile },
  { TOOL_EFFECTS, toolReturnsTargetContent, annotationsFromEffects },
] = await Promise.all([
  importFromRea("domain/evidence.ts"),
  importFromRea("domain/analysisProfile.ts"),
  importFromRea("contracts/toolEffects.ts"),
]);

const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const provider = { id: "rea.static", name: "REA Static", version: "1.2.3" };
const profile = createAnalysisProfile(provider, { mode: "static" });
const environment = {
  id: "fixture-linux",
  platform: "linux",
  architecture: "x86_64",
  isolation: "process",
};

const cases = [
  {
    name: "apk-profile-complete",
    target: {
      path: "/fixtures/sample.apk",
      sha256: "a".repeat(64),
      format: "apk",
    },
    provider,
    observation: {
      operation: "inspect_artifact",
      parameters: { depth: 1 },
      result: {
        entries: [{ path: "classes.dex", size: 42 }],
        status: "complete",
      },
      analysisProfile: profile,
      locations: [{ kind: "artifact-path", path: "classes.dex" }],
    },
    callerPath: "/fixtures/sample.apk",
  },
  {
    name: "no-subject-null-provider-version",
    target: undefined,
    provider: { id: "rea.transient", name: "Transient Provider", version: null },
    observation: {
      operation: "observe",
      parameters: {},
      result: { observed: true, note: "no artifact subject" },
      rawResult: { source: "external callback" },
      confidence: "inferred",
      authority: "analyst-inference",
      subjectUnavailableReason: "No artifact was supplied",
    },
    callerPath: "/observations/transient.json",
  },
  {
    name: "empty-provider-version-and-empty-target-path",
    target: {
      path: "",
      sha256: "b".repeat(64),
      format: "apk",
    },
    provider: { id: "rea.legacy", name: "Legacy Provider", version: "" },
    observation: {
      operation: "inspect_artifact",
      parameters: {},
      result: { entries: [], status: "complete" },
    },
    callerPath: "",
  },
  {
    name: "partial-raw-binary",
    target: {
      path: "/fixtures/raw.bin",
      sha256: "c".repeat(64),
      format: "raw-binary",
      architecture: "m68k",
    },
    provider,
    observation: {
      operation: "inspect",
      parameters: { mode: "bounded" },
      result: { regions: [{ offset: 0, length: 16 }] },
      relationshipCoverage: {
        status: "partial",
        limitations: ["callee discovery unavailable"],
        limited_directions: ["callees"],
        diagnostics: ["one direction was intentionally not measured"],
      },
    },
    callerPath: "/fixtures/raw.bin",
  },
  {
    name: "unknown-ipa-coverage",
    target: {
      path: "/fixtures/companion.ipa",
      sha256: "d".repeat(64),
      format: "ipa",
    },
    provider,
    observation: {
      operation: "inspect_artifact",
      parameters: {},
      result: { entries: [], status: "partial" },
      relationshipCoverage: {
        status: "unknown",
        reason: "provider cannot inspect runtime relationships",
        limited_directions: ["callers", "callees"],
      },
    },
    callerPath: "/fixtures/companion.ipa",
  },
  {
    name: "locations-and-safe-offset-boundary",
    target: {
      path: "/fixtures/native.bin",
      sha256: "e".repeat(64),
      format: "mach-o",
      architecture: "arm64",
    },
    provider,
    observation: {
      operation: "inspect_native",
      parameters: {},
      result: { facts: ["address", "path", "offset"] },
      relationshipCoverage: {
        status: "complete",
        diagnostics: ["complete within the selected function scope"],
      },
      locations: [
        { kind: "address", address: "0x1000" },
        { kind: "address-range", start: "0x1000", end: "0x1010" },
        { kind: "artifact-path", path: "Frameworks/Native.dylib" },
        { kind: "file-offset", offset: Number.MAX_SAFE_INTEGER },
        {
          kind: "file-offset-range",
          start: 0,
          end: Number.MAX_SAFE_INTEGER,
        },
      ],
    },
    callerPath: "/fixtures/native.bin",
  },
  {
    name: "ecmascript-numbers-and-unicode-keys",
    target: {
      path: "/fixtures/numbers.apk",
      sha256: "f".repeat(64),
      format: "apk",
    },
    provider,
    observation: {
      operation: "inspect_artifact",
      parameters: { unicode: "é/e\u0301" },
      result: {
        "": "BMP private-use",
        "𐀀": "supplementary-plane",
        "composed-é": "é",
        "decomposed-e": "e\u0301",
        numbers: [
          -0,
          -0.0,
          9007199254740993,
          18446744073709551615,
          1e23,
          1e-7,
        ],
      },
      analysisProfile: profile,
      environment,
      rawResult: { native_number_tokens: true },
    },
    callerPath: "/fixtures/numbers.apk",
  },
  {
    name: "gamecube-format-architecture-binding",
    target: {
      path: "/fixtures/game.dol",
      sha256: "1".repeat(64),
      format: "gamecube-dol",
      architecture: "powerpc",
    },
    provider,
    observation: {
      operation: "inspect_artifact",
      parameters: {},
      result: { executable: true, format: "gamecube-dol" },
      confidence: "derived",
      authority: "shipped-artifact",
    },
    callerPath: "/fixtures/game.dol",
  },
  {
    name: "raw-6502-subject",
    target: {
      path: "/fixtures/boot.bin",
      sha256: "2".repeat(64),
      format: "raw-binary",
      architecture: "6502",
    },
    provider,
    observation: {
      operation: "inspect_raw_binary",
      parameters: { load_address: "0x8000" },
      result: { entrypoint: "0x8000" },
      rawResult: { selected_architecture: "6502" },
    },
    callerPath: "/fixtures/boot.bin",
  },
];

await mkdir(outputDirectory, { recursive: false, mode: 0o700 });
const fixtureRecords = [];

for (const [index, item] of cases.entries()) {
  const created = createEvidence(item.target, item.provider, item.observation);
  const evidence = withEvidenceCallerInputPath(created, item.callerPath);
  const raw = Buffer.from(JSON.stringify(evidence, null, 2) + "\n");
  const file =
    "case-" + String(index + 1).padStart(2, "0") + "-" + item.name + ".json";
  await writeFile(resolve(outputDirectory, file), raw, {
    flag: "wx",
    mode: 0o600,
  });
  fixtureRecords.push({
    name: item.name,
    file,
    bytes: raw.length,
    raw_file_sha256: digest(raw),
    evidence_id: evidence.evidence_id,
    analysis_profile_digest: evidence.analysis_profile?.digest ?? null,
    subject_sha256: evidence.subject?.digest.sha256 ?? null,
    relationship_coverage: evidence.relationship_coverage,
  });
}

const displayOnlyMutations = [];
const keyOrderMutations = [];
for (const item of fixtureRecords) {
  const source = JSON.parse(
    await readFile(resolve(outputDirectory, item.file), "utf8"),
  );
  const displayOnly = structuredClone(source);
  if (displayOnly.subject !== null) {
    displayOnly.subject.name = "renamed-display-only.apk";
    displayOnly.subject.local_path = "/relocated/display-only.apk";
  }
  displayOnly.caller_input_path = "/relocated/caller-display-only.json";
  const displayBytes = Buffer.from(JSON.stringify(displayOnly, null, 2) + "\n");
  const displayPath = item.file.replace(/\.json$/u, ".display-only.json");
  await writeFile(resolve(outputDirectory, displayPath), displayBytes, {
    flag: "wx",
    mode: 0o600,
  });
  displayOnlyMutations.push({
    name: item.name,
    file: displayPath,
    bytes: displayBytes.length,
    raw_file_sha256: digest(displayBytes),
    expected_evidence_id: item.evidence_id,
  });

  const reverseObjectKeys = (value) => {
    if (Array.isArray(value)) return value.map(reverseObjectKeys);
    if (value === null || typeof value !== "object") return value;
    return Object.fromEntries(
      Object.entries(value)
        .reverse()
        .map(([key, child]) => [key, reverseObjectKeys(child)]),
    );
  };
  const reordered = reverseObjectKeys(source);
  const reorderedBytes = Buffer.from(JSON.stringify(reordered, null, 2) + "\n");
  const reorderedPath = item.file.replace(/\.json$/u, ".key-order.json");
  await writeFile(resolve(outputDirectory, reorderedPath), reorderedBytes, {
    flag: "wx",
    mode: 0o600,
  });
  keyOrderMutations.push({
    name: item.name,
    file: reorderedPath,
    bytes: reorderedBytes.length,
    raw_file_sha256: digest(reorderedBytes),
    expected_evidence_id: item.evidence_id,
  });
}

const invalidInputs = [
  {
    name: "stale-id-after-semantic-mutation",
    source: "apk-profile-complete",
    mutate(evidence) {
      evidence.predicate_type = evidence.predicate_type + ".tampered";
    },
    expected_error: "identity-mismatch",
  },
  {
    name: "unknown-top-level-field",
    source: "apk-profile-complete",
    mutate(evidence) {
      evidence.unreviewed_field = true;
    },
    expected_error: "schema-violation",
  },
  {
    name: "missing-required-coverage-field",
    source: "apk-profile-complete",
    mutate(evidence) {
      delete evidence.relationship_coverage;
    },
    expected_error: "schema-violation",
  },
  {
    name: "analysis-profile-digest-substitution",
    source: "apk-profile-complete",
    mutate(evidence) {
      evidence.analysis_profile.digest = "0".repeat(64);
    },
    expected_error: "identity-mismatch",
  },
  {
    name: "safe-integer-offset-overflow",
    source: "locations-and-safe-offset-boundary",
    mutate(evidence) {
      evidence.locations[3].offset = Number.MAX_SAFE_INTEGER + 1;
    },
    expected_error: "schema-violation",
  },
  {
    name: "raw-only-architecture-on-apk",
    source: "apk-profile-complete",
    mutate(evidence) {
      evidence.subject.architecture = "m68k";
    },
    expected_error: "schema-violation",
  },
];

const invalidCases = [];
for (const invalid of invalidInputs) {
  const selected = fixtureRecords.find((item) => item.name === invalid.source);
  if (!selected) throw new Error("Missing source fixture for " + invalid.name);
  const source = JSON.parse(
    await readFile(resolve(outputDirectory, selected.file), "utf8"),
  );
  invalid.mutate(source);
  let upstreamRejected = false;
  try {
    parseEvidence(source);
  } catch {
    upstreamRejected = true;
  }
  if (!upstreamRejected) {
    throw new Error(
      "Upstream REA unexpectedly accepted negative control " + invalid.name,
    );
  }
  const raw = Buffer.from(JSON.stringify(source, null, 2) + "\n");
  const file = "negative-" + invalid.name + ".json";
  await writeFile(resolve(outputDirectory, file), raw, {
    flag: "wx",
    mode: 0o600,
  });
  invalidCases.push({
    name: invalid.name,
    file,
    bytes: raw.length,
    raw_file_sha256: digest(raw),
    expected_error: invalid.expected_error,
  });
}

const surrogateCase = cases.find((item) => item.name === "apk-profile-complete");
if (!surrogateCase) throw new Error("Missing source case for surrogate boundary");
const surrogateObservation = {
  ...surrogateCase.observation,
  result: {
    ...surrogateCase.observation.result,
    surrogate_probe: String.fromCharCode(0xd800),
  },
};
const surrogateSource = withEvidenceCallerInputPath(
  createEvidence(
    surrogateCase.target,
    surrogateCase.provider,
    surrogateObservation,
  ),
  surrogateCase.callerPath,
);
parseEvidence(surrogateSource);
const surrogateBytes = Buffer.from(JSON.stringify(surrogateSource, null, 2) + "\n");
const surrogateFile = "stricter-unpaired-surrogate.json";
await writeFile(resolve(outputDirectory, surrogateFile), surrogateBytes, {
  flag: "wx",
  mode: 0o600,
});

const mobileToolEffectNames = [
  "inspect_android_jni_bindings",
  "trace_android_references",
  "inspect_android_package",
  "search_android_classes",
  "inspect_android_class",
  "inspect_android_method",
  "inspect_adb_client",
  "list_adb_devices",
  "inspect_adb_device",
  "list_adb_packages",
  "pull_adb_package",
  "read_adb_logcat",
  "inspect_adb_package",
  "pull_adb_file",
  "push_adb_file",
  "capture_adb_screen",
  "list_adb_processes",
  "list_adb_directory",
  "list_adb_features",
  "list_adb_services",
  "inspect_adb_display",
  "inspect_adb_window",
  "read_adb_setting",
  "collect_adb_bugreport",
  "resolve_adb_packages",
  "install_adb_package",
  "uninstall_adb_package",
  "start_adb_app",
  "start_adb_activity",
  "stop_adb_app",
];

const mobileToolEffects = {};
for (const name of mobileToolEffectNames) {
  if (!Object.hasOwn(TOOL_EFFECTS, name)) {
    throw new Error("Pinned upstream REA is missing audited tool effects for " + name);
  }
  mobileToolEffects[name] = {
    effects: TOOL_EFFECTS[name],
    annotations: annotationsFromEffects(TOOL_EFFECTS[name]),
    returns_target_content: toolReturnsTargetContent(name),
  };
}
const mobileToolEffectBytes = Buffer.from(
  JSON.stringify(mobileToolEffects, null, 2) + "\n",
);
const mobileToolEffectFile = "mobile-tool-effects-v1.json";
await writeFile(resolve(outputDirectory, mobileToolEffectFile), mobileToolEffectBytes, {
  flag: "wx",
  mode: 0o600,
});
const mobileToolEffectManifest = {
  file: mobileToolEffectFile,
  tool_count: mobileToolEffectNames.length,
  raw_file_sha256: digest(mobileToolEffectBytes),
};

const manifest = {
  schema: "luminous.rea-upstream-evidence-corpus.v1",
  producer: {
    repository: "https://github.com/morluto/rea",
    commit: sourceCommit,
    package: packageJson.name,
    package_version: packageJson.version,
    package_json_sha256: digest(packageBytes),
    node_version: process.version,
    npm_version: npmVersion,
    tsx_version: packageJson.devDependencies?.tsx ?? null,
  },
  generator: {
    script_sha256: digest(generatorBytes),
    command: "tsx scripts/rea/generate-upstream-evidence.mjs",
  },
  fixture_count: fixtureRecords.length,
  fixtures: fixtureRecords,
  display_only_mutations: displayOnlyMutations,
  key_order_mutations: keyOrderMutations,
  negative_controls: invalidCases,
  stricter_importer_cases: [
    {
      name: "unpaired-utf16-surrogate",
      file: surrogateFile,
      bytes: surrogateBytes.length,
      raw_file_sha256: digest(surrogateBytes),
      upstream_object_parse_accepted: true,
      expected_rust_error: "invalid-json",
      policy: "Rust String cannot represent isolated UTF-16 surrogate code units; raw importer rejects rather than normalizes or replaces them.",
    },
  ],
  mobile_tool_effects: mobileToolEffectManifest,
};
const manifestBytes = Buffer.from(JSON.stringify(manifest, null, 2) + "\n");
await writeFile(resolve(outputDirectory, "manifest.json"), manifestBytes, {
  flag: "wx",
  mode: 0o600,
});
await writeFile(
  resolve(outputDirectory, "manifest.provenance.json"),
  JSON.stringify(
    {
      schema: "luminous.rea-upstream-evidence-corpus-receipt.v1",
      manifest_sha256: digest(manifestBytes),
      fixture_count: fixtureRecords.length,
      display_only_mutation_count: displayOnlyMutations.length,
      key_order_mutation_count: keyOrderMutations.length,
      negative_control_count: invalidCases.length,
      stricter_importer_count: 1,
      mobile_tool_effects: mobileToolEffectManifest,
      producer: manifest.producer,
      generator: manifest.generator,
    },
    null,
    2,
  ) + "\n",
  { flag: "wx", mode: 0o600 },
);

process.stdout.write(JSON.stringify(manifest, null, 2) + "\n");

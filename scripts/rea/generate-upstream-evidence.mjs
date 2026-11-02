#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const [reaRootArgument, outputArgument] = process.argv.slice(2);
if (!reaRootArgument || !outputArgument) {
  throw new Error(
    "Usage: tsx generate-upstream-evidence.mjs <rea-repository-root> <output-json>",
  );
}

const reaRoot = resolve(reaRootArgument);
const outputPath = resolve(outputArgument);
const packageJson = JSON.parse(
  await readFile(resolve(reaRoot, "package.json"), "utf8"),
);
const packageBytes = await readFile(resolve(reaRoot, "package.json"));
const generatorBytes = await readFile(fileURLToPath(import.meta.url));
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

const [{ createEvidence, withEvidenceCallerInputPath }, { createAnalysisProfile }] =
  await Promise.all([
    importFromRea("domain/evidence.ts"),
    importFromRea("domain/analysisProfile.ts"),
  ]);

const provider = {
  id: "rea.static",
  name: "REA Static",
  version: "1.2.3",
};
const targetPath = "/fixtures/sample.apk";
const profile = createAnalysisProfile(provider, { mode: "static" });

const evidence = createEvidence(
  {
    path: targetPath,
    sha256: "a".repeat(64),
    format: "apk",
  },
  provider,
  {
    operation: "inspect_artifact",
    parameters: { depth: 1 },
    result: {
      entries: [{ path: "classes.dex", size: 42 }],
      status: "complete",
    },
    analysisProfile: profile,
    locations: [{ kind: "artifact-path", path: "classes.dex" }],
  },
);

const withCallerPath = withEvidenceCallerInputPath(evidence, targetPath);
const raw = JSON.stringify(withCallerPath, null, 2) + "\n";
await writeFile(outputPath, raw, { flag: "wx", mode: 0o600 });

const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const receipt = {
  schema: "luminous.rea-upstream-evidence-generator-receipt.v1",
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
  output: {
    path: outputPath,
    bytes: Buffer.byteLength(raw),
    raw_file_sha256: digest(Buffer.from(raw)),
    evidence_id: withCallerPath.evidence_id,
    analysis_profile_digest: withCallerPath.analysis_profile?.digest ?? null,
  },
};
const receiptPath = `${outputPath}.provenance.json`;
await writeFile(receiptPath, JSON.stringify(receipt, null, 2) + "\n", {
  flag: "wx",
  mode: 0o600,
});

process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");

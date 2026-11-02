#!/usr/bin/env node
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [reaRootArgument, outputArgument] = process.argv.slice(2);
if (!reaRootArgument || !outputArgument) {
  throw new Error(
    "Usage: node generate-upstream-evidence.mjs <rea-repository-root> <output-json>",
  );
}

const reaRoot = resolve(reaRootArgument);
const outputPath = resolve(outputArgument);
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

process.stdout.write(
  JSON.stringify(
    {
      upstream_evidence_id: withCallerPath.evidence_id,
      analysis_profile_digest: withCallerPath.analysis_profile?.digest ?? null,
      output_bytes: Buffer.byteLength(raw),
      output_path: outputPath,
    },
    null,
    2,
  ) + "\n",
);

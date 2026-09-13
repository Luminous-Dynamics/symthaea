# LL-009AG — Frozen Site01 campaign root

LL-009AG closes a reproducibility gap in the Site01 / Connecting Ridge lunar-terrain evidence chain. It does **not** create another terrain model, uncertainty formula, visibility result, or probability theorem. It freezes the exact campaign inputs and refuses to combine evidence after lineage drift.

## State theorem

The campaign is monotonic:

`PREPARED -> FROZEN -> FINALIZED`

`PREPARED` binds the exact repository head, campaign policy, protected scripts/configs, stage plan, and three independent runtime capsules (`acquisition`, `gis`, `analysis`). Environment or protected-input drift at this stage means the preparation is invalid and the campaign must be prepared again.

`FROZEN` is the point at which evidence begins. It binds an explicit evidence manifest, the exact present evidence-root file set, every present artifact's outer SHA-256 and byte count, canonical receipt self-hashes when requested, and verified dependency edges. After this transition, substitution or drift is fatal; evidence from different lineages must not be combined.

`FINALIZED` replays all PREPARED and FROZEN bindings, verifies the exact evidence-root file set, binds an explicit semantic assertion, and applies the policy's promotion ceiling. It does not recompute numerical terrain or visibility values.

## Explicit availability classes

Every logical artifact is declared as exactly one of:

- `required` — must exist and fully verify or the campaign fails;
- `optional_diagnostic` — may be absent, but if present it must pass exactly the same hashing, self-hash, dependency, and closure checks as required evidence;
- `not_yet_available` — an explicit future/missing evidence surface. It has no path and cannot silently masquerade as success.

A campaign containing any `not_yet_available` node is `incomplete_declared` and has `promotion_eligible=false`. Such a campaign may still be FINALIZED as an immutable integrity snapshot, but that final root is not promotable scientific evidence. Missing optional diagnostics are recorded separately and do not become positive evidence.

## No hidden discovery

The evidence manifest is authoritative and explicit. Each present artifact supplies a logical artifact ID, a normalized relative path under the evidence root, exact logical dependencies, a requirement class, and a `self_hash_mode` of either `none` or `canonical_receipt_sha256`.

LL-009AG never assigns a scientific role from a filename, glob, extension, or directory name. The presence of a field called `receipt_sha256` is not enough to opt an artifact into receipt verification; the manifest must explicitly declare the canonical self-hash mode.

The evidence directory is closed over **present** evidence: every regular file must correspond to exactly one present manifest node. Symlinks and special files fail closed. Required missing files fail closed. Optional missing files and declared-unavailable nodes remain explicit manifest state rather than hidden filesystem inference.

## Dependency edges are evidence

A declared DAG edge is not trusted by itself. Every present child must contain the exact identity of every declared parent at a reviewed JSON field path. The manifest declares one `dependency_binding` per parent using either:

- `receipt_sha256` — the parent's verified canonical receipt identity; or
- `sha256` — the exact SHA-256 of the parent file bytes.

For example:

```json
{
  "id": "visibility",
  "path": "visibility.json",
  "requirement": "required",
  "dependencies": ["horizon"],
  "dependency_bindings": [
    {
      "dependency_id": "horizon",
      "field_path": ["k_horizon_pack_sha256"],
      "identity": "sha256"
    }
  ],
  "self_hash_mode": "canonical_receipt_sha256"
}
```

The field path is data, not a heuristic. LL-009AG does not search a receipt for a plausible hash. A self-consistent child receipt that points to a different valid upstream object therefore fails closed.

A present artifact may not depend on an absent optional diagnostic or a `not_yet_available` node. Dependency cycles, duplicate logical IDs, duplicate paths, duplicate bindings, and incomplete binding coverage are rejected before evidence can freeze.

## Environment separation and capture

The campaign deliberately requires three capsules rather than pretending one Python installation governs all stages:

- `acquisition`: network/source-byte acquisition only;
- `gis`: extraction, role binding, raster/materialization, cross-method terrain work;
- `analysis`: horizon, uncertainty, visibility, semantic reconciliation, and finalization.

An environment capsule is not an arbitrary note. Schema `ll009ag.environment-capsule.v1` requires five runtime sections: `python`, `platform`, `packages`, `libraries`, and `environment`. Python identity includes implementation, exact version, and SHA-256 of the resolved interpreter executable. Platform identity includes system and machine. Package and native-library versions required by the profile are explicit policy obligations. Every determinism-sensitive environment variable named by policy must be present in the capsule, with an unset variable represented explicitly as JSON `null`.

The checked Site01 policy currently requires OpenSSL identity for acquisition; NumPy, Rasterio, GDAL and PROJ identity for GIS; NumPy for analysis; and profile-specific locale, timezone, Python-hash, GIS-data and numerical-thread environment variables. These are identity requirements, not an automatic assertion that any observed version is scientifically acceptable; upstream toolchain receipts/policies still govern their own promotion requirements.

`scripts/capture_ll009ag_environment.py` captures this information from the environment actually executing a stage. It performs no network access, probes only the versions required by the checked policy, hashes the active resolved Python executable, and uses no-clobber output semantics. A differing capsule cannot silently overwrite an earlier one.

For example, execute each capture inside the environment that will actually run that profile:

```bash
python scripts/capture_ll009ag_environment.py \
  --policy configs/lunar_transport/ll009ag_site01_campaign_v1.json \
  --profile acquisition \
  --output campaign/runtime/acquisition.json
```

Repeat separately for `gis` and `analysis` from their actual execution environments. Do not capture all three from one shell merely to satisfy the schema.

The capsule bytes and canonical payload are both bound by the PREPARED receipt. Only `source_acquisition` is network-authorized by the checked-in campaign plan; numerical stages are explicitly offline.

## Protected execution surface

PREPARED does not merely protect AG's own orchestrator. The checked policy enumerates the exact materialized code/configuration surface used by the real Site01 evidence chain from K through AF, including NASA acquisition, GIS materialization, uncertainty semantics, clone/spatial-support logic, Product90 RMS envelopes, memberwise hybrid composition, S/Z visibility semantics, SDEM cross-method/calibration audits, and AE/AF stress analysis. The dedicated qualification workflow verifies every protected path exists on the exact checked head before AG logic can pass.

The campaign policy itself is independently byte-hashed into PREPARED, so changing the protected-file list also invalidates preparation.

## Semantic ceiling

The checked-in campaign policy currently caps finalization at `hybrid_scenario_sampled_visibility`.

Promotion to that class requires frozen artifact ID `ll009z_visibility_reconciliation` to appear in the final semantic assertion's explicit basis. `risk_qualified_visibility` and `deterministic_visibility` are disabled.

This is intentional. LL-009AG does not create a joint Q × Product90 probability theorem, does not convert Product90 RMS into a certified true-second-moment bound, does not upgrade spatial support, and does not treat an SDEM as independent ground truth. An incomplete campaign root is non-promotable regardless of the semantic class carried by its present evidence.

## Receipt schemas

The three state receipts are `ll009ag.campaign-preparation-receipt.v1`, `ll009ag.campaign-freeze-receipt.v1`, and `ll009ag.campaign-finalization-receipt.v1`. Receipts use canonical JSON with sorted keys, compact separators, UTF-8, and a trailing newline. `receipt_sha256` is SHA-256 over the same canonical object with that field omitted.

The FROZEN receipt records present, missing-optional, and declared-unavailable nodes separately, together with every successfully verified dependency binding. FINALIZED records `campaign_completeness` and `promotion_eligible` rather than inferring completeness from file presence.

## Typical real-campaign flow

After capturing the three profile environments, preparation requires the exact repository head plus explicit paths to those capsules:

```bash
python scripts/freeze_ll009ag_site01_campaign.py prepare \
  --policy configs/lunar_transport/ll009ag_site01_campaign_v1.json \
  --repo-root . \
  --repo-head "$EXACT_HEAD" \
  --runtime-root campaign/runtime \
  --env acquisition=acquisition.json \
  --env gis=gis.json \
  --env analysis=analysis.json \
  --output campaign/ll009ag-prepared.json
```

After real evidence files have been produced and an explicit manifest reviewed, freeze them:

```bash
python scripts/freeze_ll009ag_site01_campaign.py freeze \
  --policy configs/lunar_transport/ll009ag_site01_campaign_v1.json \
  --repo-root . --repo-head "$EXACT_HEAD" \
  --runtime-root campaign/runtime \
  --preparation campaign/ll009ag-prepared.json \
  --manifest campaign/evidence-manifest.json \
  --evidence-root campaign/evidence \
  --output campaign/ll009ag-frozen.json
```

Finalization consumes a separate reviewed semantic assertion and cannot exceed the checked-in policy ceiling:

```bash
python scripts/freeze_ll009ag_site01_campaign.py finalize \
  --policy configs/lunar_transport/ll009ag_site01_campaign_v1.json \
  --repo-root . --repo-head "$EXACT_HEAD" \
  --runtime-root campaign/runtime \
  --preparation campaign/ll009ag-prepared.json \
  --freeze campaign/ll009ag-frozen.json \
  --manifest campaign/evidence-manifest.json \
  --evidence-root campaign/evidence \
  --assertion campaign/semantic-assertion.json \
  --output campaign/ll009ag-finalized.json
```

`verify` repeats the entire replay from frozen inputs and requires canonical equality with the supplied FINALIZED receipt.

## Qualification scope

The dedicated workflow compiles the AG tools, executes the environment-capture self-test, verifies that the complete checked protected-file surface exists, and runs a dependency-free synthetic state-machine campaign. The synthetic campaign checks deterministic PREPARED/FROZEN/FINALIZED replay; structured environment contracts; runtime and protected-tool drift; artifact substitution; exact receipt- and file-hash dependency bindings; valid-but-wrong upstream substitution; required, optional-diagnostic, and declared-unavailable handling; closed evidence-root enforcement; invalid inner receipt self-hashes; dependency cycles; missing bindings; unavailable semantic bases; missing semantic provenance; and attempted over-promotion.

The workflow downloads no NASA source bytes and produces no scientific Site01 result. A green workflow therefore qualifies LL-009AG's campaign-control logic only; it is not evidence that the Product104 Connecting Ridge archive has been acquired, that a real SDEM has been role-bound, or that any visibility class stronger than the existing upstream evidence has been established.

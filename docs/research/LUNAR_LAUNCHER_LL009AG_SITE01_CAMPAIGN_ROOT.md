# LL-009AG — Frozen Site01 campaign root

LL-009AG closes a reproducibility gap in the Site01 / Connecting Ridge lunar-terrain evidence chain. It does **not** create another terrain model, uncertainty formula, visibility result, calibration claim, or probability theorem. It freezes one exact campaign lineage and refuses to combine evidence after code, environment, source, or receipt drift.

## State theorem

The campaign is monotonic:

`PREPARED -> FROZEN -> FINALIZED`

`PREPARED` binds the exact repository head, campaign policy, protected execution/configuration files, stage plan, and three independent runtime capsules (`acquisition`, `gis`, `analysis`). Drift before evidence begins invalidates preparation and requires re-preparation.

`FROZEN` is the evidence-lineage boundary. It binds an explicit evidence manifest, every present artifact's exact file SHA-256 and byte count, any declared native receipt self-hash, the exact evidence-root file set, explicit availability state, and cryptographically verified dependency edges. After this transition, substitution or drift is fatal; evidence from a different prepared root cannot be mixed into the campaign.

`FINALIZED` replays PREPARED and FROZEN exactly, binds a separate semantic assertion, and applies the checked policy's claim ceiling. It performs no terrain or visibility recomputation.

## Explicit availability

Manifest schema `ll009ag.evidence-manifest.v2` requires every logical artifact `requirement` to be exactly one of:

- `required` — the file must exist and fully verify;
- `optional_diagnostic` — the file may be absent, but if present is frozen and verified exactly like required evidence;
- `not_yet_available` — an explicit missing/future evidence surface. It has no path and cannot silently count as success.

The evidence directory is closed over declared **present** files. Any undeclared regular file is an error. Missing required files are errors. Missing optional diagnostics are recorded explicitly. `not_yet_available` nodes are recorded explicitly and cannot be used as semantic basis.

If at least one node is `not_yet_available`, FROZEN records:

- `campaign_completeness = incomplete_declared`;
- `promotion_eligible = false`.

Such a campaign may still be FINALIZED as an immutable integrity snapshot, but FINALIZED records `effective_evidence_class = null` and `promotion_status = blocked_incomplete_declared`. This separates successful integrity sealing from scientific promotion.

## Native receipt self-hashes

AG does not assume every LL-009 producer serializes JSON the way AG does. The manifest declares one of three `self_hash_mode` values per artifact:

- `none` — no inner receipt self-hash is claimed;
- `ag_compact_receipt_sha256` — AG's sorted compact canonical JSON plus trailing newline;
- `ll009_indent2_receipt_sha256` — the established LL-009 sorted, indent-2 canonical JSON (`separators=(',', ': ')`) plus trailing newline, used by receipts such as AA and Z.

The selected native contract is replayed against the child JSON before its `receipt_sha256` is accepted. Merely containing a field named `receipt_sha256` is never enough.

AG's own PREPARED/FROZEN/FINALIZED receipts use the compact AG canonicalization. Upstream scientific receipts keep their native contract.

## Dependency edges are executable evidence

A DAG edge is not accepted because the manifest says two files are related. Every **present child** must itself contain the exact identity of every declared present parent.

Dependencies are named separately from their binding proofs. Each present child declares a dependency ID and exactly one reviewed binding for that parent:

```json
{
  "dependencies": ["ll009y_horizon"],
  "dependency_bindings": [
    {
      "dependency_id": "ll009y_horizon",
      "identity": "sha256",
      "field_path": ["k_horizon_pack_sha256"]
    }
  ]
}
```

`identity` is either:

- `sha256` — SHA-256 of the exact frozen parent bytes; or
- `receipt_sha256` — the parent's verified native receipt self-hash.

`field_path` is an explicit ordered JSON-object key path. AG never searches a child receipt for a plausible-looking hash. It resolves the reviewed path, requires a 64-hex digest, computes the declared parent identity independently, and requires exact equality.

The FROZEN receipt records every accepted binding on the frozen child: parent ID, identity mode, reviewed field path, and the independently verified digest value. A self-consistent child receipt that references a different valid upstream object therefore fails closed.

A present child may not depend on an absent optional diagnostic or a `not_yet_available` node. Duplicate logical IDs, duplicate paths, duplicate parent edges, unknown parents, self-dependencies, and cycles are rejected before freeze.

Example manifest fragment:

```json
{
  "schema_version": "ll009ag.evidence-manifest.v2",
  "study_id": "ll009-site01-connecting-ridge-v1",
  "artifacts": [
    {
      "id": "ll009q_clone_ensemble",
      "requirement": "required",
      "path": "q.json",
      "dependencies": [],
      "dependency_bindings": [],
      "self_hash_mode": "ll009_indent2_receipt_sha256"
    },
    {
      "id": "ll009z_visibility_reconciliation",
      "requirement": "required",
      "path": "z.json",
      "dependencies": ["ll009q_clone_ensemble"],
      "dependency_bindings": [
        {
          "dependency_id": "ll009q_clone_ensemble",
          "identity": "sha256",
          "field_path": ["q_receipt_sha256"]
        }
      ],
      "self_hash_mode": "ll009_indent2_receipt_sha256"
    },
    {
      "id": "future_joint_coupling_theorem",
      "requirement": "not_yet_available",
      "dependencies": [],
      "dependency_bindings": [],
      "self_hash_mode": "none"
    }
  ]
}
```

The example is structural only; the reviewed real Site01 manifest must use the exact fields actually exposed by each frozen producer.

## Environment separation

AG deliberately requires three environment capsules instead of pretending one Python installation governs every stage:

- `acquisition` — source-byte acquisition/archive integrity;
- `gis` — extraction, raster materialization, role binding, and cross-method terrain work;
- `analysis` — horizon, uncertainty, visibility, semantic reconciliation, and stress analysis.

Schema `ll009ag.environment-capsule.v1` binds Python implementation/version and SHA-256 of the resolved interpreter executable, platform identity, required package/native-library versions, and determinism-sensitive environment variables. Unset declared variables are represented explicitly as JSON `null`.

`scripts/capture_ll009ag_environment.py` captures those values from the environment actually executing the profile and refuses to overwrite a differing capsule. Only the `source_acquisition` stage is network-authorized by the checked policy; numerical stages are offline.

## Protected execution surface

PREPARED protects the actual Site01 implementation surface, not merely AG's orchestration code. The checked policy enumerates K through AF scripts/configurations including NASA acquisition, raster materialization, Q/R uncertainty support, Product90 V/W/X semantics, Y hybrid composition, S/Z visibility semantics, Product104 SDEM work, AB/AC/AD cross-method audits, and AE/AF stress analysis.

The dedicated workflow no longer treats an arbitrary file count as proof of coverage. It requires a named cross-section of the scientific stack to remain inside `protected_files`, verifies every protected path materializes on the exact checked head, then hashes all of them through the same PREPARED binding implementation.

Changing the policy or protected surface changes the preparation root and requires a new campaign lineage.

## Semantic ceiling

The checked policy caps AG at `hybrid_scenario_sampled_visibility`. `risk_qualified_visibility` and `deterministic_visibility` remain disabled.

AG does not create a joint Q × Product90 probability theorem, does not certify Product90 RMS as a true-second-moment upper bound, does not upgrade R spatial support, does not convert AE/AF stress lambda into a calibrated true multiplier, and does not treat Product104 SDEM as independent ground truth. No site-safety, delivered-power, RF-link, operations, or mission authority is inferred from terrain visibility receipts.

Even a semantically valid Z basis cannot become an effective promoted class while the campaign contains a declared unavailable node.

## Receipt schemas

AG emits:

- `ll009ag.campaign-preparation-receipt.v1`;
- `ll009ag.campaign-freeze-receipt.v2`;
- `ll009ag.campaign-finalization-receipt.v2`.

PREPARED binds code/config/environment identity. FROZEN records explicit artifact states, exact present-file closure, native self-hash verification, campaign completeness, promotion eligibility, and every verified dependency edge. FINALIZED replays the entire frozen root and records both the asserted class and the effective class after completeness gating.

## Real-campaign flow

Capture each profile inside the environment that will actually run it:

```bash
python scripts/capture_ll009ag_environment.py \
  --policy configs/lunar_transport/ll009ag_site01_campaign_v1.json \
  --profile acquisition \
  --output campaign/runtime/acquisition.json
```

Repeat separately for `gis` and `analysis`.

Prepare the exact repository subject:

```bash
python scripts/freeze_ll009ag_site01_campaign.py prepare \
  --policy configs/lunar_transport/ll009ag_site01_campaign_v1.json \
  --repo-root . --repo-head "$EXACT_HEAD" \
  --runtime-root campaign/runtime \
  --env acquisition=acquisition.json \
  --env gis=gis.json \
  --env analysis=analysis.json \
  --output campaign/ll009ag-prepared.json
```

After real evidence files and a reviewed v2 manifest exist, freeze them:

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

Finalization consumes a separate reviewed semantic assertion and cannot exceed the checked policy ceiling:

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

`verify` deterministically replays the complete finalization and requires exact equality with the supplied FINALIZED receipt.

## Qualification scope

The dedicated AG workflow compiles the campaign tools, qualifies environment-capture logic, proves the named protected scientific surface remains frozen, and executes a dependency-free synthetic state-machine campaign.

The synthetic campaign now exercises:

- deterministic PREPARED/FROZEN/FINALIZED replay;
- environment and protected-tool drift;
- required, optional-diagnostic, and declared-unavailable states;
- incomplete-campaign promotion blocking;
- native upstream receipt self-hash semantics, including rejection under the wrong declared canonicalization;
- receipt-hash and file-hash dependency edges;
- a self-consistent child with the wrong embedded parent digest;
- closed evidence-root enforcement;
- receipt tampering;
- dependency cycles;
- absent-artifact semantic bases;
- missing semantic provenance; and
- attempted over-promotion.

This qualification downloads no NASA source bytes and produces no scientific Site01 result. A green AG workflow would qualify campaign-control logic only. A real Site01 final root does not exist until the exact NASA source locks/bytes and every configured required scientific receipt have executed under one frozen lineage.

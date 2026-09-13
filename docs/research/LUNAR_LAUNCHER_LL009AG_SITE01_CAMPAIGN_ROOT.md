# LL-009AG — Frozen Site01 campaign root

LL-009AG closes a reproducibility gap in the Site01 / Connecting Ridge lunar-terrain evidence chain. It does **not** create another terrain model, uncertainty formula, visibility result, or probability theorem. It freezes the exact campaign inputs and refuses to combine evidence after lineage drift.

## State theorem

The campaign is monotonic:

`PREPARED -> FROZEN -> FINALIZED`

`PREPARED` binds the exact repository head, campaign policy, protected scripts/configs, stage plan, and three independent runtime capsules (`acquisition`, `gis`, `analysis`). Environment or protected-input drift at this stage means the campaign must be prepared again.

`FROZEN` is the point at which evidence begins. It binds an explicit evidence manifest, a closed evidence-root file set, every artifact's outer SHA-256 and byte count, the declared dependency DAG, and—only when explicitly requested by that artifact entry—the artifact's canonical `receipt_sha256`. After this transition, substitution or drift is fatal; evidence from different lineages must not be combined.

`FINALIZED` replays all PREPARED and FROZEN bindings, verifies the exact evidence-root file set, binds an explicit semantic assertion, and applies the policy's promotion ceiling. It does not recompute numerical terrain or visibility values.

## No hidden discovery

The evidence manifest is authoritative and explicit. Each artifact supplies a logical artifact ID, a normalized relative path under the evidence root, exact logical dependencies, and a `self_hash_mode` of either `none` or `canonical_receipt_sha256`.

LL-009AG never assigns a scientific role from a filename, glob, extension, or directory name. The presence of a field called `receipt_sha256` is not enough to opt an artifact into receipt verification; the manifest must explicitly declare the canonical self-hash mode.

The evidence directory is closed: every regular file must appear exactly once in the manifest, and every manifest path must exist. Symlinks and special files fail closed.

## Environment separation

The campaign deliberately requires three capsules rather than pretending one Python installation governs all stages:

- `acquisition`: network/source-byte acquisition only;
- `gis`: extraction, role binding, raster/materialization, cross-method terrain work;
- `analysis`: horizon, uncertainty, visibility, semantic reconciliation, and finalization.

Each capsule uses schema `ll009ag.environment-capsule.v1`, names its exact profile, and contains a non-empty `runtime` object. The capsule bytes and canonical payload are both bound by the PREPARED receipt.

Only `source_acquisition` is network-authorized by the checked-in policy. Numerical stages are explicitly offline in the campaign plan.

## Semantic ceiling

The checked-in campaign policy currently caps finalization at `hybrid_scenario_sampled_visibility`.

Promotion to that class requires frozen artifact ID `ll009z_visibility_reconciliation` to appear in the final semantic assertion's explicit basis. `risk_qualified_visibility` and `deterministic_visibility` are disabled.

This is intentional. LL-009AG does not create a joint Q × Product90 probability theorem, does not convert Product90 RMS into a certified true-second-moment bound, does not upgrade spatial support, and does not treat an SDEM as independent ground truth.

## Receipt schemas

The three state receipts are `ll009ag.campaign-preparation-receipt.v1`, `ll009ag.campaign-freeze-receipt.v1`, and `ll009ag.campaign-finalization-receipt.v1`. Receipts use canonical JSON with sorted keys, compact separators, UTF-8, and a trailing newline. `receipt_sha256` is SHA-256 over the same canonical object with that field omitted.

## Typical real-campaign flow

Preparation requires the exact repository head plus explicit paths to all three environment capsules:

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

The dedicated workflow performs only Python compilation and a dependency-free synthetic state-machine campaign. The synthetic campaign checks deterministic replay and fail-closed behavior for runtime drift, protected-tool drift, artifact substitution, missing or extra evidence files, invalid inner receipt self-hashes, dependency cycles, missing semantic provenance, and attempted over-promotion.

The workflow downloads no NASA source bytes and produces no scientific Site01 result. A green workflow therefore qualifies LL-009AG's campaign-control logic only; it is not evidence that the Product104 Connecting Ridge archive has been acquired, that a real SDEM has been role-bound, or that any visibility class stronger than the existing upstream evidence has been established.

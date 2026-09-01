# St. Lucia R0 S3 Preflight v1

Status: **post-asset-plan / pre-object-body-download**

This contract freezes an authenticated metadata-only availability check for the 29 exact objects in the qualified St. Lucia R0 asset plan. It is stacked on PR #264 at planner head `90b557dfa69f0b9d228b8bc02a5907b5b8e58346`.

## Frozen input

The only admissible asset plan for this preflight has:

```text
schema                 symthaea-st-lucia-r0-asset-plan/v1
tool_version           1.2.0
asset count            29
internal plan SHA-256  53a12535acc9f02bf62d78c54c3a6b0631d6ba69e22eeb187e2ad20ecb330c46
plan-file SHA-256      fa5dffc399fd0c120cdc59b479b0952862561d7db0ea10528623d8809133dff2
endpoint               https://eodata.dataspace.copernicus.eu/
bucket                 eodata
```

Every row must preserve the reviewed `s3://eodata/...` URI and exact `(endpoint,bucket,key)` decomposition.

## Network action boundary

The executable is:

```text
scripts/research/st_lucia_r0_s3_preflight.py
schema       symthaea-st-lucia-r0-s3-preflight/v1
tool_version 1.0.0
```

For each of the 29 frozen rows, it may invoke exactly:

```text
aws s3api head-object
```

against the frozen CDSE endpoint/bucket/key.

It must not invoke `get-object`, `aws s3 cp`, recursive listing, range reads, preview access, raster decoding, or any object-body request.

CDSE documentation identifies `https://eodata.dataspace.copernicus.eu/` as the default S3-compatible endpoint and documents both AWS CLI access and `head_object` as an object-size check.

Reference: https://documentation.dataspace.copernicus.eu/APIs/S3.html

## Credentials

Authentication requires CDSE S3 access/secret credentials. They must be supplied to the process via:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

Credential values must never be written into repository files, command arguments, receipts, or logs. The preflight receipt records only that the environment source was used.

## Receipt

For successful HEAD responses, retain a deliberately small metadata surface including:

- `ContentLength`;
- `ETag`;
- `LastModified`;
- `ContentType`;
- version/checksum fields when actually supplied by the server;
- selected standard content headers.

Arbitrary custom S3 metadata is omitted. `ETag` is evidence metadata only and must not be interpreted as SHA-256 or even necessarily as a content hash.

For a failed HEAD call, retain only a normalized failure state, process return code, and SHA-256 of stderr. Do not retain raw stderr in the scientific receipt.

The final receipt is content-addressed and records the exact plan identities, endpoint, bucket, AWS CLI version, Python/platform environment, per-object result, available/failed counts, and execution timestamp.

## Interpretation

A successful 29/29 preflight establishes only that every reviewed locator was authenticated and visible to `HeadObject` at the time of the run, with recorded server metadata. It does not establish byte identity because object bodies have not been acquired and SHA-256 has not yet been computed locally.

A missing, unauthorized, or otherwise failed object is a valid reportable preflight outcome. It must not trigger post-hoc product or asset substitution.

## Hard stop

After the preflight receipt is emitted, stop again before object-body download. Review:

- 29/29 vs any failures;
- exact plan hashes;
- object lengths and server metadata;
- absence of credentials/raw error text;
- deterministic receipt hash.

Only then may a separately reviewed acquisition tool fetch the 29 exact object bodies and immediately compute local SHA-256 + byte length before raster inspection or feature computation.

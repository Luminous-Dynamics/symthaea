# Protein Dynamics Source Acquisition Gate

Use this before acquiring any PeptoneBench archive. The upstream repository points to two differently purposed artifacts: the dataset archive used by the download helper/container build, and a precomputed predictions archive that may allow scoring-only reproduction. Their current sizes/checksums have not yet been recorded in this project's evidence bundle.

## 1. Inspect metadata only

The first two commands query the official Zenodo Records API and print the exact file metadata. They do not download the archive.

~~~bash
python3 scripts/research/fetch_verified_zenodo_file.py --record-id 17306061 --filename Predictions.tar.gz

python3 scripts/research/fetch_verified_zenodo_file.py --record-id 17306061 --filename PeptoneDBs.tar.gz
~~~

Review the reported exact file name, record DOI/version, metadata-response SHA-256, file size, checksum algorithm/value, and the download URL. The script fails closed if the named file is missing, ambiguous, has unsupported checksum metadata, or points outside HTTPS Zenodo. A missing file at this record is a reason to stop and verify the canonical record/version—not to guess a replacement URL.

## 2. Download only after setting a deliberate cap

Choose the byte cap from the metadata output and available disk space. Do not use a guessed unlimited cap. For example, after setting an explicit reviewed limit in bytes:

~~~bash
python3 scripts/research/fetch_verified_zenodo_file.py --record-id 17306061 --filename Predictions.tar.gz --download --output-dir runs/protein-dynamics-source/ --max-bytes "$REVIEWED_MAX_BYTES"
~~~

The tool downloads to a temporary file, enforces the metadata size and configured cap while streaming, verifies the checksum declared by Zenodo, computes a local SHA-256, preserves a hash-addressed raw record-metadata snapshot and a verification sidecar, and only then promotes the archive to its final path. It does not extract archives. Existing files are not overwritten unless explicitly requested with --force.

A provider checksum establishes agreement with the checksum currently returned by the provider over HTTPS; it is not a proof of biological validity or an independent signature by the authors. Preserve the local SHA-256 and exact metadata snapshot so the acquired bytes remain identifiable later.

## 3. Inspect archive members before extracting anything

Keep the verified original archive immutable. List members first, for example:

~~~bash
tar -tzf runs/protein-dynamics-source/Predictions.tar.gz | head -100
~~~

Review the member names and locate the smallest candidate case, corresponding prediction files, and required experimental-observable data. Do not extract the full archive by default. Before selective extraction, reject absolute paths, parent traversal components, unexpected links, and any member outside the intended bundle directory. Record selected member names and hashes in the run manifest.

If the predictions archive can supply the generated ensemble plus forward-model outputs needed for one score case, use it to reproduce PeptoneBench's scorer first. That tests evaluator reproducibility without conflating it with rebuilding the UCBShift/Pepsi forward-model container. Rebuild/regenerate forward models as a separate later gate.

## Tests and current status

Run the offline acquisition tests with:

~~~bash
python3 -m unittest discover -s scripts/research -p 'test_fetch_verified_zenodo_file.py' -v
~~~

Seventeen mocked tests cover metadata shape, unsupported checksums, untrusted URLs and embedded credentials, redirect-to-host checks for both metadata and binary fetches, size caps, checksum/size mismatch, partial-file cleanup, existing asset/sidecar overwrite protections, and raw metadata snapshots. All 17 tests passed offline using mocked responses; no live Zenodo query or dataset/prediction archive download has been performed. The fetcher follows the official [Zenodo Records API](https://developers.zenodo.org/) response fields for file size, checksum, and download links.

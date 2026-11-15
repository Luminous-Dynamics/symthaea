# D0FUS 3.2.0 pinned ITER reference run

This is an isolated external-solver integration harness. It is not production Python in the Luminous stack. The runner imports the verified D0FUS wheel or the exact upstream source tree, executes the official ITER RUN deck, and emits machine-readable evidence receipts.

## Frozen upstream identities

| Item | Pinned identity |
|---|---|
| Upstream | `IRFM/D0FUS` |
| Tag | `v3.2.0` |
| Git commit | `a682c3511dc189b2a4cdd6f4b8bc545d3cdb8924` |
| `pyproject.toml` Git blob | `cb526504e8be1b318c2e7c05195e98adb3ca7d2d` |
| ITER deck | `D0FUS_INPUTS/1_run_ITER.txt` |
| ITER deck Git blob | `d8932a85adf05e2b1b72a52d5a04a20cb25943b8` |
| PyPI wheel SHA-256 | `068a865a979367b1c74a93138bc16384e00d7d85c2dfe7e6d8b5a212d5885847` |
| PyPI sdist SHA-256 | `02a81bbead2f7bc87b7f0f981687e784acfce27dea1090c9a2931ecd44e2fd40` |

PyPI does not publish provenance for these release files. The workflow verifies their file hashes, checks the exact Git commit and deck blob, and byte-compares every packaged Python module under `D0FUS.py`, `D0FUS_BIB` and `D0FUS_EXE` against the tag checkout.

The official ITER deck is not included in the PyPI wheel; it is retrieved from the separately pinned source commit. The wheel is executed with that exact external deck.

## What the workflow does

1. Checks out the exact D0FUS source commit and verifies both pinned Git blob identities.
2. Downloads and verifies the wheel and source distribution hashes before installing the wheel.
3. Installs the runtime dependencies defined by the upstream `pyproject.toml` floors (without the optional PyVista 3-D dependency).
4. Executes the same ITER RUN deck using the installed wheel and the source checkout, with figure generation and 3-D rendering disabled.
5. Records the actual output tuple length (the release README/doc surfaces differ between 165 and 167) and requires finite, positive values for B0, energy confinement time, Q and plasma current.
6. Records full-precision gross electric, variable/fixed/total recirculation and net electric outputs, checks both independent power-balance identities, and compares the actual config-field count, output tuple length and canonicalized complete output tuple digest between wheel and source execution. Module byte differences are preserved separately, so source/wheel output parity is not mistaken for source-code identity.
7. Records the runtime Python/dependency versions, actual GlobalConfig dataclass field count, output tuple length, core observables, deck/output hashes, source/wheel identities and output report hashes.
8. Uploads the receipts and generated text reports as a workflow artifact.

The dependency floors are not a hermetic lockfile; exact resolved versions are recorded in the receipt. A subsequent environment-lock step should freeze the successful resolution before claiming bit-for-bit environment reproducibility. The 165 value in the lock is the source README's declared tuple length for reference only; the harness records the actual runtime length and requires wheel/source agreement rather than assuming that either the README or rendered documentation is correct.

## Run locally

Use Python 3.11 and a clean environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r runtime-requirements.in
```

Download the pinned D0FUS source tag and release wheel, verify the hashes in `reference_lock.json`, and run:

```sh
python run_d0fus_reference.py --mode wheel \
  --lock reference_lock.json \
  --source-root /path/to/D0FUS-v3.2.0 \
  --deck /path/to/D0FUS-v3.2.0/D0FUS_INPUTS/1_run_ITER.txt \
  --output-dir /tmp/d0fus-wheel-run \
  --receipt /tmp/d0fus-wheel-run/receipt.json

python run_d0fus_reference.py --mode source \
  --lock reference_lock.json \
  --source-root /path/to/D0FUS-v3.2.0 \
  --deck /path/to/D0FUS-v3.2.0/D0FUS_INPUTS/1_run_ITER.txt \
  --output-dir /tmp/d0fus-source-run \
  --receipt /tmp/d0fus-source-run/receipt.json

python compare_d0fus_receipts.py \
  --wheel-receipt /tmp/d0fus-wheel-run/receipt.json \
  --source-receipt /tmp/d0fus-source-run/receipt.json \
  --receipt /tmp/d0fus-parity.json
```

## Claim ceiling

A PASS establishes that the pinned D0FUS v3.2.0 wheel and the matching source tree execute the same official ITER reference deck in the recorded environment, report the actual runtime tuple length, pass the declared output sanity checks, and produce identical canonical output tuples. It does not establish independent experimental validity, validate the D0FUS physics models, prove applicability to a DEMO/commercial plant, or imply the ITER deck is a commercial power-plant design.

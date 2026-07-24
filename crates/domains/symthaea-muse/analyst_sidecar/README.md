# Muse Analyst external audio witness

The sidecar is an optional, pinned Essentia environment. It measures rendered
audio independently and emits JSON with `external-cross-check` provenance.
It is not a composer, verifier of symbolic truth, or quality model.

Run:

    nix develop 'path:.' -c uv run --python 3.14 --frozen analyze.py path/to/clip.wav

The checked-in `uv.lock` fixes the complete Python dependency graph. Historic
reports retain their tool version and artifact hash; rerunning a newer sidecar
creates another record instead of overwriting prior evidence.

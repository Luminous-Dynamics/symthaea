# HDC Dimension × Quality Frontier

The dimension frontier is a reference-only manifest for comparing already executed evidence at multiple HDC resolutions. It is deliberately not a ranking mechanism.

Each row identifies one resolution and references task-quality, performance, and resource artifacts. Structural dimension-sweep evidence is optional. The manifest itself contains no copied benchmark scores, so numeric analysis must resolve the referenced artifacts and retain their provenance.

## Why this exists

Recent HDC research reinforces that dimensionality is an experimental variable: a 2026 survey treats accuracy, efficiency, and scalability as distinct evaluation dimensions, while recent work reports substantial dimensionality reductions from deterministic structured projections. citeturn0search0turn0search4

That makes the useful question for Symthaea not “which dimension is largest?” but “what quality/cost behavior is actually evidenced for this task and representation?”

## Identity boundary

The manifest identity covers task/scenario/protocol/model identity, ordered resolutions, and referenced artifact digests. Execution details remain in the referenced evidence artifacts.

Changing any resolution or referenced artifact changes the manifest digest. Duplicate or descending dimensions fail closed.

## Intended analysis

Downstream research can derive:

- quality-versus-dimension curves,
- memory/throughput-versus-dimension curves,
- uncertainty intervals,
- task-specific saturation points,
- and Pareto/frontier views.

Those are analysis outputs, not properties asserted by this manifest.

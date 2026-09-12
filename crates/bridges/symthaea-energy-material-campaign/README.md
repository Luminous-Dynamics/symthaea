# symthaea-energy-material-campaign

Freezes a Tier-1 energy-material screening campaign before result assembly.

A seven-dimensional evidence system can still be biased after the fact by changing, for example:

- mining vs refining concentration;
- maximum-element vs mass-weighted supply aggregation;
- hazard scoring policy;
- recovery process/feedstock;
- manufacturability metric;
- model version or method parameters;
- source dataset/query.

This crate gives those choices one deterministic campaign-manifest identity before the complete dossier is admitted.

## Frozen campaign identity

A manifest binds:

- exact immutable candidate version;
- complete screening policy + policy SHA-256;
- exactly seven Tier-1 evidence lanes;
- adapter name/version per lane;
- expected model name/version;
- method-parameter map;
- source commitment;
- required evidence kinds;
- optional external registration reference;
- review notes.

Lane order and required-evidence-kind order are canonicalized for identity.

## Source commitments

Three source modes are explicit:

### `PinnedEvidenceDigest`

The exact evidence/source artifact is already known when the campaign is frozen. Admission requires that SHA-256 to appear in the prediction's evidence/model provenance.

### `InternalLineage`

An internal model/data lineage digest is frozen. Admission likewise requires that digest in prediction provenance.

### `ProspectiveAcquisition`

The source content does not yet exist locally or is intentionally acquired later. The manifest freezes source name/version/URI plus an acquisition/query SHA-256.

Result admission then requires a separate acquisition declaration containing:

- the same query SHA-256;
- acquired artifact SHA-256;
- exact source-receipt SHA-256;
- reviewer + review note.

The acquired artifact digest must appear in the resulting prediction provenance.

A projection or later data release therefore cannot silently replace the planned source/query.

## Complete-result admission

A Tier-1 result is admitted only when:

- candidate-version digest matches the campaign;
- screening-policy digest matches;
- the dossier is evidence-complete across all seven dimensions;
- each lane uses the planned metric/unit;
- prediction fidelity clears the frozen policy minimum;
- expected model name/version matches;
- required evidence kinds are present;
- pinned/internal source commitments are present in prediction provenance;
- prospective acquisitions have matching declarations and artifact provenance.

**Hard-feasibility success is not required for admission.** A scientifically complete negative/infeasible result must remain admissible rather than disappearing from the campaign record.

## Important execution-adherence limitation

The manifest cryptographically freezes the declared method parameters, but the current generic `Prediction` schema does not yet carry a native campaign-lane digest.

Therefore v0 does **not** cryptographically prove that every low-level method parameter was obeyed during receipt generation. Model/source checks narrow the gap, and an external registration reference can establish that the plan existed before results, but method adherence still requires review of the underlying source receipt.

Future Tier-1 adapter receipts should embed the exact campaign-lane SHA-256. At that point admission can verify method adherence directly and this compatibility limitation can be retired.

## Authority boundary

A campaign admission receipt means only that a complete candidate-bound dossier conforms to the frozen evidence plan at the checks described above. It is not proof of novelty, experimental truth, safety certification, synthesis success, manufacturability, investment merit, or deployment approval.

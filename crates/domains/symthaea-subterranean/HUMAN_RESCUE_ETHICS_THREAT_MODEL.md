# Human Rescue Ethics Threat Model

## Assets protected

- A person's case-specific consent, refusal, and withdrawal.
- Physical safety and protected return reserve of the rescuer.
- Rescue subject identity binding and case continuity.
- Transparent, non-discriminatory triage evidence.
- Emergency-intervention authority and reviewer independence.
- Operational checkpoint and evidence continuity.

## Failure and adversary classes

### Distress interpreted as consent

A distress message may be urgent but does not itself authorize rescue motion. Explicit case-specific consent, an accepted handoff without a fresher refusal, or a valid emergency authorization is required.

### Replayed consent

An older consent statement is replayed after refusal or withdrawal. Per-case epoch and sequence state rejects non-monotonic records.

### Forged or unauthenticated consent

A syntactically valid statement lacks an externally authenticated identity assertion. The statement fails validation and receives no authority.

### Coerced or stale consent

The crate can detect expiry and later withdrawal but cannot determine coercion, duress, legal capacity, or whether upstream identity authentication was compromised. Those remain external review obligations.

### Contradictory subject identity

Trusted reporters bind one rescue case to different opaque subjects or identity digests. The case enters reconciliation or hold; the system does not choose the majority claim.

### Contradictory care urgency

Trusted reporters materially disagree on coarse urgency. The system exposes the conflict and removes rescue selection until reconciled.

### Fabricated communication loss

One reporter claims the subject cannot communicate. Emergency intervention requires corroboration by at least two distinct trusted reporters plus two distinct hardware-backed approval roles.

### Single-operator emergency override

One person attempts to authorize emergency intervention alone or fills both required roles. Duplicate identity cannot satisfy the split-role ceremony.

### Social-value triage

A rescue policy attempts to rank by occupation, rank, wealth, nationality, mission value, payload value, or a protected attribute. Those fields do not exist in the triage candidate schema.

### Urgency overriding refusal

A high-severity or short-survival-window case has refused or withdrawn. Refusal and withdrawal remain ineligible regardless of urgency score.

### Rescue authority consuming survival reserves

A valid rescue case would violate local return reserve or physical safety. Existing return, hazard, resource, maintenance, and final-command authorities remain dominant.

### Checkpoint amnesia

Restart loses a withdrawal, identity conflict, or emergency-authorization expiry and resumes rescue. Checkpoint schema 16 preserves and validates the entire rescue-ethics state.

## Containment responses

- Invalid, replayed, expired, or unauthenticated consent: reject.
- Missing consent: await consent and stop rescue motion.
- Conflicting subject or care claims: reconcile; hold active rescue.
- Valid consent and eligible subject: rescue-only authority, no productive excavation.
- Withdrawal during active rescue: Red review hold with emergency recovery actuators preserved.
- Byzantine team inconsistency: retain the stricter Byzantine return or hold.

## Residual risks

The model cannot eliminate:

- compromised upstream identity systems;
- coercion or incapacity undetectable from authenticated messages;
- incorrect medical or survival-window claims;
- systematic bias in sensors, routing, or care evidence;
- inaccessible consent interfaces;
- malicious collusion among independent roles;
- unsafe extraction hardware;
- legal or cultural invalidity of the emergency process.

These risks require trained human oversight, accessible interfaces, external authentication, independent medical and legal review, HIL qualification, and real-world exercises.

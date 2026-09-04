# Qualification Status

Candidate only. Do not describe this crate as compiler-qualified or promotion-ready until the exact-head `IoT Actuation Guard Semantic Persistence` workflow completes successfully.

The qualification claim is limited to:

- exact `PersistedAdmissionReservation` + `VerifiedAdmissionDeviceReality` lineage;
- independently retained current `DeviceSemanticHead`;
- locked re-read before freshness/policy evaluation;
- crash-durable successor checkpoint write and canonical read-back;
- authenticated-but-unsafe reality failing before semantic head advancement; and
- no controller, final-permit, JIT, HAL, network, process-exec, or unsafe surface.

Ordinary disk persistence is not claimed to be TPM/NVRAM anti-rollback protection. The returned new semantic head must be retained independently by the deployment before the next operation.

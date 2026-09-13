# symthaea-assurance-tpm2-platform-qualification

Qualification layer for the read-only TPM2 monotonic-counter adapter.

This crate does not provision or mutate TPM state. It queries `tpm2_getcap properties-fixed` through the same shell-free executor boundary, parses only reviewed named `raw:` properties, and requires exact family/spec-revision/manufacturer/firmware values. It deliberately avoids treating the complete human-readable output as trusted YAML.

A qualification also requires independently reviewed runtime-closure evidence and an exact adapter-artifact digest. The platform query must be bracketed by two valid observations from the same reviewed TPM counter policy; the later counter may remain equal or increase, but it may not regress.

The resulting report is content-addressed evidence. It is not a TPM quote, endorsement-certificate validation, measured-boot proof, firmware-correctness proof, or physical-authority grant. Real-hardware qualification requires executing this logic against reviewed TPM/device/tool/closure subjects and independently retaining those artifacts.

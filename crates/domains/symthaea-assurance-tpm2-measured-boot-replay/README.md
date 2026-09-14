# symthaea-assurance-tpm2-measured-boot-replay

Measured-boot event-log replay assurance for the TPM trust-root stack.

This crate consumes a fresh TPM quote-possession record and proves only that an exact measured-boot log bundle replays to the exact PCR values covered by that quote. A reviewed policy can require the UEFI Final Events Table so measurements occurring after the first firmware event-log handoff are not silently omitted.

The replay verifier, verifier identity, PCR selection, possession record, quote artifacts, event-log bytes, optional final-events bytes, and replayed PCR-value digest are all content-bound.

A passing replay record means **the supplied ordered measurement evidence is consistent with the quoted PCR state**. It does not mean the measured firmware/software/configuration is approved, safe, benign, vendor-authentic, or equal to a Reference Integrity Manifest. Reference-state/RIM evaluation is a separate assurance layer.

This crate performs no TPM mutation and grants no physical authority.

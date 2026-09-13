# symthaea-assurance-tpm2-attestation-possession

Fresh TPM 2.0 quote-possession assurance for the Symthaea trust-root stack.

This crate binds a reviewed platform qualification to a fresh verifier challenge, an exact reviewed Attestation Key (AK), exact PCR selection, content-addressed quote artifacts, and an independent quote-verification receipt. Qualification requires the quote signature, caller nonce / qualifying data, PCR digest, PCR selection, TPM-generated magic, and quote attestation type to have been independently verified for the exact bound artifacts.

The challenge uses a 32-byte nonce and has a bounded validity interval. Accepted possession records are one-shot: challenge IDs, nonces, quote artifacts, and verification receipt IDs cannot be replayed into a second accepted record.

This establishes **fresh possession of the reviewed AK over the selected PCR quote evidence**. It does **not** establish that the selected PCR values are good, that an event log replays to an approved reference state, that the AK chains to a manufacturer-trusted EK, that an EK certificate is valid, or that the TPM implementation/firmware is correct. Those are separate assurance obligations.

The crate performs no TPM mutation and grants no physical authority.

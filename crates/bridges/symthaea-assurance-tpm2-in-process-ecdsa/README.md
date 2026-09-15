# symthaea-assurance-tpm2-in-process-ecdsa

Eliminates the external `tpm2_checkquote` child process for the deliberately narrow TPM2 ECDSA P-256 + SHA-256 profile while reusing the existing raw-quote semantic parser and opaque quote capability.

The bridge accepts only an exact reviewed parent quote policy, an exact reviewed running-host executable identity, X.509 SubjectPublicKeyInfo for `id-ecPublicKey` + `prime256v1`, and a TSS `TPMT_SIGNATURE` carrying ECDSA/SHA-256. Signature verification runs in-process through the workspace-pinned `aws-lc-rs` backend; the parent verifier still independently checks TPM magic/type, challenge nonce, qualified signer, PCR selection, and PCR composite.

The running host is observed through `/proc/self/exe` and must match the reviewed canonical direct Nix-store pathname, inode metadata, ownership/permissions, and exact BLAKE3. This removes a second process launch, temp-file protocol, and external verifier loader from the quote-verification path. It does not prove that mapped executable pages equal the backing file, atomically pin shared-library dependencies, exclude privileged in-place mutation, establish trusted time, or grant physical authority.

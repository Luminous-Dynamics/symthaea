# OpenSSL ML-DSA-65 interoperability vector v1

This vector is public verification material only. No private key is committed.

- Generator/verifier: OpenSSL 3.5.5, default provider
- Algorithm: ML-DSA-65 / Pure ML-DSA
- Context: empty (OpenSSL default)
- Message: 32 bytes of `0xA5`
- Signing mode: deterministic test mode (`deterministic:1`)
- Signature length: 3309 bytes
- Raw public-key length: 1952 bytes
- Message SHA-256: `fc8b64001c5fdd0f2f40fb67dae4a865a2c5bd17836676d6d5b58b7917e33717`
- Signature SHA-256: `a274d68afe37fdde6cd330a04fc91cef86756ea61c6f6a46c910d4999280c5e3`
- Raw public-key SHA-256: `a0f077786cbea674bdf68eef84713d19822f1a61c0b82be7c0ec0e2292934afa`

Generation/verification commands:

```sh
printf '<32 bytes of A5>' > message.bin
openssl genpkey -algorithm ML-DSA-65 -out key.pem
openssl pkeyutl -sign -in message.bin -inkey key.pem -out signature.bin -pkeyopt deterministic:1
openssl pkeyutl -verify -in message.bin -inkey key.pem -sigfile signature.bin
openssl pkey -in key.pem -pubout -outform DER -out public.der
```

The SubjectPublicKeyInfo DER contained an ML-DSA-65 BIT STRING with one unused-bits byte followed by the exact 1952-byte FIPS-204 public-key encoding committed in `openssl-3.5.5-mldsa65-public.hex`.

Purpose: this is a neutral third implementation vector. Symthaea/fips204 must verify it in this crate. A corresponding Xenia/RustCrypto regression should verify the identical public key, message, and signature before the cross-repository receipt contract is called qualified.

# VART-WORLD-CREATIVE-001 — RANDOM_VALID v1

Status: frozen baseline algorithm candidate for pilot qualification. This document does not authorize confirmatory execution.

## Purpose

`RANDOM_VALID` must isolate selection/judgment value without introducing a language-, library-, platform-, or RNG-version dependency. It therefore uses a deterministic SHA-256 counter construction rather than `rand`, OS entropy, or a runtime PRNG.

The chooser receives only:

- the preregistered unsigned 64-bit `seed`;
- UTF-8 `paired_block_id`;
- the exact raw-byte SHA-256 of the physically admitted candidate-set artifact;
- the number of physically admitted candidates.

It must not receive candidate outcome values, FULL/HEURISTIC choices, later observations, or human labels.

## Algorithm: `sha256-counter-v1`

Let:

- `D = b"SYMTHAEA-VART-RANDOM-VALID-v1\x00"`;
- `S = seed` encoded as unsigned 8-byte big-endian;
- `P = paired_block_id` encoded as UTF-8;
- `L = len(P)` encoded as unsigned 4-byte big-endian;
- `C = candidate_set_sha256` decoded from 64 lowercase hexadecimal characters to 32 bytes;
- `N = admissible_candidate_count`, with `N > 0`;
- `K = floor(2^256 / N) * N`.

For counter `i = 0, 1, 2, ...`:

1. encode `i` as unsigned 8-byte big-endian `I`;
2. compute `H = SHA256(D || S || L || P || C || I)`;
3. interpret `H` as an unsigned 256-bit big-endian integer `X`;
4. if `X < K`, select index `X mod N` and stop;
5. otherwise increment `i` and repeat.

The rejection step removes modulo bias. Candidate order is the frozen order of physically admitted entries in the candidate-set artifact; implementations must not re-sort it.

## Random draw receipt

Each RANDOM_VALID trial exports a receipt containing at minimum:

- `algorithm = "sha256-counter-v1"`;
- `seed`;
- `paired_block_id`;
- `candidate_set_sha256`;
- `admissible_candidate_count`;
- `counter` used for the accepted digest;
- `accepted_digest_sha256` as lowercase hex;
- `selected_index`.

The independent verifier recomputes this receipt without importing World Forge or runtime decision code.

## Scientific boundary

This mechanism is deterministic pseudorandomization, not a source of physical randomness. Its purpose is reproducible baseline assignment. Changing the domain separator, byte encoding, candidate ordering, seed interpretation, rejection rule, or hash function creates a new baseline-policy digest and therefore a new preregistration lineage for confirmatory use.

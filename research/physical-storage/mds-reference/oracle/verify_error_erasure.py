#!/usr/bin/env python3
"""Independent, standard-library-only oracle for the custom MDS(6,4) reference.

This is a research validation tool, not production decoder code. It deliberately
reimplements GF(256) operations and candidate enumeration separately from Rust.
It emits deterministic JSON so the results can be retained with the experiment.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json

N = 6
K = 4
WIDTH = 8
DISTANCE = 3
REDUCTION = 0x1D  # x^8 reduction for polynomial 0x11d

GENERATOR = (
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 1, 1),
    (1, 2, 3, 4),
)
CONTROL_TILES = {0, 15, 16, 31}
SPARE_TILES = {32, 47}


def gf_mul(left: int, right: int) -> int:
    product = 0
    for _ in range(8):
        if right & 1:
            product ^= left
        carry = left & 0x80
        left = (left << 1) & 0xFF
        if carry:
            left ^= REDUCTION
        right >>= 1
    return product


def gf_pow(value: int, exponent: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = gf_mul(result, value)
        value = gf_mul(value, value)
        exponent >>= 1
    return result


def gf_inv(value: int) -> int:
    if value == 0:
        raise ZeroDivisionError("zero has no multiplicative inverse in GF(256)")
    return gf_pow(value, 254)


def invert(matrix: list[list[int]]) -> list[list[int]]:
    size = len(matrix)
    augmented = [
        row[:] + [int(i == j) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]),
            None,
        )
        if pivot is None:
            raise ValueError("singular matrix")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        scale = gf_inv(augmented[column][column])
        augmented[column] = [gf_mul(item, scale) for item in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    item ^ gf_mul(factor, pivot_item)
                    for item, pivot_item in zip(augmented[row], augmented[column])
                ]
    return [row[size:] for row in augmented]


def encode(data: list[list[int]]) -> list[list[int]]:
    encoded = [[0] * WIDTH for _ in range(N)]
    for shard, generator_row in enumerate(GENERATOR):
        for source, coefficient in enumerate(generator_row):
            for offset in range(WIDTH):
                encoded[shard][offset] ^= gf_mul(coefficient, data[source][offset])
    return encoded


def recover_lane(received: list[int | None]) -> tuple[list[int], int] | None:
    present = [index for index, value in enumerate(received) if value is not None]
    erasures = N - len(present)
    if len(present) < K:
        return None

    candidates: dict[tuple[int, ...], int] = {}
    for selected in combinations(present, K):
        inverse = invert([list(GENERATOR[index]) for index in selected])
        candidate_data = []
        for source in range(K):
            value = 0
            for equation, shard in enumerate(selected):
                observed = received[shard]
                assert observed is not None
                value ^= gf_mul(inverse[source][equation], observed)
            candidate_data.append(value)

        codeword = []
        for generator_row in GENERATOR:
            value = 0
            for source, coefficient in enumerate(generator_row):
                value ^= gf_mul(coefficient, candidate_data[source])
            codeword.append(value)

        errors = sum(received[index] != codeword[index] for index in present)
        if 2 * errors + erasures < DISTANCE:
            candidates[tuple(candidate_data)] = errors

    if len(candidates) != 1:
        return None
    data, errors = next(iter(candidates.items()))
    return encode([[value] * WIDTH for value in data])[:0] if False else (list(data), errors)


def recover_stripe(
    received: list[list[int] | None],
) -> tuple[list[list[int]], int] | None:
    recovered_data = [[0] * WIDTH for _ in range(K)]
    corrected = 0
    for offset in range(WIDTH):
        lane = [None if shard is None else shard[offset] for shard in received]
        result = recover_lane(lane)
        if result is None:
            return None
        data_symbols, errors = result
        for source, value in enumerate(data_symbols):
            recovered_data[source][offset] = value
        corrected += errors
    return encode(recovered_data), corrected


def physical_tile_map() -> tuple[list[int], list[list[int]]]:
    available = [
        tile for tile in range(48)
        if tile not in CONTROL_TILES and tile not in SPARE_TILES
    ]
    stripes = [
        [available[stripe + 7 * shard] for shard in range(N)]
        for stripe in range(7)
    ]
    return available, stripes


def full_payload_decode(image: list[list[int] | None]) -> tuple[bytes | None, int]:
    _, stripe_tiles = physical_tile_map()
    payload = bytearray()
    corrections = 0
    for tile_indices in stripe_tiles:
        received = [image[tile] for tile in tile_indices]
        result = recover_stripe(received)
        if result is None:
            return None, corrections
        codeword, corrected = result
        corrections += corrected
        for shard in range(K):
            payload.extend(codeword[shard])
    return bytes(payload), corrections


def main() -> None:
    # Check every required 4-row minor is invertible (15 MDS minors).
    minor_count = 0
    for selected in combinations(range(N), K):
        invert([list(GENERATOR[index]) for index in selected])
        minor_count += 1

    data = [
        [(source * 41 + offset * 13 + 7) & 0xFF for offset in range(WIDTH)]
        for source in range(K)
    ]
    encoded = encode(data)

    single_symbol_cases = 0
    for shard in range(N):
        for offset in range(WIDTH):
            damaged = [row[:] for row in encoded]
            damaged[shard][offset] ^= 0xA7
            recovered = recover_stripe(damaged)
            assert recovered is not None and recovered[0] == encoded
            assert recovered[1] == 1
            single_symbol_cases += 1

    two_erasure_cases = 0
    for first, second in combinations(range(N), 2):
        damaged = [
            None if index in (first, second) else row[:]
            for index, row in enumerate(encoded)
        ]
        recovered = recover_stripe(damaged)
        assert recovered is not None and recovered[0] == encoded
        assert recovered[1] == 0
        two_erasure_cases += 1

    full_tile = [row[:] for row in encoded]
    for offset in range(WIDTH):
        full_tile[3][offset] ^= offset + 1
    full_tile_result = recover_stripe(full_tile)
    assert full_tile_result is not None and full_tile_result[0] == encoded
    assert full_tile_result[1] == WIDTH

    mixed = [None if index == 0 else row[:] for index, row in enumerate(encoded)]
    mixed[1][2] ^= 0x5A
    assert recover_stripe(mixed) is None

    # End-to-end 4x4x3 payload test with the repository's spatial tile map.
    source_payload = bytes(range(224))
    image: list[list[int] | None] = [None] * 48
    _, stripe_tiles = physical_tile_map()
    for stripe, tile_indices in enumerate(stripe_tiles):
        stripe_data = [
            list(source_payload[(stripe * K + shard) * WIDTH:
                                 (stripe * K + shard + 1) * WIDTH])
            for shard in range(K)
        ]
        stripe_codeword = encode(stripe_data)
        for shard, tile in enumerate(tile_indices):
            image[tile] = stripe_codeword[shard]

    first = stripe_tiles[0][0]
    second = stripe_tiles[0][1]
    assert image[first] is not None and image[second] is not None
    image[first][2] ^= 1
    image[second][2] ^= 1
    reconstructed, correction_count = full_payload_decode(image)
    assert reconstructed is not None
    digest_matches = sha256(reconstructed).digest() == sha256(source_payload).digest()
    assert not digest_matches
    assert correction_count == 1

    available, stripes = physical_tile_map()
    output = {
        "oracle": "independent_python_standard_library_gf256_implementation",
        "evidence_class": "independent_mathematical_source_oracle",
        "generator_four_row_minors_nonsingular": minor_count,
        "single_symbol_error_cases_corrected": single_symbol_cases,
        "two_known_erasure_pairs_recovered": two_erasure_cases,
        "full_tile_substitution_corrected_symbols": full_tile_result[1],
        "one_unknown_error_plus_one_erasure_rejected": True,
        "end_to_end_two_error_miscorrection": {
            "decoder_returned_candidate": reconstructed is not None,
            "corrected_symbol_count": correction_count,
            "payload_digest_matches": digest_matches,
            "archive_acceptance": False,
        },
        "layout_partition": {
            "physical_tiles": 48,
            "active_shards": len(available),
            "stripes": len(stripes),
            "unique_active_tiles": len(set(tile for row in stripes for tile in row)),
            "control_tiles": sorted(CONTROL_TILES),
            "spare_tiles": sorted(SPARE_TILES),
        },
        "rust_toolchain_execution": "not_performed",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

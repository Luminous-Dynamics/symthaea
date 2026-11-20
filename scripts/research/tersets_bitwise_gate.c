#include "tersets.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static double f64_from_bits(uint64_t bits) {
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t f64_to_bits(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int exact_round_trip(
    uint8_t method,
    const double *input,
    size_t input_count,
    size_t *mismatch_index,
    uint64_t *expected_bits,
    uint64_t *actual_bits
) {
    struct UncompressedValues source = {
        .data = input,
        .len = input_count,
    };
    struct CompressedValues compressed = {0};
    struct UncompressedValues decoded = {0};

    const int32_t compress_status = compress(source, &compressed, method, "{}");
    if (compress_status != 0) {
        return 1000 + compress_status;
    }

    const int32_t decompress_status = decompress(compressed, &decoded);
    if (decompress_status != 0) {
        freeCompressedValues(&compressed);
        return 2000 + decompress_status;
    }

    int result = 0;
    if (decoded.len != input_count) {
        result = 3001;
    } else {
        for (size_t i = 0; i < input_count; ++i) {
            const uint64_t expected = f64_to_bits(input[i]);
            const uint64_t actual = f64_to_bits(decoded.data[i]);
            if (expected != actual) {
                *mismatch_index = i;
                *expected_bits = expected;
                *actual_bits = actual;
                result = 3002;
                break;
            }
        }
    }

    freeUncompressedValues(&decoded);
    freeCompressedValues(&compressed);
    return result;
}

static int run_required(
    const char *method_name,
    uint8_t method,
    const char *fixture_name,
    const double *input,
    size_t count
) {
    size_t mismatch_index = 0;
    uint64_t expected = 0;
    uint64_t actual = 0;
    const int status = exact_round_trip(
        method, input, count, &mismatch_index, &expected, &actual
    );
    if (status == 0) {
        printf("PASS method=%s fixture=%s values=%zu bitwise=exact\n",
               method_name, fixture_name, count);
        return 0;
    }
    if (status == 3002) {
        fprintf(stderr,
                "FAIL method=%s fixture=%s index=%zu expected=0x%016" PRIx64
                " actual=0x%016" PRIx64 "\n",
                method_name, fixture_name, mismatch_index, expected, actual);
    } else {
        fprintf(stderr, "FAIL method=%s fixture=%s status=%d\n",
                method_name, fixture_name, status);
    }
    return 1;
}

static void run_conditional(
    const char *method_name,
    uint8_t method,
    const char *fixture_name,
    const double *input,
    size_t count
) {
    size_t mismatch_index = 0;
    uint64_t expected = 0;
    uint64_t actual = 0;
    const int status = exact_round_trip(
        method, input, count, &mismatch_index, &expected, &actual
    );
    if (status == 0) {
        printf("CONDITIONAL_PASS method=%s fixture=%s note=fixture_only_not_whitelisted\n",
               method_name, fixture_name);
    } else if (status == 3002) {
        printf("CONDITIONAL_BIT_MISMATCH method=%s fixture=%s index=%zu expected=0x%016"
               PRIx64 " actual=0x%016" PRIx64 "\n",
               method_name, fixture_name, mismatch_index, expected, actual);
    } else {
        printf("CONDITIONAL_UNSUPPORTED method=%s fixture=%s status=%d\n",
               method_name, fixture_name, status);
    }
}

int main(void) {
    static const uint64_t edge_patterns[] = {
        UINT64_C(0x0000000000000000), /* +0 */
        UINT64_C(0x8000000000000000), /* -0 */
        UINT64_C(0x7ff0000000000000), /* +inf */
        UINT64_C(0xfff0000000000000), /* -inf */
        UINT64_C(0x0000000000000001), /* smallest subnormal */
        UINT64_C(0x0010000000000000), /* smallest normal */
        UINT64_C(0x7ff8000000000001), /* quiet NaN payload 1 */
        UINT64_C(0x7ff8000000000002), /* quiet NaN payload 2 */
        UINT64_C(0xfff8123456789abc), /* negative NaN payload */
        UINT64_C(0x3ff0000000000000), /* 1 */
        UINT64_C(0xbff0000000000000), /* -1 */
        UINT64_C(0x400c000000000000), /* 3.5 */
        UINT64_C(0x7fefffffffffffff), /* max finite */
    };
    double edge_values[sizeof(edge_patterns) / sizeof(edge_patterns[0])];
    for (size_t i = 0; i < sizeof(edge_patterns) / sizeof(edge_patterns[0]); ++i) {
        edge_values[i] = f64_from_bits(edge_patterns[i]);
    }

    static const uint64_t zero_switch_bits[] = {
        UINT64_C(0x0000000000000000),
        UINT64_C(0x0000000000000000),
        UINT64_C(0x8000000000000000),
        UINT64_C(0x8000000000000000),
        UINT64_C(0x0000000000000000),
        UINT64_C(0x8000000000000000),
        UINT64_C(0x3ff0000000000000),
        UINT64_C(0xbff0000000000000),
    };
    double zero_switches[sizeof(zero_switch_bits) / sizeof(zero_switch_bits[0])];
    for (size_t i = 0; i < sizeof(zero_switch_bits) / sizeof(zero_switch_bits[0]); ++i) {
        zero_switches[i] = f64_from_bits(zero_switch_bits[i]);
    }

    static const uint64_t finite_domain_bits[] = {
        UINT64_C(0x3ff4000000000000), /* 1.25 */
        UINT64_C(0x4004000000000000), /* 2.5 */
        UINT64_C(0x400e000000000000), /* 3.75 */
        UINT64_C(0x4014000000000000), /* 5 */
        UINT64_C(0xc014000000000000), /* -5 */
        UINT64_C(0x3fc0000000000000), /* .125 */
        UINT64_C(0x3fd0000000000000), /* .25 */
        UINT64_C(0x3fe0000000000000), /* .5 */
    };
    double finite_domain[sizeof(finite_domain_bits) / sizeof(finite_domain_bits[0])];
    for (size_t i = 0; i < sizeof(finite_domain_bits) / sizeof(finite_domain_bits[0]); ++i) {
        finite_domain[i] = f64_from_bits(finite_domain_bits[i]);
    }

    int failures = 0;
    const struct {
        const char *name;
        uint8_t method;
    } required_methods[] = {
        {"Uncompressed", Uncompressed},
        {"Chimp64", Chimp64},
        {"Chimp128", Chimp128},
        {"Elf", Elf},
    };

    for (size_t method = 0;
         method < sizeof(required_methods) / sizeof(required_methods[0]);
         ++method) {
        failures += run_required(required_methods[method].name,
                                 required_methods[method].method,
                                 "ieee754_edges",
                                 edge_values,
                                 sizeof(edge_values) / sizeof(edge_values[0]));
        failures += run_required(required_methods[method].name,
                                 required_methods[method].method,
                                 "signed_zero_switches",
                                 zero_switches,
                                 sizeof(zero_switches) / sizeof(zero_switches[0]));
    }

    run_conditional("RunLengthEncoding", RunLengthEncoding,
                    "signed_zero_switches", zero_switches,
                    sizeof(zero_switches) / sizeof(zero_switches[0]));
    run_conditional("BitPackedDeltaEncoding", BitPackedDeltaEncoding,
                    "signed_zero_switches", zero_switches,
                    sizeof(zero_switches) / sizeof(zero_switches[0]));
    run_conditional("BitPackedDeltaEncoding", BitPackedDeltaEncoding,
                    "finite_nonzero_domain", finite_domain,
                    sizeof(finite_domain) / sizeof(finite_domain[0]));

    if (failures != 0) {
        fprintf(stderr, "TerseTS strict bitwise gate failures=%d\n", failures);
        return 1;
    }
    printf("TerseTS strict bitwise gate: PASS (conditional candidates reported separately)\n");
    return 0;
}

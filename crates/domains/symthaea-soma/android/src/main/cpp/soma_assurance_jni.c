/*
 * QUAL-ANDROIDINGRESS-522: JNI adapter for the checked 521 native ingress ABI.
 *
 * This file intentionally defines new AssuranceNativeBindings symbols rather than
 * altering legacy NativeBindings behavior. Java/Kotlin signed values are rejected
 * before conversion to the unsigned Rust ABI when negative values would change
 * semantics. All identity arrays are exactly 32 bytes and output identity storage
 * is exactly 96 bytes: profile || observation || ingress receipt.
 */

#include <jni.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define SOMA_ASSURANCE_ACCEPTED 0u
#define SOMA_JNI_ARGUMENT_ERROR 1000
#define SOMA_JNI_ACCESS_ERROR   1001
#define SOMA_ID_LEN 32
#define SOMA_OUTPUT_IDS_LEN 96

/* Must match #[repr(C)] QUAL-NATIVEINGRESS-521 structures. */
typedef struct {
    uint32_t struct_size;
    uint8_t platform;
    uint8_t reserved0[3];
    uint32_t abi_version;
    uint32_t reserved1;
    uint64_t max_frame_bytes;
    uint8_t capture_profile_id[32];
    uint8_t touch_input_profile_id[32];
} SomaAssuranceIngressProfileV1;

typedef struct {
    uint32_t struct_size;
    uint32_t reserved0;
    SomaAssuranceIngressProfileV1 profile;
    const uint8_t *data;
    uint64_t data_len;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t reserved1;
    uint8_t surface_id[32];
    uint64_t surface_generation;
    uint64_t interaction_generation;
    uint64_t frame_sequence;
} SomaAssuranceFrameRequestV1;

typedef struct {
    uint32_t struct_size;
    uint32_t reserved0;
    SomaAssuranceIngressProfileV1 profile;
    uint8_t input_attester_id[32];
    uint64_t interaction_generation;
    uint64_t input_sequence;
    float x;
    float y;
    float pressure;
    uint8_t action;
    uint8_t reserved1[3];
    uint64_t timestamp_ms;
} SomaAssuranceTouchRequestV1;

typedef struct {
    uint32_t struct_size;
    uint32_t status;
    uint8_t profile_id[32];
    uint8_t frame_observation_id[32];
    uint8_t ingress_receipt_id[32];
    float surprise;
    uint32_t reserved0;
} SomaAssuranceFrameResultV1;

typedef struct {
    uint32_t struct_size;
    uint32_t status;
    uint8_t profile_id[32];
    uint8_t touch_observation_id[32];
    uint8_t ingress_receipt_id[32];
} SomaAssuranceTouchResultV1;

#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(SomaAssuranceIngressProfileV1) == 88, "521 profile ABI drift");
_Static_assert(sizeof(SomaAssuranceFrameRequestV1) == 184, "521 frame request ABI drift");
_Static_assert(sizeof(SomaAssuranceTouchRequestV1) == 168, "521 touch request ABI drift");
_Static_assert(sizeof(SomaAssuranceFrameResultV1) == 112, "521 frame result ABI drift");
_Static_assert(sizeof(SomaAssuranceTouchResultV1) == 104, "521 touch result ABI drift");
#endif

/* Rust QUAL-NATIVEINGRESS-521 entrypoints. */
uint32_t soma_assurance_inject_frame_v1(
    void *engine,
    const SomaAssuranceFrameRequestV1 *request,
    SomaAssuranceFrameResultV1 *out);
uint32_t soma_assurance_touch_event_v1(
    void *engine,
    const SomaAssuranceTouchRequestV1 *request,
    SomaAssuranceTouchResultV1 *out);

static int exact_array_len(JNIEnv *env, jbyteArray array, jsize expected) {
    return array != NULL && (*env)->GetArrayLength(env, array) == expected;
}

static int exact_float_array_len(JNIEnv *env, jfloatArray array, jsize expected) {
    return array != NULL && (*env)->GetArrayLength(env, array) == expected;
}

static int copy_id(JNIEnv *env, jbyteArray source, uint8_t destination[32]) {
    if (!exact_array_len(env, source, SOMA_ID_LEN)) {
        return 0;
    }
    (*env)->GetByteArrayRegion(env, source, 0, SOMA_ID_LEN, (jbyte *)destination);
    return !(*env)->ExceptionCheck(env);
}

static int fill_profile(
    JNIEnv *env,
    SomaAssuranceIngressProfileV1 *profile,
    jint platform,
    jint abi_version,
    jlong max_frame_bytes,
    jbyteArray capture_profile_id,
    jbyteArray touch_input_profile_id) {
    if (platform < 0 || platform > UINT8_MAX || abi_version <= 0 || max_frame_bytes <= 0) {
        return 0;
    }
    memset(profile, 0, sizeof(*profile));
    profile->struct_size = (uint32_t)sizeof(*profile);
    profile->platform = (uint8_t)platform;
    profile->abi_version = (uint32_t)abi_version;
    profile->max_frame_bytes = (uint64_t)max_frame_bytes;
    if (!copy_id(env, capture_profile_id, profile->capture_profile_id)) {
        return 0;
    }
    if (!copy_id(env, touch_input_profile_id, profile->touch_input_profile_id)) {
        return 0;
    }
    return 1;
}

static void copy_frame_ids(uint8_t output[96], const SomaAssuranceFrameResultV1 *result) {
    memcpy(output, result->profile_id, 32);
    memcpy(output + 32, result->frame_observation_id, 32);
    memcpy(output + 64, result->ingress_receipt_id, 32);
}

static void copy_touch_ids(uint8_t output[96], const SomaAssuranceTouchResultV1 *result) {
    memcpy(output, result->profile_id, 32);
    memcpy(output + 32, result->touch_observation_id, 32);
    memcpy(output + 64, result->ingress_receipt_id, 32);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_soma_AssuranceNativeBindings_injectFrameV1Native(
    JNIEnv *env,
    jclass clazz,
    jlong handle,
    jbyteArray data,
    jint width,
    jint height,
    jint channels,
    jint platform,
    jint abiVersion,
    jlong maxFrameBytes,
    jbyteArray captureProfileId,
    jbyteArray touchInputProfileId,
    jbyteArray surfaceId,
    jlong surfaceGeneration,
    jlong interactionGeneration,
    jlong frameSequence,
    jbyteArray outIds,
    jfloatArray outSurprise) {
    (void)clazz;

    if (handle == 0 || data == NULL || width <= 0 || height <= 0 || channels <= 0 ||
        surfaceGeneration <= 0 || interactionGeneration <= 0 || frameSequence < 0 ||
        !exact_array_len(env, surfaceId, SOMA_ID_LEN) ||
        !exact_array_len(env, outIds, SOMA_OUTPUT_IDS_LEN) ||
        !exact_float_array_len(env, outSurprise, 1)) {
        return SOMA_JNI_ARGUMENT_ERROR;
    }

    SomaAssuranceFrameRequestV1 request;
    memset(&request, 0, sizeof(request));
    request.struct_size = (uint32_t)sizeof(request);
    if (!fill_profile(env, &request.profile, platform, abiVersion, maxFrameBytes,
                      captureProfileId, touchInputProfileId)) {
        return (*env)->ExceptionCheck(env) ? SOMA_JNI_ACCESS_ERROR : SOMA_JNI_ARGUMENT_ERROR;
    }
    if (!copy_id(env, surfaceId, request.surface_id)) {
        return (*env)->ExceptionCheck(env) ? SOMA_JNI_ACCESS_ERROR : SOMA_JNI_ARGUMENT_ERROR;
    }

    jsize data_len = (*env)->GetArrayLength(env, data);
    jbyte *bytes = (*env)->GetByteArrayElements(env, data, NULL);
    if (bytes == NULL) {
        return SOMA_JNI_ACCESS_ERROR;
    }

    request.data = (const uint8_t *)bytes;
    request.data_len = (uint64_t)data_len;
    request.width = (uint32_t)width;
    request.height = (uint32_t)height;
    request.channels = (uint32_t)channels;
    request.surface_generation = (uint64_t)surfaceGeneration;
    request.interaction_generation = (uint64_t)interactionGeneration;
    request.frame_sequence = (uint64_t)frameSequence;

    SomaAssuranceFrameResultV1 result;
    memset(&result, 0, sizeof(result));
    uint32_t status = soma_assurance_inject_frame_v1(
        (void *)(intptr_t)handle, &request, &result);
    (*env)->ReleaseByteArrayElements(env, data, bytes, JNI_ABORT);

    if (status == SOMA_ASSURANCE_ACCEPTED) {
        uint8_t ids[SOMA_OUTPUT_IDS_LEN];
        copy_frame_ids(ids, &result);
        (*env)->SetByteArrayRegion(env, outIds, 0, SOMA_OUTPUT_IDS_LEN, (const jbyte *)ids);
        jfloat surprise = result.surprise;
        (*env)->SetFloatArrayRegion(env, outSurprise, 0, 1, &surprise);
        if ((*env)->ExceptionCheck(env)) {
            return SOMA_JNI_ACCESS_ERROR;
        }
    }
    return (jint)status;
}

JNIEXPORT jint JNICALL
Java_io_symthaea_soma_AssuranceNativeBindings_touchEventV1Native(
    JNIEnv *env,
    jclass clazz,
    jlong handle,
    jint platform,
    jint abiVersion,
    jlong maxFrameBytes,
    jbyteArray captureProfileId,
    jbyteArray touchInputProfileId,
    jbyteArray inputAttesterId,
    jlong interactionGeneration,
    jlong inputSequence,
    jfloat x,
    jfloat y,
    jint action,
    jfloat pressure,
    jlong timestampMs,
    jbyteArray outIds) {
    (void)clazz;

    if (handle == 0 || interactionGeneration <= 0 || inputSequence <= 0 ||
        timestampMs < 0 || action < 0 || action > UINT8_MAX ||
        !exact_array_len(env, inputAttesterId, SOMA_ID_LEN) ||
        !exact_array_len(env, outIds, SOMA_OUTPUT_IDS_LEN)) {
        return SOMA_JNI_ARGUMENT_ERROR;
    }

    SomaAssuranceTouchRequestV1 request;
    memset(&request, 0, sizeof(request));
    request.struct_size = (uint32_t)sizeof(request);
    if (!fill_profile(env, &request.profile, platform, abiVersion, maxFrameBytes,
                      captureProfileId, touchInputProfileId)) {
        return (*env)->ExceptionCheck(env) ? SOMA_JNI_ACCESS_ERROR : SOMA_JNI_ARGUMENT_ERROR;
    }
    if (!copy_id(env, inputAttesterId, request.input_attester_id)) {
        return (*env)->ExceptionCheck(env) ? SOMA_JNI_ACCESS_ERROR : SOMA_JNI_ARGUMENT_ERROR;
    }

    request.interaction_generation = (uint64_t)interactionGeneration;
    request.input_sequence = (uint64_t)inputSequence;
    request.x = x;
    request.y = y;
    request.pressure = pressure;
    request.action = (uint8_t)action;
    request.timestamp_ms = (uint64_t)timestampMs;

    SomaAssuranceTouchResultV1 result;
    memset(&result, 0, sizeof(result));
    uint32_t status = soma_assurance_touch_event_v1(
        (void *)(intptr_t)handle, &request, &result);

    if (status == SOMA_ASSURANCE_ACCEPTED) {
        uint8_t ids[SOMA_OUTPUT_IDS_LEN];
        copy_touch_ids(ids, &result);
        (*env)->SetByteArrayRegion(env, outIds, 0, SOMA_OUTPUT_IDS_LEN, (const jbyte *)ids);
        if ((*env)->ExceptionCheck(env)) {
            return SOMA_JNI_ACCESS_ERROR;
        }
    }
    return (jint)status;
}

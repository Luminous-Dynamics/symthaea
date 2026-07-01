/**
 * JNI C glue layer for symthaea-soma Android bindings.
 *
 * Maps Kotlin/Java calls from io.symthaea.soma.NativeBindings to the
 * Rust extern "C" FFI defined in native_ffi.rs. Engine handles are passed
 * as jlong (opaque pointer cast).
 *
 * Memory contract:
 *   - Rust strings returned by soma_*() must be freed with soma_string_free().
 *   - JNI strings from GetStringUTFChars must be released with ReleaseStringUTFChars.
 *   - The caller (Kotlin) serialises access to the engine pointer.
 */

#include <jni.h>
#include <stdint.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Forward declarations of Rust extern "C" functions (native_ffi.rs)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Lifecycle */
void *soma_engine_new(void);
void *soma_engine_new_mobile(void);
void *soma_engine_new_with_config(const char *config_json);
void  soma_engine_free(void *engine);

/* Core cycle */
float soma_engine_cycle(void *engine, const char *input);
char *soma_engine_cycle_json(void *engine, const char *input);

/* State inspection */
float    soma_engine_consciousness_level(const void *engine);
uint64_t soma_engine_cycle_count(const void *engine);
float    soma_engine_substrate_feasibility(const void *engine);
float    soma_engine_harmony_alignment(const void *engine);
char    *soma_engine_consciousness_report(const void *engine);
char    *soma_engine_neuromod_json(const void *engine);

/* Platform integration */
void soma_engine_set_thermal_level(void *engine, uint8_t level);
void soma_engine_set_battery_state(void *engine, uint8_t charge_percent, uint8_t is_charging);
void soma_engine_set_night_mode(void *engine, uint8_t is_night);

/* Dream */
uint8_t soma_engine_dream_cycle(void *engine);

/* Metabolism */
void    soma_engine_wake_signal(void *engine, uint8_t signal);
uint8_t soma_engine_wake_state(const void *engine);

/* Sensors */
void    soma_engine_set_sensors(void *engine, float accel, float light,
                                 uint8_t proximity_near, float barometer,
                                 float gps_novelty);
uint8_t soma_engine_motion_state(const void *engine);
uint8_t soma_engine_privacy_mode(const void *engine);

/* Expanded senses */
void soma_engine_set_gyroscope(void *engine, float rotation_rate);
void soma_engine_set_step_delta(void *engine, uint32_t steps);
void soma_engine_set_ambient_db(void *engine, float db);
void soma_engine_set_social_pressure(void *engine, uint32_t notification_count);
void soma_engine_set_media_state(void *engine, uint8_t state);

/* Compass */
char *soma_engine_compass_json(const void *engine);

/* Sharing */
void soma_engine_set_sharing_config(void *engine, const char *json);

/* Haptic */
char    *soma_haptic_drain(void *engine);
uint32_t soma_haptic_pending(const void *engine);
void     soma_haptic_set_enabled(void *engine, uint8_t enabled);

/* Dream journal */
char    *soma_dream_journal_latest(const void *engine);
char    *soma_dream_journal_all(const void *engine);
uint32_t soma_dream_journal_count(const void *engine);
void     soma_engine_dream_consolidate(void *engine);

/* Holon bridge */
char *soma_engine_holon_drain_outbound(void *engine);
void  soma_engine_holon_receive(void *engine, const char *json);
void  soma_engine_holon_set_connected(void *engine, uint8_t connected);

/* BLE mesh */
uint8_t  soma_ble_receive_peer(void *engine, uint64_t peer_id,
                                const uint8_t *cv_data, uint32_t len);
uint32_t soma_ble_advertise_payload(void *engine, uint8_t *out_buf, uint32_t buf_len);
uint32_t soma_ble_peer_count(const void *engine);
float    soma_ble_collective_phi(const void *engine);

/* Broca language generation */
char *soma_engine_generate_text(void *engine, uint32_t max_tokens);
char *soma_engine_generate_text_with_input(void *engine, const char *input, uint32_t max_tokens);
char *soma_engine_generate_embodied_text(void *engine);
uint8_t soma_engine_load_broca_checkpoint(void *engine, const uint8_t *data, uint32_t len);

/* Engagement */
void soma_engine_set_engagement_score(void *engine, float score);

/* Screen vision */
float soma_engine_inject_frame(void *engine, const uint8_t *data,
                               uint32_t width, uint32_t height, uint32_t channels);
void  soma_engine_touch_event(void *engine, float x, float y,
                              uint8_t action, float pressure);
char *soma_engine_screen_salient_regions_json(const void *engine);

/* Persistence */
uint8_t soma_engine_save_checkpoint(void *engine);
uint8_t soma_engine_load_checkpoint(void *engine);
void    soma_engine_set_storage_path(void *engine, const char *path);

/* String management */
void soma_string_free(char *s);


/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: convert a Rust-allocated C string to jstring, then free it.
 * Returns NULL jstring if rust_str is NULL.
 * ═══════════════════════════════════════════════════════════════════════════ */
static jstring rust_string_to_jstring(JNIEnv *env, char *rust_str) {
    if (rust_str == NULL) {
        return NULL;
    }
    jstring result = (*env)->NewStringUTF(env, rust_str);
    soma_string_free(rust_str);
    return result;
}

/* Helper: extract a C string from a nullable jstring. Returns NULL for null. */
static const char *jstring_to_cstr(JNIEnv *env, jstring jstr) {
    if (jstr == NULL) {
        return NULL;
    }
    return (*env)->GetStringUTFChars(env, jstr, NULL);
}

/* Helper: release a C string obtained from jstring_to_cstr. */
static void release_cstr(JNIEnv *env, jstring jstr, const char *cstr) {
    if (jstr != NULL && cstr != NULL) {
        (*env)->ReleaseStringUTFChars(env, jstr, cstr);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jlong JNICALL
Java_io_symthaea_soma_NativeBindings_engineNew(JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return (jlong)(intptr_t)soma_engine_new();
}

JNIEXPORT jlong JNICALL
Java_io_symthaea_soma_NativeBindings_engineNewMobile(JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return (jlong)(intptr_t)soma_engine_new_mobile();
}

JNIEXPORT jlong JNICALL
Java_io_symthaea_soma_NativeBindings_engineNewWithConfig(JNIEnv *env, jclass clazz,
                                                          jstring configJson) {
    (void)clazz;
    if (configJson == NULL) {
        return 0;
    }
    const char *json = (*env)->GetStringUTFChars(env, configJson, NULL);
    if (json == NULL) {
        return 0;
    }
    void *engine = soma_engine_new_with_config(json);
    (*env)->ReleaseStringUTFChars(env, configJson, json);
    return (jlong)(intptr_t)engine;
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_engineFree(JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    soma_engine_free((void *)(intptr_t)handle);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Core cycle
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jfloat JNICALL
Java_io_symthaea_soma_NativeBindings_cycle(JNIEnv *env, jclass clazz,
                                            jlong handle, jstring input) {
    (void)clazz;
    void *engine = (void *)(intptr_t)handle;
    const char *text = jstring_to_cstr(env, input);
    float result = soma_engine_cycle(engine, text);
    release_cstr(env, input, text);
    return result;
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_cycleJson(JNIEnv *env, jclass clazz,
                                                jlong handle, jstring input) {
    (void)clazz;
    void *engine = (void *)(intptr_t)handle;
    const char *text = jstring_to_cstr(env, input);
    char *json = soma_engine_cycle_json(engine, text);
    release_cstr(env, input, text);
    return rust_string_to_jstring(env, json);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — State inspection
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jfloat JNICALL
Java_io_symthaea_soma_NativeBindings_consciousnessLevel(JNIEnv *env, jclass clazz,
                                                         jlong handle) {
    (void)env; (void)clazz;
    return soma_engine_consciousness_level((const void *)(intptr_t)handle);
}

JNIEXPORT jlong JNICALL
Java_io_symthaea_soma_NativeBindings_cycleCount(JNIEnv *env, jclass clazz,
                                                 jlong handle) {
    (void)env; (void)clazz;
    return (jlong)soma_engine_cycle_count((const void *)(intptr_t)handle);
}

JNIEXPORT jfloat JNICALL
Java_io_symthaea_soma_NativeBindings_substrateFeasibility(JNIEnv *env, jclass clazz,
                                                           jlong handle) {
    (void)env; (void)clazz;
    return soma_engine_substrate_feasibility((const void *)(intptr_t)handle);
}

JNIEXPORT jfloat JNICALL
Java_io_symthaea_soma_NativeBindings_harmonyAlignment(JNIEnv *env, jclass clazz,
                                                       jlong handle) {
    (void)env; (void)clazz;
    return soma_engine_harmony_alignment((const void *)(intptr_t)handle);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_consciousnessReport(JNIEnv *env, jclass clazz,
                                                          jlong handle) {
    (void)clazz;
    char *report = soma_engine_consciousness_report((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, report);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_neuromodJson(JNIEnv *env, jclass clazz,
                                                   jlong handle) {
    (void)clazz;
    char *json = soma_engine_neuromod_json((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Platform integration
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setThermalLevel(JNIEnv *env, jclass clazz,
                                                      jlong handle, jint level) {
    (void)env; (void)clazz;
    soma_engine_set_thermal_level((void *)(intptr_t)handle, (uint8_t)level);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setBatteryState(JNIEnv *env, jclass clazz,
                                                      jlong handle,
                                                      jint chargePercent,
                                                      jboolean isCharging) {
    (void)env; (void)clazz;
    soma_engine_set_battery_state((void *)(intptr_t)handle,
                                   (uint8_t)chargePercent,
                                   isCharging ? 1 : 0);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setNightMode(JNIEnv *env, jclass clazz,
                                                   jlong handle, jboolean isNight) {
    (void)env; (void)clazz;
    soma_engine_set_night_mode((void *)(intptr_t)handle, isNight ? 1 : 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Dream
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jboolean JNICALL
Java_io_symthaea_soma_NativeBindings_dreamCycle(JNIEnv *env, jclass clazz,
                                                 jlong handle) {
    (void)env; (void)clazz;
    return soma_engine_dream_cycle((void *)(intptr_t)handle) != 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Metabolism (sleep/wake)
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_wakeSignal(JNIEnv *env, jclass clazz,
                                                 jlong handle, jint signal) {
    (void)env; (void)clazz;
    soma_engine_wake_signal((void *)(intptr_t)handle, (uint8_t)signal);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_soma_NativeBindings_wakeState(JNIEnv *env, jclass clazz,
                                                jlong handle) {
    (void)env; (void)clazz;
    return (jint)soma_engine_wake_state((const void *)(intptr_t)handle);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Sensors
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setSensors(JNIEnv *env, jclass clazz,
                                                 jlong handle,
                                                 jfloat accel, jfloat light,
                                                 jboolean proximityNear,
                                                 jfloat barometer,
                                                 jfloat gpsNovelty) {
    (void)env; (void)clazz;
    soma_engine_set_sensors((void *)(intptr_t)handle,
                             accel, light,
                             proximityNear ? 1 : 0,
                             barometer, gpsNovelty);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_soma_NativeBindings_motionState(JNIEnv *env, jclass clazz,
                                                  jlong handle) {
    (void)env; (void)clazz;
    return (jint)soma_engine_motion_state((const void *)(intptr_t)handle);
}

JNIEXPORT jboolean JNICALL
Java_io_symthaea_soma_NativeBindings_privacyMode(JNIEnv *env, jclass clazz,
                                                  jlong handle) {
    (void)env; (void)clazz;
    return soma_engine_privacy_mode((const void *)(intptr_t)handle) != 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Expanded senses
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setGyroscope(JNIEnv *env, jclass clazz,
                                                   jlong handle, jfloat rotationRate) {
    (void)env; (void)clazz;
    soma_engine_set_gyroscope((void *)(intptr_t)handle, rotationRate);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setStepDelta(JNIEnv *env, jclass clazz,
                                                   jlong handle, jint steps) {
    (void)env; (void)clazz;
    soma_engine_set_step_delta((void *)(intptr_t)handle, (uint32_t)steps);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setAmbientDb(JNIEnv *env, jclass clazz,
                                                   jlong handle, jfloat db) {
    (void)env; (void)clazz;
    soma_engine_set_ambient_db((void *)(intptr_t)handle, db);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setSocialPressure(JNIEnv *env, jclass clazz,
                                                        jlong handle,
                                                        jint notificationCount) {
    (void)env; (void)clazz;
    soma_engine_set_social_pressure((void *)(intptr_t)handle, (uint32_t)notificationCount);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setMediaState(JNIEnv *env, jclass clazz,
                                                    jlong handle, jint state) {
    (void)env; (void)clazz;
    soma_engine_set_media_state((void *)(intptr_t)handle, (uint8_t)state);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Consciousness compass
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_compassJson(JNIEnv *env, jclass clazz,
                                                  jlong handle) {
    (void)clazz;
    char *json = soma_engine_compass_json((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Sharing config
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setSharingConfig(JNIEnv *env, jclass clazz,
                                                       jlong handle, jstring json) {
    (void)clazz;
    if (json == NULL) {
        return;
    }
    const char *cstr = (*env)->GetStringUTFChars(env, json, NULL);
    if (cstr == NULL) {
        return;
    }
    soma_engine_set_sharing_config((void *)(intptr_t)handle, cstr);
    (*env)->ReleaseStringUTFChars(env, json, cstr);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Haptic awareness
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_hapticDrain(JNIEnv *env, jclass clazz,
                                                  jlong handle) {
    (void)clazz;
    char *json = soma_haptic_drain((void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_soma_NativeBindings_hapticPending(JNIEnv *env, jclass clazz,
                                                    jlong handle) {
    (void)env; (void)clazz;
    return (jint)soma_haptic_pending((const void *)(intptr_t)handle);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_hapticSetEnabled(JNIEnv *env, jclass clazz,
                                                       jlong handle,
                                                       jboolean enabled) {
    (void)env; (void)clazz;
    soma_haptic_set_enabled((void *)(intptr_t)handle, enabled ? 1 : 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Dream journal
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_dreamJournalLatest(JNIEnv *env, jclass clazz,
                                                         jlong handle) {
    (void)clazz;
    char *json = soma_dream_journal_latest((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_dreamJournalAll(JNIEnv *env, jclass clazz,
                                                      jlong handle) {
    (void)clazz;
    char *json = soma_dream_journal_all((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_soma_NativeBindings_dreamJournalCount(JNIEnv *env, jclass clazz,
                                                        jlong handle) {
    (void)env; (void)clazz;
    return (jint)soma_dream_journal_count((const void *)(intptr_t)handle);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_dreamConsolidate(JNIEnv *env, jclass clazz,
                                                       jlong handle) {
    (void)env; (void)clazz;
    soma_engine_dream_consolidate((void *)(intptr_t)handle);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Holon bridge
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_holonDrainOutbound(JNIEnv *env, jclass clazz,
                                                         jlong handle) {
    (void)clazz;
    char *json = soma_engine_holon_drain_outbound((void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_holonReceive(JNIEnv *env, jclass clazz,
                                                   jlong handle, jstring json) {
    (void)clazz;
    if (json == NULL) {
        return;
    }
    const char *cstr = (*env)->GetStringUTFChars(env, json, NULL);
    if (cstr == NULL) {
        return;
    }
    soma_engine_holon_receive((void *)(intptr_t)handle, cstr);
    (*env)->ReleaseStringUTFChars(env, json, cstr);
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_holonSetConnected(JNIEnv *env, jclass clazz,
                                                        jlong handle,
                                                        jboolean connected) {
    (void)env; (void)clazz;
    soma_engine_holon_set_connected((void *)(intptr_t)handle, connected ? 1 : 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — BLE mesh
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jboolean JNICALL
Java_io_symthaea_soma_NativeBindings_bleReceivePeer(JNIEnv *env, jclass clazz,
                                                     jlong handle,
                                                     jlong peerId,
                                                     jbyteArray cvData) {
    (void)clazz;
    if (cvData == NULL) {
        return JNI_FALSE;
    }
    jsize len = (*env)->GetArrayLength(env, cvData);
    if (len < 12) {
        return JNI_FALSE;
    }
    jbyte *buf = (*env)->GetByteArrayElements(env, cvData, NULL);
    if (buf == NULL) {
        return JNI_FALSE;
    }
    uint8_t result = soma_ble_receive_peer((void *)(intptr_t)handle,
                                            (uint64_t)peerId,
                                            (const uint8_t *)buf,
                                            (uint32_t)len);
    (*env)->ReleaseByteArrayElements(env, cvData, buf, JNI_ABORT);
    return result != 0;
}

JNIEXPORT jbyteArray JNICALL
Java_io_symthaea_soma_NativeBindings_bleAdvertisePayload(JNIEnv *env, jclass clazz,
                                                          jlong handle) {
    (void)clazz;
    /* Allocate a stack buffer large enough for the BLE payload (12 bytes typical). */
    uint8_t buf[64];
    uint32_t written = soma_ble_advertise_payload((void *)(intptr_t)handle,
                                                   buf, sizeof(buf));
    if (written == 0) {
        return (*env)->NewByteArray(env, 0);
    }
    jbyteArray result = (*env)->NewByteArray(env, (jsize)written);
    if (result != NULL) {
        (*env)->SetByteArrayRegion(env, result, 0, (jsize)written, (const jbyte *)buf);
    }
    return result;
}

JNIEXPORT jint JNICALL
Java_io_symthaea_soma_NativeBindings_blePeerCount(JNIEnv *env, jclass clazz,
                                                   jlong handle) {
    (void)env; (void)clazz;
    return (jint)soma_ble_peer_count((const void *)(intptr_t)handle);
}

JNIEXPORT jfloat JNICALL
Java_io_symthaea_soma_NativeBindings_bleCollectivePhi(JNIEnv *env, jclass clazz,
                                                       jlong handle) {
    (void)env; (void)clazz;
    return soma_ble_collective_phi((const void *)(intptr_t)handle);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Broca language generation
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_generateText(JNIEnv *env, jclass clazz,
                                                   jlong handle, jint maxTokens) {
    (void)clazz;
    char *json = soma_engine_generate_text((void *)(intptr_t)handle, (uint32_t)maxTokens);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_generateTextWithInput(JNIEnv *env, jclass clazz,
                                                            jlong handle, jstring input,
                                                            jint maxTokens) {
    (void)clazz;
    const char *cInput = (*env)->GetStringUTFChars(env, input, NULL);
    if (cInput == NULL) return NULL;
    char *json = soma_engine_generate_text_with_input(
        (void *)(intptr_t)handle, cInput, (uint32_t)maxTokens);
    (*env)->ReleaseStringUTFChars(env, input, cInput);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_generateEmbodiedText(JNIEnv *env, jclass clazz,
                                                           jlong handle) {
    (void)clazz;
    char *json = soma_engine_generate_embodied_text((void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jboolean JNICALL
Java_io_symthaea_soma_NativeBindings_loadBrocaCheckpoint(JNIEnv *env, jclass clazz,
                                                          jlong handle, jbyteArray data) {
    (void)clazz;
    if (data == NULL) return JNI_FALSE;
    jsize len = (*env)->GetArrayLength(env, data);
    jbyte *bytes = (*env)->GetByteArrayElements(env, data, NULL);
    if (bytes == NULL) return JNI_FALSE;
    uint8_t ok = soma_engine_load_broca_checkpoint(
        (void *)(intptr_t)handle, (const uint8_t *)bytes, (uint32_t)len);
    (*env)->ReleaseByteArrayElements(env, data, bytes, JNI_ABORT);
    return ok != 0;
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setEngagementScore(JNIEnv *env, jclass clazz,
                                                         jlong handle, jfloat score) {
    (void)env; (void)clazz;
    soma_engine_set_engagement_score((void *)(intptr_t)handle, score);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Persistence
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jboolean JNICALL
Java_io_symthaea_soma_NativeBindings_saveCheckpoint(JNIEnv *env, jclass clazz,
                                                     jlong handle) {
    (void)env; (void)clazz;
    return soma_engine_save_checkpoint((void *)(intptr_t)handle) != 0;
}

JNIEXPORT jboolean JNICALL
Java_io_symthaea_soma_NativeBindings_loadCheckpoint(JNIEnv *env, jclass clazz,
                                                     jlong handle) {
    (void)env; (void)clazz;
    return soma_engine_load_checkpoint((void *)(intptr_t)handle) != 0;
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_setStoragePath(JNIEnv *env, jclass clazz,
                                                     jlong handle, jstring path) {
    (void)clazz;
    if (path == NULL) return;
    const char *cstr = (*env)->GetStringUTFChars(env, path, NULL);
    if (cstr == NULL) return;
    soma_engine_set_storage_path((void *)(intptr_t)handle, cstr);
    (*env)->ReleaseStringUTFChars(env, path, cstr);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Screen vision
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jfloat JNICALL
Java_io_symthaea_soma_NativeBindings_injectFrame(JNIEnv *env, jclass clazz,
                                                  jlong handle, jbyteArray data,
                                                  jint width, jint height, jint channels) {
    (void)clazz;
    if (data == NULL || width <= 0 || height <= 0 || channels <= 0) return 0.0f;
    jsize len = (*env)->GetArrayLength(env, data);
    if (len < width * height * channels) return 0.0f;
    jbyte *bytes = (*env)->GetByteArrayElements(env, data, NULL);
    if (bytes == NULL) return 0.0f;
    float surprise = soma_engine_inject_frame(
        (void *)(intptr_t)handle,
        (const uint8_t *)bytes,
        (uint32_t)width, (uint32_t)height, (uint32_t)channels
    );
    (*env)->ReleaseByteArrayElements(env, data, bytes, JNI_ABORT);
    return surprise;
}

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_touchEvent(JNIEnv *env, jclass clazz,
                                                 jlong handle, jfloat x, jfloat y,
                                                 jint action, jfloat pressure) {
    (void)env; (void)clazz;
    soma_engine_touch_event(
        (void *)(intptr_t)handle, x, y, (uint8_t)action, pressure
    );
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_screenSalientRegionsJson(JNIEnv *env, jclass clazz,
                                                               jlong handle) {
    (void)clazz;
    char *json = soma_engine_screen_salient_regions_json((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Prism epistemic search (requires prism-search feature)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Forward declarations (from native_ffi.rs, feature-gated) */
void  soma_engine_prism_init(void *engine);
char *soma_engine_prism_search(const void *engine, const char *query, uint32_t top_k);
uint8_t soma_engine_prism_available(const void *engine);

JNIEXPORT void JNICALL
Java_io_symthaea_soma_NativeBindings_prismInit(JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    soma_engine_prism_init((void *)(intptr_t)handle);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_soma_NativeBindings_prismSearch(JNIEnv *env, jclass clazz,
                                                  jlong handle, jstring query, jint topK) {
    (void)clazz;
    if (query == NULL) return NULL;
    const char *q = (*env)->GetStringUTFChars(env, query, NULL);
    if (q == NULL) return NULL;
    char *json = soma_engine_prism_search((const void *)(intptr_t)handle, q, (uint32_t)topK);
    (*env)->ReleaseStringUTFChars(env, query, q);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jboolean JNICALL
Java_io_symthaea_soma_NativeBindings_prismAvailable(JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    return soma_engine_prism_available((const void *)(intptr_t)handle) != 0;
}

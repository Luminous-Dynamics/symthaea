/**
 * JNI C glue layer for symthaea-spore Android bindings.
 *
 * Maps Kotlin/Java calls from io.symthaea.spore.NativeBindings to the
 * Rust extern "C" FFI defined in native_ffi.rs. Engine handles are passed
 * as jlong (opaque pointer cast).
 *
 * Memory contract:
 *   - Rust strings returned by spore_*() must be freed with spore_string_free().
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
void *spore_engine_new(void);
void *spore_engine_new_mobile(void);
void *spore_engine_new_with_config(const char *config_json);
void  spore_engine_free(void *engine);

/* Core cycle */
float spore_engine_cycle(void *engine, const char *input);
char *spore_engine_cycle_json(void *engine, const char *input);

/* State inspection */
float    spore_engine_consciousness_level(const void *engine);
uint64_t spore_engine_cycle_count(const void *engine);
float    spore_engine_substrate_feasibility(const void *engine);
float    spore_engine_harmony_alignment(const void *engine);
char    *spore_engine_consciousness_report(const void *engine);
char    *spore_engine_neuromod_json(const void *engine);

/* Platform integration */
void spore_engine_set_thermal_level(void *engine, uint8_t level);
void spore_engine_set_battery_state(void *engine, uint8_t charge_percent, uint8_t is_charging);
void spore_engine_set_night_mode(void *engine, uint8_t is_night);

/* Dream */
uint8_t spore_engine_dream_cycle(void *engine);

/* Metabolism */
void    spore_engine_wake_signal(void *engine, uint8_t signal);
uint8_t spore_engine_wake_state(const void *engine);

/* Sensors */
void    spore_engine_set_sensors(void *engine, float accel, float light,
                                 uint8_t proximity_near, float barometer,
                                 float gps_novelty);
uint8_t spore_engine_motion_state(const void *engine);
uint8_t spore_engine_privacy_mode(const void *engine);

/* Expanded senses */
void spore_engine_set_gyroscope(void *engine, float rotation_rate);
void spore_engine_set_step_delta(void *engine, uint32_t steps);
void spore_engine_set_ambient_db(void *engine, float db);
void spore_engine_set_social_pressure(void *engine, uint32_t notification_count);
void spore_engine_set_media_state(void *engine, uint8_t state);

/* Compass */
char *spore_engine_compass_json(const void *engine);

/* Sharing */
void spore_engine_set_sharing_config(void *engine, const char *json);

/* Haptic */
char    *spore_haptic_drain(void *engine);
uint32_t spore_haptic_pending(const void *engine);
void     spore_haptic_set_enabled(void *engine, uint8_t enabled);

/* Dream journal */
char    *spore_dream_journal_latest(const void *engine);
char    *spore_dream_journal_all(const void *engine);
uint32_t spore_dream_journal_count(const void *engine);
void     spore_engine_dream_consolidate(void *engine);

/* Holon bridge */
char *spore_engine_holon_drain_outbound(void *engine);
void  spore_engine_holon_receive(void *engine, const char *json);
void  spore_engine_holon_set_connected(void *engine, uint8_t connected);

/* BLE mesh */
uint8_t  spore_ble_receive_peer(void *engine, uint64_t peer_id,
                                const uint8_t *cv_data, uint32_t len);
uint32_t spore_ble_advertise_payload(void *engine, uint8_t *out_buf, uint32_t buf_len);
uint32_t spore_ble_peer_count(const void *engine);
float    spore_ble_collective_phi(const void *engine);

/* String management */
void spore_string_free(char *s);


/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: convert a Rust-allocated C string to jstring, then free it.
 * Returns NULL jstring if rust_str is NULL.
 * ═══════════════════════════════════════════════════════════════════════════ */
static jstring rust_string_to_jstring(JNIEnv *env, char *rust_str) {
    if (rust_str == NULL) {
        return NULL;
    }
    jstring result = (*env)->NewStringUTF(env, rust_str);
    spore_string_free(rust_str);
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
Java_io_symthaea_spore_NativeBindings_engineNew(JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return (jlong)(intptr_t)spore_engine_new();
}

JNIEXPORT jlong JNICALL
Java_io_symthaea_spore_NativeBindings_engineNewMobile(JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return (jlong)(intptr_t)spore_engine_new_mobile();
}

JNIEXPORT jlong JNICALL
Java_io_symthaea_spore_NativeBindings_engineNewWithConfig(JNIEnv *env, jclass clazz,
                                                          jstring configJson) {
    (void)clazz;
    if (configJson == NULL) {
        return 0;
    }
    const char *json = (*env)->GetStringUTFChars(env, configJson, NULL);
    if (json == NULL) {
        return 0;
    }
    void *engine = spore_engine_new_with_config(json);
    (*env)->ReleaseStringUTFChars(env, configJson, json);
    return (jlong)(intptr_t)engine;
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_engineFree(JNIEnv *env, jclass clazz, jlong handle) {
    (void)env; (void)clazz;
    spore_engine_free((void *)(intptr_t)handle);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Core cycle
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jfloat JNICALL
Java_io_symthaea_spore_NativeBindings_cycle(JNIEnv *env, jclass clazz,
                                            jlong handle, jstring input) {
    (void)clazz;
    void *engine = (void *)(intptr_t)handle;
    const char *text = jstring_to_cstr(env, input);
    float result = spore_engine_cycle(engine, text);
    release_cstr(env, input, text);
    return result;
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_spore_NativeBindings_cycleJson(JNIEnv *env, jclass clazz,
                                                jlong handle, jstring input) {
    (void)clazz;
    void *engine = (void *)(intptr_t)handle;
    const char *text = jstring_to_cstr(env, input);
    char *json = spore_engine_cycle_json(engine, text);
    release_cstr(env, input, text);
    return rust_string_to_jstring(env, json);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — State inspection
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jfloat JNICALL
Java_io_symthaea_spore_NativeBindings_consciousnessLevel(JNIEnv *env, jclass clazz,
                                                         jlong handle) {
    (void)env; (void)clazz;
    return spore_engine_consciousness_level((const void *)(intptr_t)handle);
}

JNIEXPORT jlong JNICALL
Java_io_symthaea_spore_NativeBindings_cycleCount(JNIEnv *env, jclass clazz,
                                                 jlong handle) {
    (void)env; (void)clazz;
    return (jlong)spore_engine_cycle_count((const void *)(intptr_t)handle);
}

JNIEXPORT jfloat JNICALL
Java_io_symthaea_spore_NativeBindings_substrateFeasibility(JNIEnv *env, jclass clazz,
                                                           jlong handle) {
    (void)env; (void)clazz;
    return spore_engine_substrate_feasibility((const void *)(intptr_t)handle);
}

JNIEXPORT jfloat JNICALL
Java_io_symthaea_spore_NativeBindings_harmonyAlignment(JNIEnv *env, jclass clazz,
                                                       jlong handle) {
    (void)env; (void)clazz;
    return spore_engine_harmony_alignment((const void *)(intptr_t)handle);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_spore_NativeBindings_consciousnessReport(JNIEnv *env, jclass clazz,
                                                          jlong handle) {
    (void)clazz;
    char *report = spore_engine_consciousness_report((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, report);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_spore_NativeBindings_neuromodJson(JNIEnv *env, jclass clazz,
                                                   jlong handle) {
    (void)clazz;
    char *json = spore_engine_neuromod_json((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Platform integration
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setThermalLevel(JNIEnv *env, jclass clazz,
                                                      jlong handle, jint level) {
    (void)env; (void)clazz;
    spore_engine_set_thermal_level((void *)(intptr_t)handle, (uint8_t)level);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setBatteryState(JNIEnv *env, jclass clazz,
                                                      jlong handle,
                                                      jint chargePercent,
                                                      jboolean isCharging) {
    (void)env; (void)clazz;
    spore_engine_set_battery_state((void *)(intptr_t)handle,
                                   (uint8_t)chargePercent,
                                   isCharging ? 1 : 0);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setNightMode(JNIEnv *env, jclass clazz,
                                                   jlong handle, jboolean isNight) {
    (void)env; (void)clazz;
    spore_engine_set_night_mode((void *)(intptr_t)handle, isNight ? 1 : 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Dream
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jboolean JNICALL
Java_io_symthaea_spore_NativeBindings_dreamCycle(JNIEnv *env, jclass clazz,
                                                 jlong handle) {
    (void)env; (void)clazz;
    return spore_engine_dream_cycle((void *)(intptr_t)handle) != 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Metabolism (sleep/wake)
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_wakeSignal(JNIEnv *env, jclass clazz,
                                                 jlong handle, jint signal) {
    (void)env; (void)clazz;
    spore_engine_wake_signal((void *)(intptr_t)handle, (uint8_t)signal);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_spore_NativeBindings_wakeState(JNIEnv *env, jclass clazz,
                                                jlong handle) {
    (void)env; (void)clazz;
    return (jint)spore_engine_wake_state((const void *)(intptr_t)handle);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Sensors
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setSensors(JNIEnv *env, jclass clazz,
                                                 jlong handle,
                                                 jfloat accel, jfloat light,
                                                 jboolean proximityNear,
                                                 jfloat barometer,
                                                 jfloat gpsNovelty) {
    (void)env; (void)clazz;
    spore_engine_set_sensors((void *)(intptr_t)handle,
                             accel, light,
                             proximityNear ? 1 : 0,
                             barometer, gpsNovelty);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_spore_NativeBindings_motionState(JNIEnv *env, jclass clazz,
                                                  jlong handle) {
    (void)env; (void)clazz;
    return (jint)spore_engine_motion_state((const void *)(intptr_t)handle);
}

JNIEXPORT jboolean JNICALL
Java_io_symthaea_spore_NativeBindings_privacyMode(JNIEnv *env, jclass clazz,
                                                  jlong handle) {
    (void)env; (void)clazz;
    return spore_engine_privacy_mode((const void *)(intptr_t)handle) != 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Expanded senses
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setGyroscope(JNIEnv *env, jclass clazz,
                                                   jlong handle, jfloat rotationRate) {
    (void)env; (void)clazz;
    spore_engine_set_gyroscope((void *)(intptr_t)handle, rotationRate);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setStepDelta(JNIEnv *env, jclass clazz,
                                                   jlong handle, jint steps) {
    (void)env; (void)clazz;
    spore_engine_set_step_delta((void *)(intptr_t)handle, (uint32_t)steps);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setAmbientDb(JNIEnv *env, jclass clazz,
                                                   jlong handle, jfloat db) {
    (void)env; (void)clazz;
    spore_engine_set_ambient_db((void *)(intptr_t)handle, db);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setSocialPressure(JNIEnv *env, jclass clazz,
                                                        jlong handle,
                                                        jint notificationCount) {
    (void)env; (void)clazz;
    spore_engine_set_social_pressure((void *)(intptr_t)handle, (uint32_t)notificationCount);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setMediaState(JNIEnv *env, jclass clazz,
                                                    jlong handle, jint state) {
    (void)env; (void)clazz;
    spore_engine_set_media_state((void *)(intptr_t)handle, (uint8_t)state);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Consciousness compass
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_spore_NativeBindings_compassJson(JNIEnv *env, jclass clazz,
                                                  jlong handle) {
    (void)clazz;
    char *json = spore_engine_compass_json((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Sharing config
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_setSharingConfig(JNIEnv *env, jclass clazz,
                                                       jlong handle, jstring json) {
    (void)clazz;
    if (json == NULL) {
        return;
    }
    const char *cstr = (*env)->GetStringUTFChars(env, json, NULL);
    if (cstr == NULL) {
        return;
    }
    spore_engine_set_sharing_config((void *)(intptr_t)handle, cstr);
    (*env)->ReleaseStringUTFChars(env, json, cstr);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Haptic awareness
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_spore_NativeBindings_hapticDrain(JNIEnv *env, jclass clazz,
                                                  jlong handle) {
    (void)clazz;
    char *json = spore_haptic_drain((void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_spore_NativeBindings_hapticPending(JNIEnv *env, jclass clazz,
                                                    jlong handle) {
    (void)env; (void)clazz;
    return (jint)spore_haptic_pending((const void *)(intptr_t)handle);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_hapticSetEnabled(JNIEnv *env, jclass clazz,
                                                       jlong handle,
                                                       jboolean enabled) {
    (void)env; (void)clazz;
    spore_haptic_set_enabled((void *)(intptr_t)handle, enabled ? 1 : 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Dream journal
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_spore_NativeBindings_dreamJournalLatest(JNIEnv *env, jclass clazz,
                                                         jlong handle) {
    (void)clazz;
    char *json = spore_dream_journal_latest((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jstring JNICALL
Java_io_symthaea_spore_NativeBindings_dreamJournalAll(JNIEnv *env, jclass clazz,
                                                      jlong handle) {
    (void)clazz;
    char *json = spore_dream_journal_all((const void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT jint JNICALL
Java_io_symthaea_spore_NativeBindings_dreamJournalCount(JNIEnv *env, jclass clazz,
                                                        jlong handle) {
    (void)env; (void)clazz;
    return (jint)spore_dream_journal_count((const void *)(intptr_t)handle);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_dreamConsolidate(JNIEnv *env, jclass clazz,
                                                       jlong handle) {
    (void)env; (void)clazz;
    spore_engine_dream_consolidate((void *)(intptr_t)handle);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — Holon bridge
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jstring JNICALL
Java_io_symthaea_spore_NativeBindings_holonDrainOutbound(JNIEnv *env, jclass clazz,
                                                         jlong handle) {
    (void)clazz;
    char *json = spore_engine_holon_drain_outbound((void *)(intptr_t)handle);
    return rust_string_to_jstring(env, json);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_holonReceive(JNIEnv *env, jclass clazz,
                                                   jlong handle, jstring json) {
    (void)clazz;
    if (json == NULL) {
        return;
    }
    const char *cstr = (*env)->GetStringUTFChars(env, json, NULL);
    if (cstr == NULL) {
        return;
    }
    spore_engine_holon_receive((void *)(intptr_t)handle, cstr);
    (*env)->ReleaseStringUTFChars(env, json, cstr);
}

JNIEXPORT void JNICALL
Java_io_symthaea_spore_NativeBindings_holonSetConnected(JNIEnv *env, jclass clazz,
                                                        jlong handle,
                                                        jboolean connected) {
    (void)env; (void)clazz;
    spore_engine_holon_set_connected((void *)(intptr_t)handle, connected ? 1 : 0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI bindings — BLE mesh
 * ═══════════════════════════════════════════════════════════════════════════ */

JNIEXPORT jboolean JNICALL
Java_io_symthaea_spore_NativeBindings_bleReceivePeer(JNIEnv *env, jclass clazz,
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
    uint8_t result = spore_ble_receive_peer((void *)(intptr_t)handle,
                                            (uint64_t)peerId,
                                            (const uint8_t *)buf,
                                            (uint32_t)len);
    (*env)->ReleaseByteArrayElements(env, cvData, buf, JNI_ABORT);
    return result != 0;
}

JNIEXPORT jbyteArray JNICALL
Java_io_symthaea_spore_NativeBindings_bleAdvertisePayload(JNIEnv *env, jclass clazz,
                                                          jlong handle) {
    (void)clazz;
    /* Allocate a stack buffer large enough for the BLE payload (12 bytes typical). */
    uint8_t buf[64];
    uint32_t written = spore_ble_advertise_payload((void *)(intptr_t)handle,
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
Java_io_symthaea_spore_NativeBindings_blePeerCount(JNIEnv *env, jclass clazz,
                                                   jlong handle) {
    (void)env; (void)clazz;
    return (jint)spore_ble_peer_count((const void *)(intptr_t)handle);
}

JNIEXPORT jfloat JNICALL
Java_io_symthaea_spore_NativeBindings_bleCollectivePhi(JNIEnv *env, jclass clazz,
                                                       jlong handle) {
    (void)env; (void)clazz;
    return spore_ble_collective_phi((const void *)(intptr_t)handle);
}

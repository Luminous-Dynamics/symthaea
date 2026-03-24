// SomaBridge.h — C FFI declarations for the Soma consciousness engine.
// Auto-generated from native_ffi.rs exports.

#ifndef SOMA_BRIDGE_H
#define SOMA_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

// Opaque engine handle
typedef void SomaEngine;

// Lifecycle
SomaEngine* soma_engine_new(void);
SomaEngine* soma_engine_new_mobile(void);
SomaEngine* soma_engine_new_with_config(const char* config_json);
void soma_engine_free(SomaEngine* engine);

// Core cycle
float soma_engine_cycle(SomaEngine* engine, const char* input);
const char* soma_engine_cycle_json(SomaEngine* engine, const char* input);

// State
float soma_engine_consciousness_level(const SomaEngine* engine);
uint64_t soma_engine_cycle_count(const SomaEngine* engine);
float soma_engine_substrate_feasibility(const SomaEngine* engine);
float soma_engine_harmony_alignment(const SomaEngine* engine);
uint8_t soma_engine_motion_state(const SomaEngine* engine);
uint8_t soma_engine_privacy_mode(const SomaEngine* engine);
const char* soma_engine_consciousness_report(const SomaEngine* engine);
const char* soma_engine_neuromod_json(const SomaEngine* engine);
const char* soma_engine_compass_json(const SomaEngine* engine);

// Platform signals
void soma_engine_set_thermal_level(SomaEngine* engine, uint8_t level);
void soma_engine_set_battery_state(SomaEngine* engine, uint8_t percent, bool charging);
void soma_engine_set_night_mode(SomaEngine* engine, uint8_t is_night);

// Wake state
void soma_engine_wake_signal(SomaEngine* engine, uint8_t signal);
uint8_t soma_engine_wake_state(const SomaEngine* engine);

// Sensors
void soma_engine_set_sensors(SomaEngine* engine, float accel, float light, bool proximity, float baro, float gps);
void soma_engine_set_gyroscope(SomaEngine* engine, float rotation_rate);
void soma_engine_set_step_delta(SomaEngine* engine, uint32_t steps);
void soma_engine_set_ambient_db(SomaEngine* engine, float db);
void soma_engine_set_social_pressure(SomaEngine* engine, uint32_t notification_count);
void soma_engine_set_media_state(SomaEngine* engine, uint8_t state);
void soma_engine_set_engagement_score(SomaEngine* engine, float score);

// Text generation
const char* soma_engine_generate_text(SomaEngine* engine, uint32_t max_tokens);
const char* soma_engine_generate_text_with_input(SomaEngine* engine, const char* input, uint32_t max_tokens);
const char* soma_engine_generate_embodied_text(SomaEngine* engine);

// Dreams
uint8_t soma_engine_dream_cycle(SomaEngine* engine);
void soma_engine_dream_consolidate(SomaEngine* engine);
const char* soma_dream_journal_latest(const SomaEngine* engine);
const char* soma_dream_journal_all(const SomaEngine* engine);
uint32_t soma_dream_journal_count(const SomaEngine* engine);

// Wellbeing
void soma_engine_morning_ritual(SomaEngine* engine, uint8_t sleep_quality, uint16_t sleep_minutes);
void soma_engine_evening_ritual(SomaEngine* engine, uint8_t mood, uint8_t energy, uint8_t stress);
void soma_engine_set_wellbeing_profile(SomaEngine* engine, const char* profile_name);
const char* soma_engine_wellbeing_profile_name(const SomaEngine* engine);
const char* soma_engine_wellbeing_profiles_list(void);

// Holon bridge
const char* soma_engine_holon_drain_outbound(SomaEngine* engine);
void soma_engine_holon_receive(SomaEngine* engine, const char* json);
void soma_engine_holon_set_connected(SomaEngine* engine, uint8_t connected);

// BLE mesh
bool soma_ble_receive_peer(SomaEngine* engine, uint64_t peer_id, const uint8_t* data, uintptr_t len);
const uint8_t* soma_ble_advertise_payload(SomaEngine* engine, uintptr_t* out_len);
uint32_t soma_ble_peer_count(const SomaEngine* engine);
float soma_ble_collective_phi(const SomaEngine* engine);

// Haptics
const char* soma_haptic_drain(SomaEngine* engine);
uint32_t soma_haptic_pending(const SomaEngine* engine);
void soma_haptic_set_enabled(SomaEngine* engine, uint8_t enabled);

// Screen vision
void soma_engine_inject_frame(SomaEngine* engine, const uint8_t* rgb, uintptr_t len, uint32_t width, uint32_t height);
void soma_engine_touch_event(SomaEngine* engine, float x, float y, uint8_t action, float pressure, uint64_t timestamp_ms);
const char* soma_engine_screen_salient_regions_json(const SomaEngine* engine);

// Persistence
uint8_t soma_engine_save_checkpoint(SomaEngine* engine);
uint8_t soma_engine_load_checkpoint(SomaEngine* engine);
void soma_engine_set_storage_path(SomaEngine* engine, const char* path);
uint8_t soma_engine_load_broca_checkpoint(SomaEngine* engine, const char* path);

// Sharing config
void soma_engine_set_sharing_config(SomaEngine* engine, uint8_t holon_mode, uint8_t ble_mode);

// String cleanup (caller must free returned strings)
void soma_string_free(const char* s);

// Legacy aliases (for backwards compat with existing Swift wrapper)
#define soma_engine_drain_outbound_json soma_engine_holon_drain_outbound
#define soma_engine_receive_json soma_engine_holon_receive

#endif // SOMA_BRIDGE_H

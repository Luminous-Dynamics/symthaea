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
const char* soma_engine_cycle(SomaEngine* engine, const char* input);
const char* soma_engine_cycle_json(SomaEngine* engine, const char* input);

// State
float soma_engine_consciousness_level(const SomaEngine* engine);
uint64_t soma_engine_cycle_count(const SomaEngine* engine);
float soma_engine_substrate_feasibility(const SomaEngine* engine);
float soma_engine_harmony_alignment(const SomaEngine* engine);

// Platform signals
void soma_engine_set_thermal_level(SomaEngine* engine, uint8_t level);
void soma_engine_set_battery_state(SomaEngine* engine, uint8_t percent, bool charging);
void soma_engine_set_night_mode(SomaEngine* engine, bool night);

// Wake state
void soma_engine_wake_signal(SomaEngine* engine);
uint8_t soma_engine_wake_state(const SomaEngine* engine);

// Sensors
void soma_engine_set_sensors(SomaEngine* engine, float accel, float light, bool proximity, float baro, float gps);
void soma_engine_set_gyroscope(SomaEngine* engine, float rotation_rate);
void soma_engine_set_step_delta(SomaEngine* engine, uint32_t steps);
void soma_engine_set_ambient_db(SomaEngine* engine, float db);
void soma_engine_set_social_pressure(SomaEngine* engine, uint32_t notification_count);
void soma_engine_set_media_state(SomaEngine* engine, uint8_t state);

// Text generation
const char* soma_engine_generate_text(SomaEngine* engine, uint32_t max_tokens);
const char* soma_engine_generate_text_with_input(SomaEngine* engine, const char* input, uint32_t max_tokens);

// Dreams
bool soma_engine_dream_consolidate(SomaEngine* engine);

// Holon bridge
const char* soma_engine_drain_outbound_json(SomaEngine* engine);
void soma_engine_receive_json(SomaEngine* engine, const char* json);
void soma_engine_set_holon_connected(SomaEngine* engine, bool connected);

// String cleanup (caller must free returned strings)
void soma_string_free(const char* s);

#endif // SOMA_BRIDGE_H

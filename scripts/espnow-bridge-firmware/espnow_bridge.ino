/**
 * ESP-NOW SLIP Bridge — Relays serial SLIP frames to/from ESP-NOW broadcast.
 *
 * This is the "dumb radio" firmware for the Symthaea mesh network.
 * All intelligence (consciousness, routing, compression) lives on the host.
 * The ESP32 just relays bytes between serial and ESP-NOW.
 *
 * Protocol: SLIP (RFC 1055) over serial ↔ ESP-NOW broadcast (250 byte max)
 *
 * Hardware: Any ESP32 board (DevKitC, S3, C3, etc.)
 * Baud: 921600 (configurable)
 * WiFi Channel: 1 (must match all peers)
 */

#include <esp_now.h>
#include <WiFi.h>

// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════

#define BAUD_RATE    921600
#define WIFI_CHANNEL 1
#define MAX_PAYLOAD  250  // ESP-NOW max

// SLIP constants
#define SLIP_END     0xC0
#define SLIP_ESC     0xDB
#define SLIP_ESC_END 0xDC
#define SLIP_ESC_ESC 0xDD

// Broadcast MAC (all peers)
uint8_t broadcastMAC[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ═══════════════════════════════════════════════════════════════════
// SLIP decoder state machine
// ═══════════════════════════════════════════════════════════════════

uint8_t slip_buf[MAX_PAYLOAD];
int slip_len = 0;
bool slip_escape = false;

// ═══════════════════════════════════════════════════════════════════
// ESP-NOW callbacks
// ═══════════════════════════════════════════════════════════════════

// Called when we receive an ESP-NOW frame from the air.
// Encode it as SLIP and send over serial to the host.
void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
    if (len <= 0 || len > MAX_PAYLOAD) return;

    // SLIP encode and write to serial
    Serial.write(SLIP_END);
    for (int i = 0; i < len; i++) {
        switch (data[i]) {
            case SLIP_END:
                Serial.write(SLIP_ESC);
                Serial.write(SLIP_ESC_END);
                break;
            case SLIP_ESC:
                Serial.write(SLIP_ESC);
                Serial.write(SLIP_ESC_ESC);
                break;
            default:
                Serial.write(data[i]);
                break;
        }
    }
    Serial.write(SLIP_END);
    Serial.flush();
}

// Called after ESP-NOW send completes (for debugging).
void onSent(const uint8_t *mac, esp_now_send_status_t status) {
    // Could blink LED on failure, but keep it minimal
}

// ═══════════════════════════════════════════════════════════════════
// Setup
// ═══════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(BAUD_RATE);

    // Init WiFi in station mode (required for ESP-NOW)
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    // Set WiFi channel
    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

    // Init ESP-NOW
    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init failed");
        return;
    }

    // Register callbacks
    esp_now_register_recv_cb(onReceive);
    esp_now_register_send_cb(onSent);

    // Add broadcast peer
    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, broadcastMAC, 6);
    peer.channel = WIFI_CHANNEL;
    peer.encrypt = false;
    esp_now_add_peer(&peer);
}

// ═══════════════════════════════════════════════════════════════════
// Main loop — read SLIP from serial, send via ESP-NOW
// ═══════════════════════════════════════════════════════════════════

void loop() {
    while (Serial.available()) {
        uint8_t b = Serial.read();

        if (slip_escape) {
            slip_escape = false;
            switch (b) {
                case SLIP_ESC_END: slip_buf[slip_len++] = SLIP_END; break;
                case SLIP_ESC_ESC: slip_buf[slip_len++] = SLIP_ESC; break;
                default: slip_len = 0; break; // Invalid escape
            }
        } else {
            switch (b) {
                case SLIP_END:
                    if (slip_len > 0) {
                        // Complete frame — send via ESP-NOW
                        esp_now_send(broadcastMAC, slip_buf, slip_len);
                        slip_len = 0;
                    }
                    break;
                case SLIP_ESC:
                    slip_escape = true;
                    break;
                default:
                    if (slip_len < MAX_PAYLOAD) {
                        slip_buf[slip_len++] = b;
                    } else {
                        slip_len = 0; // Overflow — drop frame
                    }
                    break;
            }
        }
    }
}

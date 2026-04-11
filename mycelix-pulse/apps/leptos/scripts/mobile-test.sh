#!/usr/bin/env bash
# Mycelix Pulse — Mobile Test Pipeline v2
# Usage: ./scripts/mobile-test.sh [--skip-build] [--pages inbox,compose,chat] [--baseline] [--diff]
#
# Modes:
#   (default)      Build + test + screenshot each page
#   --skip-build   Skip trunk build, test existing dist/
#   --baseline     Save screenshots as baseline for future comparison
#   --diff         Compare current screenshots against baseline
#   --interactive  Run touch interaction tests (tap, type, swipe)
#   --perf         Measure WASM load time + page paint timing
#   --pwa          Test PWA install flow
#   --full         Run everything (baseline + interactive + perf + pwa)
#
# Prerequisites:
#   - ADB connected (adb devices shows your phone)
#   - SPA server running on :8117 OR script will start one
#   - trunk installed (~/.cargo/bin/trunk)
#   - ImageMagick (compare) for --diff mode

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ──
SKIP_BUILD=false
DO_BASELINE=false
DO_DIFF=false
DO_INTERACTIVE=false
DO_PERF=false
DO_PWA=false
PAGES="inbox compose contacts search settings calendar chat meet"
SCREENSHOT_DIR="/tmp/mycelix-mobile-test-$(date +%Y%m%d-%H%M%S)"
BASELINE_DIR="$(dirname "$0")/baselines"
SPA_PORT=8117
ADB="${ADB_CMD:-adb}"
TRUNK="${HOME}/.cargo/bin/trunk"
WAIT_AFTER_NAV=3  # seconds to wait after navigating before screenshot

# ── Parse args ──
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=true ;;
        --baseline) DO_BASELINE=true ;;
        --diff) DO_DIFF=true ;;
        --interactive) DO_INTERACTIVE=true ;;
        --perf) DO_PERF=true ;;
        --pwa) DO_PWA=true ;;
        --full) DO_BASELINE=true; DO_INTERACTIVE=true; DO_PERF=true; DO_PWA=true ;;
        --pages=*) PAGES="${arg#*=}" ;;
        --help|-h)
            head -16 "$0" | tail -15
            exit 0
            ;;
    esac
done

# ── Helpers ──
adb_screenshot() {
    local dest="$1"
    $ADB shell screencap -p /sdcard/_mtest.png 2>/dev/null
    $ADB pull /sdcard/_mtest.png "$dest" >/dev/null 2>&1
    $ADB shell rm /sdcard/_mtest.png 2>/dev/null
}

adb_tap() {
    $ADB shell input tap "$1" "$2"
}

adb_swipe() {
    $ADB shell input swipe "$1" "$2" "$3" "$4" "${5:-300}"
}

adb_type() {
    # Escape spaces for ADB
    local text="${1// /%s}"
    $ADB shell input text "$text"
}

adb_key() {
    $ADB shell input keyevent "$1"
}

adb_open_url() {
    $ADB shell am start -a android.intent.action.VIEW -d "$1" >/dev/null 2>&1
}

get_lan_ip() {
    ip -4 addr show | grep -oP '(?<=inet )(10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)' | head -1
}

timestamp_ms() {
    echo $(($(date +%s%N) / 1000000))
}

# ── Preflight ──
if ! $ADB devices 2>/dev/null | grep -q "device$"; then
    echo "ERROR: No ADB device found."
    echo "  1. Connect phone via USB"
    echo "  2. Enable USB debugging in Developer Options"
    echo "  3. Authorize this computer on the phone"
    exit 1
fi

DEVICE=$($ADB devices | grep "device$" | head -1 | cut -f1)
MODEL=$($ADB -s "$DEVICE" shell getprop ro.product.model 2>/dev/null)
ANDROID=$($ADB -s "$DEVICE" shell getprop ro.build.version.release 2>/dev/null)
SCREEN=$($ADB -s "$DEVICE" shell wm size 2>/dev/null | grep -oP '\d+x\d+')

echo "╔══════════════════════════════════════╗"
echo "║   Mycelix Pulse — Mobile Test v2     ║"
echo "╠══════════════════════════════════════╣"
echo "║ Device:  $MODEL"
echo "║ Android: $ANDROID"
echo "║ Screen:  $SCREEN"
echo "║ Device:  $DEVICE"
echo "╚══════════════════════════════════════╝"
echo ""

mkdir -p "$SCREENSHOT_DIR"

# ── Step 1: Build ──
if [ "$SKIP_BUILD" = false ]; then
    echo "━━━ [1/6] Building release WASM ━━━"
    START=$(timestamp_ms)
    CARGO_TARGET_DIR=/tmp/mycelix-trunk-target $TRUNK build --release 2>&1 | tail -5
    END=$(timestamp_ms)
    BUILD_TIME=$(( (END - START) / 1000 ))
    WASM_FILE=$(ls -S dist/*.wasm 2>/dev/null | head -1)
    if [ -n "$WASM_FILE" ]; then
        WASM_SIZE=$(du -h "$WASM_FILE" | cut -f1)
        WASM_GZIP=$(gzip -c "$WASM_FILE" | wc -c | awk '{printf "%.0fKB", $1/1024}')
        echo "  WASM: $WASM_SIZE raw, $WASM_GZIP gzipped"
    fi
    echo "  Build time: ${BUILD_TIME}s"
else
    echo "━━━ [1/6] Skipping build (--skip-build) ━━━"
fi

# ── Step 2: SPA server ──
echo ""
echo "━━━ [2/6] SPA Server ━━━"
SPA_PID=""
if ! curl -s -o /dev/null "http://localhost:${SPA_PORT}" 2>/dev/null; then
    echo "  Starting SPA server on :${SPA_PORT}..."
    cd dist && python3 -m http.server "$SPA_PORT" --bind 0.0.0.0 &>/dev/null &
    SPA_PID=$!
    cd ..
    sleep 1
    echo "  Started (PID: $SPA_PID)"
else
    echo "  Already running on :${SPA_PORT}"
fi

LAN_IP=$(get_lan_ip)
BASE_URL="http://${LAN_IP:-localhost}:${SPA_PORT}"
echo "  URL: $BASE_URL"

# ── Step 3: Page screenshots ──
echo ""
echo "━━━ [3/6] Page Screenshots ━━━"
PASS=0
FAIL=0

for page in $PAGES; do
    route="/"
    case "$page" in
        inbox) route="/" ;;
        *) route="/$page" ;;
    esac

    printf "  %-12s " "$page"

    adb_open_url "${BASE_URL}${route}"
    sleep "$WAIT_AFTER_NAV"

    SHOT="${SCREENSHOT_DIR}/${page}.png"
    adb_screenshot "$SHOT"

    if [ -f "$SHOT" ] && [ "$(stat -c%s "$SHOT" 2>/dev/null || echo 0)" -gt 1000 ]; then
        SIZE=$(du -h "$SHOT" | cut -f1)
        echo "✓ ($SIZE)"
        PASS=$((PASS + 1))

        # Save as baseline if requested
        if [ "$DO_BASELINE" = true ]; then
            mkdir -p "$BASELINE_DIR"
            cp "$SHOT" "$BASELINE_DIR/${page}.png"
        fi
    else
        echo "✗ FAIL"
        FAIL=$((FAIL + 1))
    fi
done

echo "  Result: $PASS passed, $FAIL failed"

# ── Step 4: Screenshot diff against baseline ──
if [ "$DO_DIFF" = true ]; then
    echo ""
    echo "━━━ [4/6] Screenshot Diff ━━━"
    if [ ! -d "$BASELINE_DIR" ]; then
        echo "  No baseline found. Run with --baseline first."
    elif ! command -v compare &>/dev/null; then
        echo "  ImageMagick 'compare' not found. Install with: nix-shell -p imagemagick"
    else
        DIFF_DIR="${SCREENSHOT_DIR}/diffs"
        mkdir -p "$DIFF_DIR"
        for page in $PAGES; do
            BASELINE="${BASELINE_DIR}/${page}.png"
            CURRENT="${SCREENSHOT_DIR}/${page}.png"
            if [ -f "$BASELINE" ] && [ -f "$CURRENT" ]; then
                printf "  %-12s " "$page"
                DIFF_FILE="${DIFF_DIR}/${page}_diff.png"
                # compare returns non-zero if images differ
                METRIC=$(compare -metric RMSE "$BASELINE" "$CURRENT" "$DIFF_FILE" 2>&1 || true)
                RMSE=$(echo "$METRIC" | grep -oP '^\d+(\.\d+)?' || echo "?")
                if [ "$RMSE" = "0" ] 2>/dev/null; then
                    echo "✓ identical"
                else
                    echo "△ changed (RMSE: $RMSE) → $DIFF_FILE"
                fi
            fi
        done
    fi
else
    echo ""
    echo "━━━ [4/6] Screenshot Diff (skipped, use --diff) ━━━"
fi

# ── Step 5: Interactive touch tests ──
if [ "$DO_INTERACTIVE" = true ]; then
    echo ""
    echo "━━━ [5/6] Interactive Tests ━━━"

    # Get screen dimensions for relative positioning
    SCREEN_W=$(echo "$SCREEN" | cut -dx -f1)
    SCREEN_H=$(echo "$SCREEN" | cut -dx -f2)
    CENTER_X=$((SCREEN_W / 2))

    # Test 1: Navigate to inbox, tap compose button
    echo "  [test] Tap compose button..."
    adb_open_url "${BASE_URL}/"
    sleep 3
    # Bottom nav compose is typically center-left area
    # Tap the "Compose" nav link in bottom bar (approximate position)
    BOTTOM_Y=$((SCREEN_H - 60))
    adb_tap $((SCREEN_W / 4)) "$BOTTOM_Y"
    sleep 2
    adb_screenshot "${SCREENSHOT_DIR}/test_compose_nav.png"
    echo "    → screenshot: test_compose_nav.png"

    # Test 2: Type in compose To field
    echo "  [test] Type in To field..."
    # Tap the To input field (approx top third of compose page)
    adb_tap "$CENTER_X" $((SCREEN_H / 4))
    sleep 1
    adb_type "alice@mycelix.net"
    sleep 1
    adb_screenshot "${SCREENSHOT_DIR}/test_compose_typing.png"
    echo "    → screenshot: test_compose_typing.png"

    # Test 3: Type in subject
    echo "  [test] Type subject..."
    adb_key 61  # TAB to next field
    sleep 0.5
    adb_type "Test%sfrom%sPixel"
    sleep 1
    adb_screenshot "${SCREENSHOT_DIR}/test_compose_subject.png"
    echo "    → screenshot: test_compose_subject.png"

    # Test 4: Navigate to chat, swipe test
    echo "  [test] Chat swipe gesture..."
    adb_open_url "${BASE_URL}/chat"
    sleep 3
    # Swipe right on a chat bubble (mid-screen)
    adb_swipe $((CENTER_X - 100)) $((SCREEN_H / 2)) $((CENTER_X + 200)) $((SCREEN_H / 2)) 200
    sleep 1
    adb_screenshot "${SCREENSHOT_DIR}/test_chat_swipe.png"
    echo "    → screenshot: test_chat_swipe.png"

    # Test 5: Calendar NLP input
    echo "  [test] Calendar NLP event creation..."
    adb_open_url "${BASE_URL}/calendar"
    sleep 3
    # Tap the NLP input bar (near top)
    adb_tap "$CENTER_X" $((SCREEN_H / 5))
    sleep 1
    adb_type "Meeting%stomorrow%sat%s2pm"
    sleep 0.5
    # Tap Add button
    adb_tap $((SCREEN_W - 80)) $((SCREEN_H / 5))
    sleep 1
    adb_screenshot "${SCREENSHOT_DIR}/test_calendar_nlp.png"
    echo "    → screenshot: test_calendar_nlp.png"

    # Test 6: Settings search
    echo "  [test] Settings search filter..."
    adb_open_url "${BASE_URL}/settings"
    sleep 3
    # Tap search input in settings sidebar
    adb_tap 150 $((SCREEN_H / 6))
    sleep 0.5
    adb_type "encrypt"
    sleep 1
    adb_screenshot "${SCREENSHOT_DIR}/test_settings_search.png"
    echo "    → screenshot: test_settings_search.png"

    # Test 7: Theme toggle
    echo "  [test] Theme toggle..."
    adb_open_url "${BASE_URL}/"
    sleep 2
    # Tap theme toggle in navbar (top right area)
    adb_tap $((SCREEN_W - 120)) 40
    sleep 1
    adb_screenshot "${SCREENSHOT_DIR}/test_theme_toggle.png"
    echo "    → screenshot: test_theme_toggle.png"

    echo "  Interactive tests complete: 7 scenarios"
else
    echo ""
    echo "━━━ [5/6] Interactive Tests (skipped, use --interactive) ━━━"
fi

# ── Step 6: Performance timing ──
if [ "$DO_PERF" = true ]; then
    echo ""
    echo "━━━ [6/6] Performance Timing ━━━"

    # Clear browser cache first
    echo "  Clearing browser data..."
    $ADB shell pm clear com.brave.browser 2>/dev/null || \
    $ADB shell pm clear com.android.chrome 2>/dev/null || true
    sleep 1

    # Cold load timing
    echo "  Cold load timing..."
    LOAD_START=$(timestamp_ms)
    adb_open_url "${BASE_URL}/"

    # Poll for page load (check if loading screen is gone)
    LOADED=false
    for i in $(seq 1 20); do
        sleep 0.5
        # Take a quick screenshot and check if it's not the loading screen
        adb_screenshot "/tmp/_perf_check.png"
        FSIZE=$(stat -c%s "/tmp/_perf_check.png" 2>/dev/null || echo 0)
        if [ "$FSIZE" -gt 50000 ]; then
            # Large screenshot = app has rendered (loading screen is simpler)
            LOADED=true
            break
        fi
    done
    LOAD_END=$(timestamp_ms)
    LOAD_TIME=$(( LOAD_END - LOAD_START ))

    if [ "$LOADED" = true ]; then
        echo "  Cold load: ${LOAD_TIME}ms"
    else
        echo "  Cold load: timeout (>${LOAD_TIME}ms)"
    fi
    adb_screenshot "${SCREENSHOT_DIR}/perf_cold_load.png"

    # Warm load timing (second visit)
    echo "  Warm load timing..."
    adb_key 4  # back
    sleep 1
    WARM_START=$(timestamp_ms)
    adb_open_url "${BASE_URL}/"
    sleep 2
    WARM_END=$(timestamp_ms)
    WARM_TIME=$(( WARM_END - WARM_START ))
    echo "  Warm load: ${WARM_TIME}ms"
    adb_screenshot "${SCREENSHOT_DIR}/perf_warm_load.png"

    # Navigation timing (page to page)
    echo "  Navigation timing..."
    NAV_START=$(timestamp_ms)
    adb_open_url "${BASE_URL}/compose"
    sleep 2
    NAV_END=$(timestamp_ms)
    NAV_TIME=$(( NAV_END - NAV_START ))
    echo "  Nav (→compose): ${NAV_TIME}ms"

    # Scroll performance (inbox)
    echo "  Scroll performance..."
    adb_open_url "${BASE_URL}/"
    sleep 3
    SCROLL_START=$(timestamp_ms)
    for i in $(seq 1 5); do
        adb_swipe "$CENTER_X" $((SCREEN_H * 3 / 4)) "$CENTER_X" $((SCREEN_H / 4)) 150
        sleep 0.3
    done
    SCROLL_END=$(timestamp_ms)
    SCROLL_TIME=$(( SCROLL_END - SCROLL_START ))
    echo "  5x scroll: ${SCROLL_TIME}ms (${SCROLL_TIME}ms / 5 = $((SCROLL_TIME / 5))ms avg)"
    adb_screenshot "${SCREENSHOT_DIR}/perf_scroll.png"

    # PWA install test
    if [ "$DO_PWA" = true ]; then
        echo ""
        echo "  PWA Install Test..."
        # Check if manifest.json is accessible
        MANIFEST=$(curl -s "http://localhost:${SPA_PORT}/manifest.json" 2>/dev/null)
        if echo "$MANIFEST" | grep -q "Mycelix"; then
            echo "  ✓ manifest.json accessible"
            # Check service worker
            SW=$(curl -s "http://localhost:${SPA_PORT}/sw.js" 2>/dev/null | head -1)
            if echo "$SW" | grep -q "Mycelix\|Service Worker"; then
                echo "  ✓ sw.js accessible"
            else
                echo "  ✗ sw.js not found or empty"
            fi
            # Check WASM MIME type
            MIME=$(curl -sI "http://localhost:${SPA_PORT}/$(ls dist/*.wasm 2>/dev/null | head -1 | xargs basename 2>/dev/null)" 2>/dev/null | grep -i content-type || echo "")
            if echo "$MIME" | grep -qi "wasm"; then
                echo "  ✓ WASM MIME type correct"
            else
                echo "  △ WASM MIME: $MIME (may need application/wasm)"
            fi
        else
            echo "  ✗ manifest.json not found"
        fi
    fi

    echo ""
    echo "  Performance Summary"
    echo "  ┌────────────────┬──────────┐"
    echo "  │ Cold load      │ ${LOAD_TIME}ms    │"
    echo "  │ Warm load      │ ${WARM_TIME}ms    │"
    echo "  │ Navigation     │ ${NAV_TIME}ms    │"
    echo "  │ Scroll (avg)   │ $((SCROLL_TIME / 5))ms     │"
    echo "  └────────────────┴──────────┘"
else
    echo ""
    echo "━━━ [6/6] Performance (skipped, use --perf) ━━━"
fi

# ── Summary ──
echo ""
echo "╔══════════════════════════════════════╗"
echo "║           Test Complete              ║"
echo "╠══════════════════════════════════════╣"
echo "║ Pages:   $PASS/$((PASS + FAIL)) passed"
echo "║ Shots:   $SCREENSHOT_DIR/"
if [ "$DO_BASELINE" = true ]; then
echo "║ Baseline saved to: $BASELINE_DIR/"
fi
echo "╚══════════════════════════════════════╝"

# Cleanup SPA server if we started it
if [ -n "$SPA_PID" ]; then
    kill "$SPA_PID" 2>/dev/null || true
fi

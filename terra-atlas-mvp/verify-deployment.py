#!/usr/bin/env python3
import sys
sys.path.insert(0, '/srv/luminous-dynamics/claude-webpilot')

try:
    from webpilot import WebPilot

    print("🚀 Starting WebPilot verification...")
    pilot = WebPilot()

    print("📡 Navigating to production site...")
    pilot.start("https://atlas.luminousdynamics.io")

    print("⏳ Waiting 5 seconds for page load...")
    import time
    time.sleep(5)

    print("📸 Taking screenshot...")
    pilot.screenshot("screenshots/webpilot-production.png")

    print("✅ Screenshot saved to screenshots/webpilot-production.png")
    print("🔍 Checking for verification markers in page source...")

    # Try to find verification markers
    page_source = pilot.browser.page_source if hasattr(pilot.browser, 'page_source') else "N/A"

    if "BRAND NEW EARTH GLOBE" in page_source:
        print("✅ VERIFICATION MARKER FOUND - New component deployed!")
    elif "red" in page_source.lower() and "marker" in page_source.lower():
        print("⚠️  Possible marker found, check screenshot")
    else:
        print("❌ No verification markers found - old component still deployed")

except ImportError as e:
    print(f"❌ WebPilot not available: {e}")
    print("💡 Falling back to simple curl check...")

    import subprocess
    result = subprocess.run(
        ['curl', '-s', 'https://atlas.luminousdynamics.io'],
        capture_output=True,
        text=True
    )

    if "BRAND NEW EARTH GLOBE" in result.stdout:
        print("✅ VERIFICATION MARKER FOUND in HTML!")
    else:
        print("❌ No verification markers in initial HTML (may load client-side)")
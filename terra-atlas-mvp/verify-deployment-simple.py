#!/usr/bin/env python3
"""
Simple deployment verification without browser dependencies.
Checks JavaScript bundles and page metadata.
"""
import requests
import re
from datetime import datetime

print("🔍 Checking Terra Atlas deployment...")
print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Fetch production page
url = "https://atlas.luminousdynamics.io"
print(f"📡 Fetching {url}...")

try:
    response = requests.get(url, timeout=10)
    print(f"✅ Status: {response.status_code}")
    print(f"📊 Content length: {len(response.text):,} bytes\n")

    # Check for Next.js build ID
    build_id_match = re.search(r'"buildId":"([^"]+)"', response.text)
    if build_id_match:
        build_id = build_id_match.group(1)
        print(f"🏗️  Next.js Build ID: {build_id}\n")

    # Look for component imports in the HTML/scripts
    print("🔍 Searching for globe components...")

    components_found = {
        'EarthGlobeCustom': 'EarthGlobeCustom' in response.text,
        'TerraGlobeThree': 'TerraGlobeThree' in response.text,
        'TerraGlobeBackground': 'TerraGlobeBackground' in response.text,
        'AnimatedGlobe': 'AnimatedGlobe' in response.text,
    }

    print("Component Detection Results:")
    for component, found in components_found.items():
        status = "✅ FOUND" if found else "❌ NOT FOUND"
        print(f"  {component}: {status}")

    # Check for verification markers (they load client-side but might be in bundle)
    print("\n🎯 Checking for verification markers...")
    markers = {
        'RED MARKER text': 'BRAND NEW EARTH GLOBE' in response.text,
        'red border style': 'border: \'10px solid red\'' in response.text or '10px solid red' in response.text,
        'bg-red-600 class': 'bg-red-600' in response.text,
    }

    for marker, found in markers.items():
        status = "✅ FOUND" if found else "⚠️  NOT FOUND (may load client-side)"
        print(f"  {marker}: {status}")

    # Check for old globe.gl mentions
    print("\n🔍 Checking for globe.gl (old library)...")
    old_globe = 'globe.gl' in response.text or 'react-globe' in response.text
    if old_globe:
        print("  ⚠️  globe.gl references still present")
    else:
        print("  ✅ No globe.gl references (good!)")

    # Final determination
    print("\n" + "="*50)
    print("📋 DEPLOYMENT STATUS:")
    print("="*50)

    if components_found['EarthGlobeCustom']:
        print("✅ NEW COMPONENT DETECTED: EarthGlobeCustom is in the build!")
        print("   → Deployment likely successful")
        print("   → Visual verification recommended at:")
        print(f"   → {url}")
    elif components_found['TerraGlobeThree']:
        print("⚠️  INTERMEDIATE COMPONENT: TerraGlobeThree detected")
        print("   → Older version may be cached")
    elif components_found['TerraGlobeBackground'] or components_found['AnimatedGlobe']:
        print("❌ OLD COMPONENTS DETECTED: Still showing original globe")
        print("   → Vercel may not have rebuilt yet")
        print("   → Check Vercel dashboard for deployment status")
    else:
        print("⚠️  NO GLOBE COMPONENTS DETECTED")
        print("   → Page structure may have changed")
        print("   → Check JavaScript bundles manually")

    print("\n💡 Next Steps:")
    print("   1. Open production site in browser for visual confirmation")
    print("   2. Look for red verification markers")
    print("   3. Check Vercel dashboard if still showing old version")

except requests.RequestException as e:
    print(f"❌ Error fetching site: {e}")
    print("   → Site may be temporarily down")
    print("   → Check again in a few minutes")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
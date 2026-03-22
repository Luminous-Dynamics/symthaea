#!/usr/bin/env python3
"""
Check Next.js JavaScript bundles for our components.
"""
import requests
import re
from urllib.parse import urljoin

print("🔍 Analyzing Next.js JavaScript bundles...\n")

base_url = "https://atlas.luminousdynamics.io"

# Fetch the main page to find bundle URLs
response = requests.get(base_url, timeout=10)
html = response.text

# Extract all script src URLs
script_urls = re.findall(r'<script[^>]+src="([^"]+)"', html)
print(f"📦 Found {len(script_urls)} script tags\n")

# Look for main app chunks and component chunks
interesting_scripts = [url for url in script_urls if '/_next/static/' in url]
print(f"🎯 Analyzing {len(interesting_scripts)} Next.js bundles:\n")

# Check each bundle for our component code
component_evidence = {
    'EarthGlobeCustom': False,
    'TerraGlobeThree': False,
    'TerraGlobeBackground': False,
    'AnimatedGlobe': False,
    'earth-blue-marble': False,  # Our texture file
    'BRAND NEW EARTH GLOBE': False,  # Verification marker
    '10px solid red': False,  # Verification border
}

for i, script_url in enumerate(interesting_scripts[:10], 1):  # Check first 10 bundles
    full_url = urljoin(base_url, script_url)
    filename = script_url.split('/')[-1][:40]

    try:
        print(f"{i}. Checking {filename}...")
        script_response = requests.get(full_url, timeout=10)
        script_content = script_response.text

        # Check for evidence of each component
        for key in component_evidence.keys():
            if key in script_content and not component_evidence[key]:
                component_evidence[key] = True
                print(f"   ✅ Found: {key}")

        print(f"   📊 Size: {len(script_content):,} bytes")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

print("\n" + "="*60)
print("📊 COMPONENT DETECTION SUMMARY:")
print("="*60)

for component, found in component_evidence.items():
    status = "✅ FOUND" if found else "❌ NOT FOUND"
    print(f"  {component}: {status}")

print("\n" + "="*60)
print("🎯 DEPLOYMENT VERDICT:")
print("="*60)

if component_evidence['EarthGlobeCustom'] or component_evidence['BRAND NEW EARTH GLOBE']:
    print("✅ NEW COMPONENT DEPLOYED!")
    print("   Your latest changes are live in production")
elif component_evidence['TerraGlobeThree']:
    print("⚠️  INTERMEDIATE VERSION DEPLOYED")
    print("   EarthGlobeCustom may still be deploying")
elif component_evidence['TerraGlobeBackground'] or component_evidence['AnimatedGlobe']:
    print("❌ OLD COMPONENTS STILL LIVE")
    print("   Vercel hasn't deployed the new version yet")
elif component_evidence['earth-blue-marble']:
    print("✅ CUSTOM TEXTURES FOUND")
    print("   New globe code is likely deployed (minified)")
else:
    print("⚠️  UNCLEAR STATUS")
    print("   Component names may be minified beyond recognition")

print("\n💡 RECOMMENDATION:")
print("   Open https://atlas.luminousdynamics.io in your browser")
print("   to visually verify the globe rendering.")
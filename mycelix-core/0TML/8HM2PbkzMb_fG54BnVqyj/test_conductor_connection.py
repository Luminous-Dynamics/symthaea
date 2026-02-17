#!/usr/bin/env python3
"""
Test if we can connect to a running Holochain conductor admin interface.
This validates the Python client and WebSocket connectivity.
"""

import asyncio
import sys

try:
    from holochain_client.api.admin.client import AdminClient
except ImportError:
    print("❌ holochain_client not installed")
    sys.exit(1)


async def test_connection():
    """Test connection to conductor admin interface"""
    print("🔍 Testing conductor connection...")
    print("📝 Note: This requires a running conductor on ws://localhost:8888")

    try:
        # Try to create client
        client = await AdminClient.create("ws://localhost:8888", defaultTimeout=5)
        print("✅ Successfully connected to conductor!")

        # Try to list installed apps
        apps = await client.list_apps()
        print(f"✅ Retrieved app list: {len(apps)} apps installed")

        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("This is expected if conductor is not running")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_connection())
    sys.exit(0 if result else 1)

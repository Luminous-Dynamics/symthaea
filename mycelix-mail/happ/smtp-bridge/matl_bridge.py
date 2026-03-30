#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Mail - MATL Trust Score Bridge
Syncs trust scores from the 0TML MATL system to Holochain DHT
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the 0TML directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "0TML" / "src"))

try:
    from holochain_client import HolochainClient
except ImportError:
    print("⚠️  holochain_client not installed. Install with: pip install holochain-client-python")
    sys.exit(1)


class MATLBridge:
    """
    Bridges MATL trust scores to Holochain

    Periodically queries the MATL system for trust scores and publishes
    them to the Holochain DHT where the trust_filter zome can access them.
    """

    def __init__(
        self,
        holochain_url: str = "ws://localhost:8888",
        matl_endpoint: str = "http://localhost:8080",
        sync_interval: int = 300  # 5 minutes
    ):
        self.holochain_url = holochain_url
        self.matl_endpoint = matl_endpoint
        self.sync_interval = sync_interval
        self.hc = None

    async def connect(self):
        """Connect to Holochain conductor and discover cell ID"""
        print(f"🔌 Connecting to Holochain at {self.holochain_url}...")

        try:
            self.hc = await HolochainClient.connect(self.holochain_url)
            print("✅ Connected to Holochain")

            # Discover and cache the cell ID
            cell_id = await self.discover_cell_id()
            if cell_id:
                print(f"✅ Cell ID discovered successfully")
            else:
                print("⚠️  Cell ID not discovered - some operations may fail")

        except Exception as e:
            print(f"❌ Failed to connect to Holochain: {e}")
            raise

    async def get_active_dids(self) -> list[str]:
        """
        Query Holochain for all DIDs that have sent or received messages

        Queries the DHT via the mail_messages zome for all unique sender/receiver DIDs.
        """
        try:
            # Query Holochain DHT for all active DIDs via mail_messages zome
            result = await self.hc.call_zome(
                cell_id=self.get_cell_id(),
                zome_name="mail_messages",
                fn_name="get_active_dids",
                payload={}
            )

            if result and isinstance(result, list):
                print(f"   Retrieved {len(result)} DIDs from DHT")
                return result

            # Fallback: Query trust_filter zome for known DIDs
            trust_result = await self.hc.call_zome(
                cell_id=self.get_cell_id(),
                zome_name="trust_filter",
                fn_name="get_all_trust_scores",
                payload={}
            )

            if trust_result and isinstance(trust_result, list):
                dids = [entry.get("did") for entry in trust_result if entry.get("did")]
                print(f"   Retrieved {len(dids)} DIDs from trust_filter")
                return dids

            return []

        except Exception as e:
            print(f"   Warning: Could not query DHT for DIDs: {e}")
            # Return empty list if DHT query fails - no mock data in production
            return []

    async def get_matl_trust_score(self, did: str) -> dict:
        """
        Query MATL system for a DID's trust score

        Calls the MATL HTTP API to get composite trust scores including
        POGQ, TCDM, and entropy metrics.
        """
        import aiohttp

        try:
            # Query MATL HTTP API for trust score
            async with aiohttp.ClientSession() as session:
                url = f"{self.matl_endpoint}/api/v1/trust/score/{did}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "did": did,
                            "score": data.get("composite_score", 0.5),
                            "pogq": data.get("pogq_score", 0.5),
                            "tcdm": data.get("tcdm_score", 0.5),
                            "entropy": data.get("entropy_score", 0.5),
                            "source": "matl_mode1"
                        }
                    elif response.status == 404:
                        # DID not found in MATL - return neutral score for new users
                        print(f"   DID {did} not found in MATL, assigning neutral score")
                        return {
                            "did": did,
                            "score": 0.5,
                            "pogq": 0.5,
                            "tcdm": 0.5,
                            "entropy": 0.5,
                            "source": "matl_mode1_default"
                        }
                    else:
                        print(f"   Warning: MATL API returned status {response.status} for {did}")

        except aiohttp.ClientError as e:
            print(f"   Warning: MATL API connection error for {did}: {e}")
        except asyncio.TimeoutError:
            print(f"   Warning: MATL API timeout for {did}")
        except Exception as e:
            print(f"   Warning: Unexpected error querying MATL for {did}: {e}")

        # Fallback: Return neutral score if MATL is unavailable
        return {
            "did": did,
            "score": 0.5,
            "pogq": 0.5,
            "tcdm": 0.5,
            "entropy": 0.5,
            "source": "matl_fallback"
        }

    async def update_holochain_trust_score(self, trust_data: dict):
        """
        Publish trust score to Holochain DHT via trust_filter zome
        """
        try:
            result = await self.hc.call_zome(
                cell_id=self.get_cell_id(),
                zome_name="trust_filter",
                fn_name="update_trust_score",
                payload={
                    "did": trust_data["did"],
                    "score": trust_data["score"],
                    "last_updated": int(time.time()),
                    "matl_source": trust_data["source"]
                }
            )
            print(f"  ✅ Updated {trust_data['did']}: {trust_data['score']:.2f}")
            return result
        except Exception as e:
            print(f"  ❌ Failed to update {trust_data['did']}: {e}")
            return None

    async def discover_cell_id(self):
        """
        Discover the cell ID for the mycelix-mail DNA from the conductor.

        Queries the Holochain conductor for installed apps and finds the
        cell ID for the mycelix_mail DNA.
        """
        try:
            # Get app info from the conductor
            app_info = await self.hc.app_info()

            if not app_info:
                print("   Warning: Could not get app info from conductor")
                return None

            # Look for mycelix_mail DNA in the installed cells
            for role_name, cell_info in app_info.cell_info.items():
                if "mail" in role_name.lower() or "mycelix" in role_name.lower():
                    # Extract cell_id from the first provisioned cell
                    if cell_info and hasattr(cell_info, 'cell_id'):
                        self._cell_id = cell_info.cell_id
                        print(f"   Discovered cell ID for role: {role_name}")
                        return self._cell_id

            # Fallback: use the first available cell
            for role_name, cell_info in app_info.cell_info.items():
                if cell_info and hasattr(cell_info, 'cell_id'):
                    self._cell_id = cell_info.cell_id
                    print(f"   Using first available cell ID for role: {role_name}")
                    return self._cell_id

            print("   Warning: No cells found in app info")
            return None

        except Exception as e:
            print(f"   Warning: Cell ID discovery failed: {e}")
            return None

    def get_cell_id(self):
        """
        Get the cached cell ID for the mycelix-mail DNA.

        Returns the previously discovered cell ID, or None if not yet discovered.
        Call discover_cell_id() during initialization to populate this.
        """
        return getattr(self, '_cell_id', None)

    async def sync_trust_scores(self):
        """
        Main sync loop - periodically updates all trust scores
        """
        print(f"🔄 Starting trust score sync (interval: {self.sync_interval}s)")

        while True:
            try:
                print("\n📊 Syncing trust scores...")

                # Get all active DIDs
                dids = await self.get_active_dids()
                print(f"   Found {len(dids)} active DIDs")

                # Update each DID's trust score
                for did in dids:
                    # Get trust score from MATL
                    trust_data = await self.get_matl_trust_score(did)

                    # Publish to Holochain
                    await self.update_holochain_trust_score(trust_data)

                print(f"✅ Sync complete. Sleeping for {self.sync_interval}s...")

            except Exception as e:
                print(f"❌ Error during sync: {e}")

            await asyncio.sleep(self.sync_interval)

    async def run(self):
        """Start the bridge service"""
        await self.connect()
        await self.sync_trust_scores()


async def main():
    """Main entry point"""
    print("🍄 Mycelix Mail - MATL Trust Bridge")
    print("=" * 50)

    bridge = MATLBridge(
        holochain_url="ws://localhost:8888",
        matl_endpoint="http://localhost:8080",
        sync_interval=300  # 5 minutes
    )

    try:
        await bridge.run()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

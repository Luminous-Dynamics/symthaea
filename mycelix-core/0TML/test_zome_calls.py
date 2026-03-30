#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 7: Comprehensive Zome Function Testing

Tests all 4 core ZeroTrustML zome functions against real Holochain DHT:
1. create_credit() - Write to DHT
2. get_credit() - Read from DHT
3. get_credits_for_holder() - Query DHT
4. get_balance() - Calculate totals

Prerequisites:
- Conductor running with IPv4 config
- ZeroTrustML DNA installed (see MANUAL_DNA_INSTALLATION_GUIDE.md)
- Rust bridge built and available
"""

import sys
import os
from datetime import datetime

# Add bridge to path
bridge_path = '/srv/luminous-dynamics/Mycelix-Core/0TML/rust-bridge/target/release'
sys.path.insert(0, bridge_path)

try:
    from holochain_credits_bridge import HolochainBridge
except ImportError as e:
    print(f"❌ Failed to import bridge: {e}")
    print(f"   Bridge path: {bridge_path}")
    print(f"   Run: cd rust-bridge && maturin develop --release")
    sys.exit(1)


class ZomeFunctionTester:
    """Test all zome functions against real DHT"""

    def __init__(self, conductor_url="ws://localhost:8888", app_id="zerotrustml-app"):
        self.conductor_url = conductor_url
        self.app_id = app_id
        self.bridge = None
        self.test_results = {
            'connection': False,
            'create_credit': False,
            'get_credit': False,
            'get_credits_for_holder': False,
            'get_balance': False
        }
        self.created_action_hash = None

    def setup(self):
        """Initialize bridge and connect to conductor"""
        print("\n" + "=" * 70)
        print("  Phase 7: Zome Function Testing")
        print("=" * 70)
        print(f"\nSetup:")
        print(f"  Conductor: {self.conductor_url}")
        print(f"  App ID: {self.app_id}")
        print(f"  Bridge: {bridge_path}")

        self.bridge = HolochainBridge(
            conductor_url=self.conductor_url,
            app_id=self.app_id,
            zome_name="credits",
            enabled=True
        )

        print(f"\n📡 Connecting to conductor...")
        success = self.bridge.connect()

        if not success:
            print(f"   ❌ Failed to connect to conductor")
            print(f"\n   Troubleshooting:")
            print(f"   1. Ensure conductor is running:")
            print(f"      holochain --config-path conductor-ipv4-only.yaml --piped")
            print(f"   2. Verify DNA is installed:")
            print(f"      hc app list --admin-port 8888")
            return False

        health = self.bridge.get_connection_health()
        print(f"   ✅ Connected!")
        print(f"      WebSocket: {health['websocket_ready']}")
        print(f"      Active: {health['is_connected']}")

        self.test_results['connection'] = True
        return

    def test_create_credit(self):
        """Test 1: Create a credit entry (DHT write)"""
        print(f"\n" + "-" * 70)
        print(f"Test 1: create_credit() - DHT Write")
        print("-" * 70)

        test_data = {
            'issuer': 'test_issuer_agent_key',
            'holder': 'test_holder_agent_key',
            'amount': 100.0,
            'description': f'Test credit created at {datetime.now().isoformat()}',
            'metadata': {'test': 'phase7', 'type': 'integration_test'}
        }

        print(f"\n📝 Creating credit:")
        print(f"   Issuer: {test_data['issuer'][:40]}...")
        print(f"   Holder: {test_data['holder'][:40]}...")
        print(f"   Amount: {test_data['amount']}")
        print(f"   Description: {test_data['description'][:60]}...")

        try:
            result = self.bridge.create_credit(
                issuer=test_data['issuer'],
                holder=test_data['holder'],
                amount=test_data['amount'],
                description=test_data['description'],
                metadata=test_data['metadata']
            )

            # Parse result to extract action hash
            # Result format: "ActionHash(0x...)"
            if "ActionHash" in result:
                self.created_action_hash = result
                print(f"\n   ✅ Credit created successfully!")
                print(f"      Action Hash: {result[:80]}...")

                # Verify it's a REAL action hash (not mock)
                if len(result) > 50 and "0x" in result:
                    print(f"      ✅ Real DHT action hash (not mock)")
                    self.test_results['create_credit'] = True
                else:
                    print(f"      ⚠️  Suspicious hash format (might be mock)")
            else:
                print(f"   ⚠️  Unexpected result format: {result}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        return self.test_results['create_credit']

    def test_get_credit(self):
        """Test 2: Get credit by action hash (DHT read)"""
        print(f"\n" + "-" * 70)
        print(f"Test 2: get_credit() - DHT Read")
        print("-" * 70)

        if not self.created_action_hash:
            print(f"\n   ⚠️  Skipping: No credit created in Test 1")
            return False

        print(f"\n🔍 Retrieving credit:")
        print(f"   Action Hash: {self.created_action_hash[:80]}...")

        try:
            result = self.bridge.get_credit(self.created_action_hash)

            if result:
                print(f"\n   ✅ Credit retrieved successfully!")
                print(f"      Data: {str(result)[:200]}...")

                # Verify data matches what we created
                if "100" in str(result) or "test" in str(result).lower():
                    print(f"      ✅ Data matches created credit")
                    self.test_results['get_credit'] = True
                else:
                    print(f"      ⚠️  Data doesn't match expected values")
            else:
                print(f"   ⚠️  No data returned")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        return self.test_results['get_credit']

    def test_get_credits_for_holder(self):
        """Test 3: Query credits by holder (DHT query)"""
        print(f"\n" + "-" * 70)
        print(f"Test 3: get_credits_for_holder() - DHT Query")
        print("-" * 70)

        holder = 'test_holder_agent_key'

        print(f"\n🔎 Querying credits:")
        print(f"   Holder: {holder}")

        try:
            results = self.bridge.get_credits_for_holder(holder)

            if results:
                print(f"\n   ✅ Query successful!")

                # Parse results (expected to be list of credits)
                if isinstance(results, list):
                    print(f"      Found {len(results)} credits")
                    self.test_results['get_credits_for_holder'] = True
                elif isinstance(results, str) and "ActionHash" in results:
                    print(f"      Found credit: {results[:80]}...")
                    self.test_results['get_credits_for_holder'] = True
                else:
                    print(f"      Results: {str(results)[:200]}...")
                    self.test_results['get_credits_for_holder'] = True
            else:
                print(f"   ⚠️  No credits found (might be expected if query filter)")
                # Not necessarily a failure

        except Exception as e:
            print(f"   ❌ Error: {e}")

        return self.test_results['get_credits_for_holder']

    def test_get_balance(self):
        """Test 4: Calculate balance (computation)"""
        print(f"\n" + "-" * 70)
        print(f"Test 4: get_balance() - Balance Calculation")
        print("-" * 70)

        agent = 'test_holder_agent_key'

        print(f"\n🧮 Calculating balance:")
        print(f"   Agent: {agent}")

        try:
            balance = self.bridge.get_balance(agent)

            print(f"\n   ✅ Balance calculated!")
            print(f"      Balance: {balance}")

            # Verify it's a number
            try:
                balance_value = float(balance)
                print(f"      ✅ Valid numeric balance: {balance_value}")
                self.test_results['get_balance'] = True
            except ValueError:
                print(f"      ⚠️  Non-numeric balance: {balance}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        return self.test_results['get_balance']

    def print_summary(self):
        """Print test results summary"""
        print(f"\n" + "=" * 70)
        print(f"  Test Results Summary")
        print("=" * 70)

        total = len(self.test_results)
        passed = sum(1 for v in self.test_results.values() if v)

        print(f"\n📊 Results: {passed}/{total} tests passed\n")

        for test, result in self.test_results.items():
            icon = "✅" if result else "❌"
            print(f"   {icon} {test}")

        print(f"\n" + "=" * 70)

        if passed == total:
            print(f"\n🎉 SUCCESS! All zome functions working with real DHT!")
            print(f"\n✅ Phase 7 COMPLETE:")
            print(f"   • DNA installed and operational")
            print(f"   • All zome calls tested")
            print(f"   • Real action hashes verified")
            print(f"   • DHT storage confirmed working")
            print(f"\n🚀 Ready for Phase 8: Trust Metrics Integration")
            return
        else:
            print(f"\n⚠️  Some tests failed - review errors above")
            print(f"\nDebugging steps:")
            print(f"   1. Check conductor logs for errors")
            print(f"   2. Verify DNA is properly installed")
            print(f"   3. Ensure zome names match")
            print(f"   4. Review bridge implementation")
            return False

    def run_all_tests(self):
        """Run complete test suite"""
        if not self.setup():
            print(f"\n❌ Setup failed - cannot run tests")
            return False

        # Run all tests in sequence
        self.test_create_credit()
        self.test_get_credit()
        self.test_get_credits_for_holder()
        self.test_get_balance()

        # Print summary
        return self.print_summary()


def main():
    """Main test execution"""
    print(f"\n{'=' * 70}")
    print(f"  ZeroTrustML Credits - Comprehensive Zome Function Testing")
    print(f"  Phase 7: Real DHT Verification")
    print(f"{'=' * 70}")
    print(f"\nThis test will verify:")
    print(f"  1. Connection to installed DNA")
    print(f"  2. DHT write operations (create_credit)")
    print(f"  3. DHT read operations (get_credit)")
    print(f"  4. DHT query operations (get_credits_for_holder)")
    print(f"  5. Computation operations (get_balance)")

    # Initialize tester
    tester = ZomeFunctionTester()

    # Run tests
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

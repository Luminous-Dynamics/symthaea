#!/usr/bin/env python3
"""
Mycelix Mail Integration Test Suite
====================================

Tests the complete L1→L5→L6 integration:
- Layer 1 (DHT): Holochain DNA operations
- Layer 5 (Identity): DID resolution
- Layer 6 (Trust): MATL score sync

This test suite validates end-to-end functionality without requiring
a running Holochain conductor (uses mock interfaces).
"""

import sys
import os
import unittest
import json
import tempfile
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestMailMessage:
    """Test mail message structure matching DNA entry type"""
    from_did: str
    to_did: str
    subject_encrypted: bytes
    body_cid: str
    timestamp: int
    thread_id: Optional[str]
    epistemic_tier: str


@dataclass
class TestTrustScore:
    """Test trust score structure matching DNA entry type"""
    did: str
    score: float
    last_updated: int
    matl_source: str


@dataclass
class RecordedSpamReport:
    """Represents a spam report stored in the DHT"""
    reporter_did: str
    spammer_did: str
    message_hash: str
    reason: str
    reported_at: int


class MockDIDRegistry:
    """Mock DID Registry for testing"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize test database schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS did_mapping (
                did TEXT PRIMARY KEY,
                agent_pub_key TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def register_did(self, did: str, agent_pub_key: str, metadata: Dict = None) -> bool:
        """Register a DID → AgentPubKey mapping"""
        try:
            now = int(datetime.now().timestamp())
            self.conn.execute("""
                INSERT INTO did_mapping (did, agent_pub_key, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (did, agent_pub_key, now, now, json.dumps(metadata or {})))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def resolve_did(self, did: str) -> Optional[str]:
        """Resolve DID to AgentPubKey"""
        cursor = self.conn.execute(
            "SELECT agent_pub_key FROM did_mapping WHERE did = ?",
            (did,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self):
        """Close database connection"""
        self.conn.close()


class MockMATLBridge:
    """Mock MATL Bridge for testing"""

    def __init__(self):
        self.scores: Dict[str, float] = {}

    def set_trust_score(self, did: str, score: float):
        """Set trust score for a DID"""
        if not 0.0 <= score <= 1.0:
            raise ValueError("Trust score must be between 0.0 and 1.0")
        self.scores[did] = score

    def get_trust_score(self, did: str) -> Optional[float]:
        """Get trust score for a DID"""
        return self.scores.get(did)

    def sync_to_holochain(self, did: str) -> Optional[TestTrustScore]:
        """Create TrustScore entry for Holochain"""
        score = self.get_trust_score(did)
        if score is None:
            return None

        return TestTrustScore(
            did=did,
            score=score,
            last_updated=int(datetime.now().timestamp()),
            matl_source="mock-matl-v1"
        )


class MockSpamReportStore:
    """Mock spam report storage representing DNA spam reports"""

    def __init__(self):
        self.reports: List[RecordedSpamReport] = []
        self._last_reported_at: int = 0

    def report_spam(self, reporter_did: str, spammer_did: str, message_hash: str, reason: str):
        if not reporter_did or not spammer_did:
            raise ValueError("Reporter and spammer DIDs are required")

        reported_at = int(datetime.now().timestamp())
        if reported_at <= self._last_reported_at:
            reported_at = self._last_reported_at + 1
        self._last_reported_at = reported_at

        report = RecordedSpamReport(
            reporter_did=reporter_did,
            spammer_did=spammer_did,
            message_hash=message_hash,
            reason=reason,
            reported_at=reported_at,
        )
        self.reports.append(report)
        return report

    def get_reports_since(self, since: int) -> List[RecordedSpamReport]:
        return [report for report in self.reports if report.reported_at > since]


class TestDIDResolution(unittest.TestCase):
    """Test Layer 5: DID Resolution"""

    def setUp(self):
        """Set up test database"""
        self.db_file = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.db_file.name
        self.db_file.close()
        self.registry = MockDIDRegistry(self.db_path)

    def tearDown(self):
        """Clean up test database"""
        self.registry.close()
        os.unlink(self.db_path)

    def test_register_did(self):
        """Test DID registration"""
        did = "did:mycelix:alice"
        agent_key = "uhCAkAbCdEf1234567890"

        result = self.registry.register_did(did, agent_key)
        self.assertTrue(result)

    def test_resolve_did(self):
        """Test DID resolution"""
        did = "did:mycelix:bob"
        agent_key = "uhCAkXyZ9876543210"

        self.registry.register_did(did, agent_key)
        resolved = self.registry.resolve_did(did)

        self.assertEqual(resolved, agent_key)

    def test_resolve_nonexistent_did(self):
        """Test resolving nonexistent DID"""
        resolved = self.registry.resolve_did("did:mycelix:nonexistent")
        self.assertIsNone(resolved)

    def test_duplicate_did_registration(self):
        """Test that duplicate DID registration fails"""
        did = "did:mycelix:charlie"
        agent_key1 = "uhCAk111111"
        agent_key2 = "uhCAk222222"

        result1 = self.registry.register_did(did, agent_key1)
        result2 = self.registry.register_did(did, agent_key2)

        self.assertTrue(result1)
        self.assertFalse(result2)

        # Original mapping should be preserved
        resolved = self.registry.resolve_did(did)
        self.assertEqual(resolved, agent_key1)


class TestMATLIntegration(unittest.TestCase):
    """Test Layer 6: MATL Trust Score Integration"""

    def setUp(self):
        """Set up MATL bridge"""
        self.matl = MockMATLBridge()

    def test_set_trust_score(self):
        """Test setting trust score"""
        did = "did:mycelix:trusted"
        score = 0.85

        self.matl.set_trust_score(did, score)
        result = self.matl.get_trust_score(did)

        self.assertEqual(result, score)

    def test_trust_score_bounds(self):
        """Test trust score bounds validation"""
        did = "did:mycelix:test"

        # Valid scores
        self.matl.set_trust_score(did, 0.0)
        self.matl.set_trust_score(did, 1.0)
        self.matl.set_trust_score(did, 0.5)

        # Invalid scores
        with self.assertRaises(ValueError):
            self.matl.set_trust_score(did, -0.1)

        with self.assertRaises(ValueError):
            self.matl.set_trust_score(did, 1.1)

    def test_sync_to_holochain(self):
        """Test MATL → Holochain sync"""
        did = "did:mycelix:sync-test"
        score = 0.75

        self.matl.set_trust_score(did, score)
        trust_entry = self.matl.sync_to_holochain(did)

        self.assertIsNotNone(trust_entry)
        self.assertEqual(trust_entry.did, did)
        self.assertEqual(trust_entry.score, score)
        self.assertEqual(trust_entry.matl_source, "mock-matl-v1")

    def test_sync_nonexistent_did(self):
        """Test syncing DID without trust score"""
        result = self.matl.sync_to_holochain("did:mycelix:nonexistent")
        self.assertIsNone(result)


class TestEndToEndIntegration(unittest.TestCase):
    """Test complete L1→L5→L6 integration"""

    def setUp(self):
        """Set up complete test environment"""
        self.db_file = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.db_file.name
        self.db_file.close()

        self.registry = MockDIDRegistry(self.db_path)
        self.matl = MockMATLBridge()
        self.spam_reports = MockSpamReportStore()

    def tearDown(self):
        """Clean up test environment"""
        self.registry.close()
        os.unlink(self.db_path)

    def test_full_message_flow(self):
        """Test complete message sending flow"""
        # Setup: Register DIDs
        alice_did = "did:mycelix:alice"
        alice_key = "uhCAkAlice123"
        bob_did = "did:mycelix:bob"
        bob_key = "uhCAkBob456"

        self.registry.register_did(alice_did, alice_key)
        self.registry.register_did(bob_did, bob_key)

        # Setup: Set trust scores
        self.matl.set_trust_score(alice_did, 0.95)  # High trust
        self.matl.set_trust_score(bob_did, 0.80)    # Good trust

        # Step 1: Resolve DIDs
        alice_resolved = self.registry.resolve_did(alice_did)
        bob_resolved = self.registry.resolve_did(bob_did)

        self.assertEqual(alice_resolved, alice_key)
        self.assertEqual(bob_resolved, bob_key)

        # Step 2: Check trust score (spam filtering)
        alice_score = self.matl.get_trust_score(alice_did)
        self.assertIsNotNone(alice_score)
        self.assertGreater(alice_score, 0.5)  # Above spam threshold

        # Step 3: Create message (would be sent to Holochain)
        message = TestMailMessage(
            from_did=alice_did,
            to_did=bob_did,
            subject_encrypted=b"encrypted_subject_data",
            body_cid="QmAbCdEf123",  # IPFS CID
            timestamp=int(datetime.now().timestamp()),
            thread_id=None,
            epistemic_tier="Tier2PrivatelyVerifiable"
        )

        # Verify message structure
        self.assertEqual(message.from_did, alice_did)
        self.assertEqual(message.to_did, bob_did)
        self.assertIn("Tier", message.epistemic_tier)

    def test_spam_filtering(self):
        """Test spam filtering based on trust scores"""
        spammer_did = "did:mycelix:spammer"
        spammer_key = "uhCAkSpammer999"
        recipient_did = "did:mycelix:recipient"

        # Register spammer with low trust
        self.registry.register_did(spammer_did, spammer_key)
        self.matl.set_trust_score(spammer_did, 0.2)  # Low trust

        # Check if message should be filtered
        trust_score = self.matl.get_trust_score(spammer_did)
        spam_threshold = 0.3

        should_filter = trust_score < spam_threshold
        self.assertTrue(should_filter)

    def test_trust_score_sync(self):
        """Test complete trust score synchronization"""
        test_did = "did:mycelix:trusttest"
        test_key = "uhCAkTest789"

        # Register DID
        self.registry.register_did(test_did, test_key)

        # Set and sync trust score
        self.matl.set_trust_score(test_did, 0.88)
        trust_entry = self.matl.sync_to_holochain(test_did)

        # Verify complete entry
        self.assertIsNotNone(trust_entry)
        self.assertEqual(trust_entry.did, test_did)
        self.assertEqual(trust_entry.score, 0.88)
        self.assertGreater(trust_entry.last_updated, 0)

    def test_spam_report_storage(self):
        """Test spam reports can be stored and queried"""
        reporter = "did:mycelix:reporter"
        spammer = "did:mycelix:spammer"
        message_hash = "hash123"
        reason = "phishing"

        report = self.spam_reports.report_spam(reporter, spammer, message_hash, reason)
        self.assertEqual(report.spammer_did, spammer)
        self.assertEqual(report.reason, reason)

        results = self.spam_reports.get_reports_since(report.reported_at - 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].message_hash, message_hash)

        older_results = self.spam_reports.get_reports_since(report.reported_at + 1)
        self.assertEqual(len(older_results), 0)


class TestSpamReporting(unittest.TestCase):
    """Targeted tests for spam-report data validation"""

    def setUp(self):
        self.store = MockSpamReportStore()

    def test_missing_spammer_did(self):
        with self.assertRaises(ValueError):
            self.store.report_spam("did:reporter", "", "hash", "reason")

    def test_report_timestamp_ordering(self):
        first = self.store.report_spam("did:reporter", "did:spammer", "hash1", "malware")
        second = self.store.report_spam("did:reporter", "did:spammer", "hash2", "phishing")

        self.assertLess(first.reported_at, second.reported_at)
        recent_reports = self.store.get_reports_since(first.reported_at)
        self.assertEqual(len(recent_reports), 1)
        self.assertEqual(recent_reports[0].message_hash, "hash2")


class TestEpistemicTiers(unittest.TestCase):
    """Test Epistemic Charter v2.0 tier handling"""

    def test_tier_values(self):
        """Test valid epistemic tier values"""
        valid_tiers = [
            "Tier0Null",
            "Tier1Testimonial",
            "Tier2PrivatelyVerifiable",
            "Tier3CryptographicallyProven",
            "Tier4PubliclyReproducible"
        ]

        for tier in valid_tiers:
            message = TestMailMessage(
                from_did="did:mycelix:sender",
                to_did="did:mycelix:receiver",
                subject_encrypted=b"test",
                body_cid="QmTest",
                timestamp=int(datetime.now().timestamp()),
                thread_id=None,
                epistemic_tier=tier
            )
            self.assertIn("Tier", message.epistemic_tier)

    def test_tier_progression(self):
        """Test epistemic tier progression makes sense"""
        tiers_by_strength = [
            ("Tier0Null", 0),
            ("Tier1Testimonial", 1),
            ("Tier2PrivatelyVerifiable", 2),
            ("Tier3CryptographicallyProven", 3),
            ("Tier4PubliclyReproducible", 4)
        ]

        # Verify tiers are ordered by strength
        for i in range(len(tiers_by_strength) - 1):
            tier1_name, tier1_level = tiers_by_strength[i]
            tier2_name, tier2_level = tiers_by_strength[i + 1]

            self.assertLess(tier1_level, tier2_level)


def run_test_suite():
    """Run the complete integration test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDIDResolution))
    suite.addTests(loader.loadTestsFromTestCase(TestMATLIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSpamReporting))
    suite.addTests(loader.loadTestsFromTestCase(TestEpistemicTiers))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("Integration Test Suite Summary")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)

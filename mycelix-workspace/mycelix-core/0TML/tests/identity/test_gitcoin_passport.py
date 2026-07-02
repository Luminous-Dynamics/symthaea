# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Comprehensive tests for Gitcoin Passport Integration.

Tests the GitcoinPassportClient, caching, retry logic, and
integration with the Zero-TrustML identity system.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import pytest
import aiohttp

from zerotrustml.identity.gitcoin_passport import (
    GitcoinPassportClient,
    GitcoinPassportVerifier,
    PassportScore,
    Stamp,
    StampProvider,
    GovernanceVerificationResult,
    PassportError,
    RateLimitError,
    AuthenticationError,
    AddressNotFoundError,
    RetryExhaustedError,
    RetryConfig,
    PassportCache,
    CacheEntry,
    verify_humanity,
    get_passport_score,
)
from zerotrustml.identity.factors import (
    GitcoinPassportFactor,
    FactorStatus,
    FactorCategory,
)


# Test fixtures and mock data
MOCK_API_KEY = "test-api-key-12345"
MOCK_SCORER_ID = "test-scorer-123"
MOCK_ADDRESS = "0x1234567890123456789012345678901234567890"
MOCK_ADDRESS_NORMALIZED = MOCK_ADDRESS.lower()


def create_mock_score_response(
    score: float = 42.5,
    passing: bool = True,
    stamps: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Create a mock API response for score endpoint."""
    if stamps is None:
        stamps = {
            "Google": {
                "score": 5.0,
                "hash": "abc123",
                "dedup": False,
                "expiration_date": "2025-12-31T00:00:00Z",
            },
            "Github": {
                "score": 10.0,
                "hash": "def456",
                "dedup": False,
                "expiration_date": "2025-12-31T00:00:00Z",
            },
            "Twitter": {
                "score": 7.5,
                "hash": "ghi789",
                "dedup": True,  # Duplicate stamp
                "expiration_date": "2025-12-31T00:00:00Z",
            },
        }
    return {
        "score": score,
        "passing_score": passing,
        "stamps": stamps,
        "last_score_timestamp": "2024-01-15T10:00:00Z",
        "expiration_timestamp": "2024-04-15T10:00:00Z",
    }


def create_mock_stamps_response(num_stamps: int = 3) -> Dict[str, Any]:
    """Create a mock API response for stamps endpoint."""
    items = []
    providers = ["Google", "Github", "Twitter", "Discord", "ENS"]
    for i in range(min(num_stamps, len(providers))):
        items.append({
            "id": f"stamp-{i}",
            "credential": {
                "credentialSubject": {
                    "provider": providers[i],
                    "hash": f"hash-{i}",
                },
                "issuer": "did:key:passport-issuer",
                "issuanceDate": "2024-01-01T00:00:00Z",
                "expirationDate": "2025-12-31T00:00:00Z",
            },
        })
    return {"items": items}


class TestStamp:
    """Tests for the Stamp dataclass."""

    def test_stamp_creation(self):
        """Test basic stamp creation."""
        stamp = Stamp(
            provider="Google",
            credential_hash="abc123",
            score=5.0,
        )
        assert stamp.provider == "Google"
        assert stamp.credential_hash == "abc123"
        assert stamp.score == 5.0
        assert not stamp.is_duplicate
        assert stamp.expires_at is None

    def test_stamp_is_expired(self):
        """Test stamp expiration check."""
        # Not expired
        future = datetime.now(timezone.utc) + timedelta(days=30)
        stamp = Stamp(
            provider="Github",
            credential_hash="def456",
            expires_at=future,
        )
        assert not stamp.is_expired

        # Expired
        past = datetime.now(timezone.utc) - timedelta(days=1)
        stamp_expired = Stamp(
            provider="Github",
            credential_hash="def456",
            expires_at=past,
        )
        assert stamp_expired.is_expired

    def test_stamp_is_valid(self):
        """Test stamp validity (not expired and not duplicate)."""
        future = datetime.now(timezone.utc) + timedelta(days=30)

        # Valid stamp
        valid_stamp = Stamp(
            provider="Google",
            credential_hash="abc123",
            expires_at=future,
            is_duplicate=False,
        )
        assert valid_stamp.is_valid

        # Duplicate stamp (invalid)
        duplicate_stamp = Stamp(
            provider="Google",
            credential_hash="abc123",
            expires_at=future,
            is_duplicate=True,
        )
        assert not duplicate_stamp.is_valid

        # Expired stamp (invalid)
        past = datetime.now(timezone.utc) - timedelta(days=1)
        expired_stamp = Stamp(
            provider="Google",
            credential_hash="abc123",
            expires_at=past,
            is_duplicate=False,
        )
        assert not expired_stamp.is_valid

    def test_stamp_to_dict(self):
        """Test stamp serialization."""
        stamp = Stamp(
            provider="Github",
            credential_hash="def456",
            score=10.0,
            metadata={"test": "data"},
        )
        data = stamp.to_dict()

        assert data["provider"] == "Github"
        assert data["credential_hash"] == "def456"
        assert data["score"] == 10.0
        assert data["metadata"] == {"test": "data"}


class TestPassportScore:
    """Tests for the PassportScore dataclass."""

    def test_passport_score_creation(self):
        """Test basic PassportScore creation."""
        score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=42.5,
            passing=True,
            threshold=20.0,
        )
        assert score.address == MOCK_ADDRESS_NORMALIZED
        assert score.score == 42.5
        assert score.passing
        assert score.threshold == 20.0
        assert score.stamp_count == 0

    def test_passport_score_stamp_count(self):
        """Test stamp count calculation (only valid stamps)."""
        future = datetime.now(timezone.utc) + timedelta(days=30)
        stamps = [
            Stamp(provider="Google", credential_hash="a", is_duplicate=False, expires_at=future),
            Stamp(provider="Github", credential_hash="b", is_duplicate=False, expires_at=future),
            Stamp(provider="Twitter", credential_hash="c", is_duplicate=True, expires_at=future),  # Invalid
        ]

        score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=50.0,
            passing=True,
            threshold=20.0,
            stamps=stamps,
        )

        assert score.stamp_count == 2  # Only non-duplicate stamps

    def test_passport_score_get_stamp(self):
        """Test getting a specific stamp by provider."""
        stamps = [
            Stamp(provider="Google", credential_hash="a"),
            Stamp(provider="Github", credential_hash="b"),
        ]

        score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=50.0,
            passing=True,
            threshold=20.0,
            stamps=stamps,
        )

        google_stamp = score.get_stamp("Google")
        assert google_stamp is not None
        assert google_stamp.provider == "Google"

        # Case insensitive
        github_stamp = score.get_stamp("github")
        assert github_stamp is not None

        # Not found
        assert score.get_stamp("NonExistent") is None

    def test_passport_score_has_stamp(self):
        """Test checking for stamp presence."""
        future = datetime.now(timezone.utc) + timedelta(days=30)
        stamps = [
            Stamp(provider="Google", credential_hash="a", expires_at=future),
            Stamp(provider="Github", credential_hash="b", is_duplicate=True, expires_at=future),
        ]

        score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=50.0,
            passing=True,
            threshold=20.0,
            stamps=stamps,
        )

        # Valid stamp present
        assert score.has_stamp("Google")
        assert score.has_stamp(StampProvider.GOOGLE)

        # Duplicate stamp (not valid)
        assert not score.has_stamp("Github")

        # Not present
        assert not score.has_stamp("Twitter")

    def test_passport_score_is_expired(self):
        """Test score expiration check."""
        # Not expired
        future = datetime.now(timezone.utc) + timedelta(days=30)
        score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=50.0,
            passing=True,
            threshold=20.0,
            expiration=future,
        )
        assert not score.is_expired

        # Expired
        past = datetime.now(timezone.utc) - timedelta(days=1)
        score_expired = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=50.0,
            passing=True,
            threshold=20.0,
            expiration=past,
        )
        assert score_expired.is_expired

    def test_passport_score_to_dict(self):
        """Test PassportScore serialization."""
        score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=42.5,
            passing=True,
            threshold=20.0,
        )
        data = score.to_dict()

        assert data["address"] == MOCK_ADDRESS_NORMALIZED
        assert data["score"] == 42.5
        assert data["passing"]
        assert data["threshold"] == 20.0


class TestRetryConfig:
    """Tests for the RetryConfig class."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter

    def test_calculate_delay_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=100.0,
            jitter=False,
        )

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_calculate_delay_max_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=5.0,
            jitter=False,
        )

        # 2^10 = 1024, but should be capped at 5
        assert config.calculate_delay(10) == 5.0

    def test_calculate_delay_with_jitter(self):
        """Test jitter adds randomness."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=100.0,
            jitter=True,
        )

        # With jitter, values should vary
        delays = [config.calculate_delay(0) for _ in range(10)]
        # Not all values should be the same
        assert len(set(delays)) > 1


class TestCacheEntry:
    """Tests for the CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(data={"test": "data"}, ttl_seconds=300)
        assert entry.data == {"test": "data"}
        assert entry.ttl_seconds == 300
        assert entry.access_count == 0

    def test_cache_entry_is_expired(self):
        """Test cache entry expiration."""
        # Not expired
        entry = CacheEntry(data="test", ttl_seconds=3600)
        assert not entry.is_expired

        # Expired (create entry with old timestamp)
        old_entry = CacheEntry(data="test", ttl_seconds=1)
        old_entry.created_at = time.time() - 10  # 10 seconds ago
        assert old_entry.is_expired

    def test_cache_entry_touch(self):
        """Test updating access statistics."""
        entry = CacheEntry(data="test")
        initial_access_count = entry.access_count
        initial_last_accessed = entry.last_accessed

        time.sleep(0.01)  # Small delay to ensure time difference
        entry.touch()

        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed >= initial_last_accessed


class TestPassportCache:
    """Tests for the PassportCache class."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache for each test."""
        return PassportCache(max_size=10, default_ttl=300)

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("0x123", {"score": 42.5})
        result = await cache.get("0x123")
        assert result == {"score": 42.5}

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_address_normalization(self, cache):
        """Test addresses are normalized to lowercase."""
        await cache.set("0xABCDEF", {"score": 50.0})

        # Should find with lowercase
        result = await cache.get("0xabcdef")
        assert result == {"score": 50.0}

        # Should find with uppercase
        result = await cache.get("0xABCDEF")
        assert result == {"score": 50.0}

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache):
        """Test expired entries are not returned."""
        # Create cache with very short TTL
        short_cache = PassportCache(default_ttl=0)
        await short_cache.set("0x123", {"score": 42.5})

        # Force expiration by manipulating the entry
        key = short_cache._normalize_key("0x123")
        short_cache._cache[key].created_at = time.time() - 100

        # Should not return expired entry
        result = await short_cache.get("0x123")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_invalidate(self, cache):
        """Test invalidating specific entries."""
        await cache.set("0x123", {"score": 42.5})
        await cache.set("0x456", {"score": 50.0})

        result = await cache.invalidate("0x123")
        assert result is True

        # 0x123 should be gone
        assert await cache.get("0x123") is None
        # 0x456 should still exist
        assert await cache.get("0x456") is not None

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache):
        """Test clearing all entries."""
        await cache.set("0x123", {"score": 42.5})
        await cache.set("0x456", {"score": 50.0})

        count = await cache.clear()
        assert count == 2
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        small_cache = PassportCache(max_size=2, default_ttl=300)

        await small_cache.set("0x001", {"score": 1})
        await small_cache.set("0x002", {"score": 2})

        # Access 0x001 to make it more recently used
        await small_cache.get("0x001")

        # Adding a third entry should evict 0x002 (LRU)
        await small_cache.set("0x003", {"score": 3})

        assert await small_cache.get("0x001") is not None
        assert await small_cache.get("0x002") is None
        assert await small_cache.get("0x003") is not None

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache):
        """Test cache statistics."""
        await cache.set("0x123", {"score": 42.5})

        # Hit
        await cache.get("0x123")
        await cache.get("0x123")

        # Miss
        await cache.get("nonexistent")

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == pytest.approx(2/3)


class TestGitcoinPassportClient:
    """Tests for the GitcoinPassportClient class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.closed = False
        return session

    @pytest.fixture
    def client(self, mock_session):
        """Create a client with mocked session."""
        with patch.dict('os.environ', {
            'GITCOIN_PASSPORT_API_KEY': MOCK_API_KEY,
            'GITCOIN_PASSPORT_SCORER_ID': MOCK_SCORER_ID,
        }):
            client = GitcoinPassportClient(
                api_key=MOCK_API_KEY,
                scorer_id=MOCK_SCORER_ID,
                enable_cache=True,
            )
            client._session = mock_session
            return client

    def test_client_initialization(self):
        """Test client initialization with parameters."""
        client = GitcoinPassportClient(
            api_key="test-key",
            scorer_id="test-scorer",
            timeout=30,
            cache_ttl=600,
            cache_max_size=500,
        )
        assert client.api_key == "test-key"
        assert client.scorer_id == "test-scorer"
        assert client.timeout == 30

    def test_client_initialization_from_env(self):
        """Test client initialization from environment variables."""
        with patch.dict('os.environ', {
            'GITCOIN_PASSPORT_API_KEY': 'env-api-key',
            'GITCOIN_PASSPORT_SCORER_ID': 'env-scorer-id',
        }):
            client = GitcoinPassportClient()
            assert client.api_key == 'env-api-key'
            assert client.scorer_id == 'env-scorer-id'

    def test_client_requires_api_key(self):
        """Test client raises error without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                GitcoinPassportClient()

    def test_build_url(self, client):
        """Test URL building."""
        url = client._build_url("stamps", "scorer-123", "score", "0x123")
        assert url == "https://api.passport.xyz/v2/stamps/scorer-123/score/0x123"

    @pytest.mark.asyncio
    async def test_verify_passport(self, client, mock_session):
        """Test verify_passport returns PassportScore."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"score": 42.5}')
        mock_response.json = AsyncMock(return_value=create_mock_score_response())

        mock_session.request.return_value.__aenter__.return_value = mock_response

        score = await client.verify_passport(MOCK_ADDRESS)

        assert isinstance(score, PassportScore)
        assert score.score == 42.5
        assert score.passing
        assert len(score.stamps) == 3

    @pytest.mark.asyncio
    async def test_verify_passport_caches_result(self, client, mock_session):
        """Test verify_passport caches results."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"score": 42.5}')
        mock_response.json = AsyncMock(return_value=create_mock_score_response())

        mock_session.request.return_value.__aenter__.return_value = mock_response

        # First call
        score1 = await client.verify_passport(MOCK_ADDRESS)

        # Second call should use cache
        score2 = await client.verify_passport(MOCK_ADDRESS)

        # API should only be called once
        assert mock_session.request.call_count == 1
        assert score1.score == score2.score

    @pytest.mark.asyncio
    async def test_verify_passport_bypass_cache(self, client, mock_session):
        """Test verify_passport can bypass cache."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"score": 42.5}')
        mock_response.json = AsyncMock(return_value=create_mock_score_response())

        mock_session.request.return_value.__aenter__.return_value = mock_response

        # First call
        await client.verify_passport(MOCK_ADDRESS)

        # Second call without cache
        await client.verify_passport(MOCK_ADDRESS, use_cache=False)

        # API should be called twice
        assert mock_session.request.call_count == 2

    @pytest.mark.asyncio
    async def test_get_stamps(self, client, mock_session):
        """Test get_stamps returns list of stamps."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{}')
        mock_response.json = AsyncMock(return_value=create_mock_stamps_response(3))

        mock_session.request.return_value.__aenter__.return_value = mock_response

        stamps = await client.get_stamps(MOCK_ADDRESS)

        assert isinstance(stamps, list)
        assert len(stamps) == 3
        assert all(isinstance(s, Stamp) for s in stamps)

    @pytest.mark.asyncio
    async def test_check_humanity_threshold_passes(self, client, mock_session):
        """Test check_humanity_threshold returns True for passing score."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"score": 42.5}')
        mock_response.json = AsyncMock(return_value=create_mock_score_response(score=42.5))

        mock_session.request.return_value.__aenter__.return_value = mock_response

        is_human = await client.check_humanity_threshold(MOCK_ADDRESS, min_score=20.0)
        assert is_human is True

    @pytest.mark.asyncio
    async def test_check_humanity_threshold_fails_low_score(self, client, mock_session):
        """Test check_humanity_threshold returns False for low score."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"score": 10.0}')
        mock_response.json = AsyncMock(return_value=create_mock_score_response(score=10.0))

        mock_session.request.return_value.__aenter__.return_value = mock_response

        is_human = await client.check_humanity_threshold(MOCK_ADDRESS, min_score=20.0)
        assert is_human is False

    @pytest.mark.asyncio
    async def test_check_humanity_threshold_with_required_stamps(self, client, mock_session):
        """Test check_humanity_threshold with required stamps."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"score": 42.5}')
        mock_response.json = AsyncMock(return_value=create_mock_score_response())

        mock_session.request.return_value.__aenter__.return_value = mock_response

        # Should pass - has Google stamp
        is_human = await client.check_humanity_threshold(
            MOCK_ADDRESS,
            min_score=20.0,
            required_stamps=[StampProvider.GOOGLE],
        )
        assert is_human is True

    @pytest.mark.asyncio
    async def test_check_humanity_threshold_missing_stamp(self, client, mock_session):
        """Test check_humanity_threshold fails with missing required stamp."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"score": 42.5}')
        mock_response.json = AsyncMock(return_value=create_mock_score_response())

        mock_session.request.return_value.__aenter__.return_value = mock_response

        # Should fail - doesn't have BRIGHTID stamp
        is_human = await client.check_humanity_threshold(
            MOCK_ADDRESS,
            min_score=20.0,
            required_stamps=[StampProvider.BRIGHTID],
        )
        assert is_human is False

    @pytest.mark.asyncio
    async def test_check_humanity_threshold_address_not_found(self, client, mock_session):
        """Test check_humanity_threshold returns False for unknown address."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value='{"error": "not found"}')

        mock_session.request.return_value.__aenter__.return_value = mock_response

        is_human = await client.check_humanity_threshold(MOCK_ADDRESS, min_score=20.0)
        assert is_human is False

    @pytest.mark.asyncio
    async def test_authentication_error(self, client, mock_session):
        """Test authentication error handling."""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value='{"error": "invalid api key"}')

        mock_session.request.return_value.__aenter__.return_value = mock_response

        with pytest.raises(AuthenticationError):
            await client.verify_passport(MOCK_ADDRESS)

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, client, mock_session):
        """Test rate limit error handling."""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value='{"error": "rate limited"}')

        mock_session.request.return_value.__aenter__.return_value = mock_response

        # Disable retries for this test
        client.retry_config = RetryConfig(max_retries=0)

        with pytest.raises(RetryExhaustedError):
            await client.verify_passport(MOCK_ADDRESS)

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, client, mock_session):
        """Test retry logic on rate limit."""
        # First call returns 429, second returns success
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.text = AsyncMock(return_value='{"error": "rate limited"}')

        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.text = AsyncMock(return_value='{"score": 42.5}')
        mock_response_200.json = AsyncMock(return_value=create_mock_score_response())

        # Configure retry with no delay for testing
        client.retry_config = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_response_429
            return mock_response_200

        mock_session.request.return_value.__aenter__ = mock_request

        # Should succeed after retry
        score = await client.verify_passport(MOCK_ADDRESS, use_cache=False)
        assert score.score == 42.5
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_stats_property(self, client):
        """Test cache_stats property."""
        stats = client.cache_stats
        assert stats["enabled"] is True
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, client, mock_session):
        """Test cache invalidation."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value='{"score": 42.5}')
        mock_response.json = AsyncMock(return_value=create_mock_score_response())

        mock_session.request.return_value.__aenter__.return_value = mock_response

        # Populate cache
        await client.verify_passport(MOCK_ADDRESS)

        # Invalidate
        result = await client.invalidate_cache(MOCK_ADDRESS)
        assert result is True

        # Should fetch again
        await client.verify_passport(MOCK_ADDRESS)
        assert mock_session.request.call_count == 2

    @pytest.mark.asyncio
    async def test_create_passport_factor(self, client):
        """Test creating a GitcoinPassportFactor from client data."""
        stamps = [
            Stamp(provider="Google", credential_hash="abc", score=5.0),
            Stamp(provider="Github", credential_hash="def", score=10.0),
        ]

        factor = client.create_passport_factor(
            address=MOCK_ADDRESS,
            score=42.5,
            stamps=stamps,
        )

        assert isinstance(factor, GitcoinPassportFactor)
        assert factor.passport_address == MOCK_ADDRESS
        assert factor.score == 42.5
        assert factor.status == FactorStatus.ACTIVE
        assert "Google" in factor.stamps
        assert "Github" in factor.stamps


class TestGitcoinPassportVerifier:
    """Tests for the GitcoinPassportVerifier class."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier with mocked client."""
        verifier = GitcoinPassportVerifier(
            api_key=MOCK_API_KEY,
            scorer_id=MOCK_SCORER_ID,
            cache_ttl_seconds=300,
        )
        return verifier

    @pytest.mark.asyncio
    async def test_verify_for_governance_passes(self, verifier):
        """Test governance verification for passing score."""
        mock_score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=35.0,
            passing=True,
            threshold=20.0,
        )

        with patch.object(verifier.client, 'get_score', return_value=mock_score):
            result = await verifier.verify_for_governance(MOCK_ADDRESS)

        assert isinstance(result, GovernanceVerificationResult)
        assert result.passes_e2 is True
        assert result.passes_e3 is False  # Score < 50
        assert result.assurance_level == "E2_PrivatelyVerifiable"

    @pytest.mark.asyncio
    async def test_verify_for_governance_e3(self, verifier):
        """Test governance verification for E3 level."""
        mock_score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=60.0,
            passing=True,
            threshold=20.0,
        )

        with patch.object(verifier.client, 'get_score', return_value=mock_score):
            result = await verifier.verify_for_governance(MOCK_ADDRESS)

        assert result.passes_e2 is True
        assert result.passes_e3 is True
        assert result.assurance_level == "E3_CryptographicallyProven"
        assert result.vote_weight_multiplier == 1.0

    @pytest.mark.asyncio
    async def test_verify_for_governance_fails(self, verifier):
        """Test governance verification for failing score."""
        mock_score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=10.0,
            passing=False,
            threshold=20.0,
        )

        with patch.object(verifier.client, 'get_score', return_value=mock_score):
            result = await verifier.verify_for_governance(MOCK_ADDRESS)

        assert result.passes_e2 is False
        assert result.passes_e3 is False
        assert result.assurance_level == "E1_Testimonial"

    @pytest.mark.asyncio
    async def test_verify_for_governance_address_not_found(self, verifier):
        """Test governance verification for unknown address."""
        with patch.object(
            verifier.client,
            'get_score',
            side_effect=AddressNotFoundError("not found"),
        ):
            result = await verifier.verify_for_governance(MOCK_ADDRESS)

        assert result.passes_e2 is False
        assert result.passes_e3 is False
        assert result.error == "Address not found in Passport registry"

    @pytest.mark.asyncio
    async def test_verify_for_governance_caching(self, verifier):
        """Test governance verification caches results."""
        mock_score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=35.0,
            passing=True,
            threshold=20.0,
        )

        with patch.object(verifier.client, 'get_score', return_value=mock_score) as mock_get:
            # First call
            result1 = await verifier.verify_for_governance(MOCK_ADDRESS)

            # Second call should use cache
            result2 = await verifier.verify_for_governance(MOCK_ADDRESS)

            # get_score should only be called once
            assert mock_get.call_count == 1
            assert result1.from_cache is False
            assert result2.from_cache is True

    def test_governance_result_to_dict(self):
        """Test GovernanceVerificationResult serialization."""
        score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=35.0,
            passing=True,
            threshold=20.0,
        )

        result = GovernanceVerificationResult(
            address=MOCK_ADDRESS,
            score=score,
            passes_e2=True,
            passes_e3=False,
        )

        data = result.to_dict()
        assert data["address"] == MOCK_ADDRESS
        assert data["passes_e2"] is True
        assert data["passes_e3"] is False
        assert data["assurance_level"] == "E2_PrivatelyVerifiable"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_verify_humanity_function(self):
        """Test the verify_humanity convenience function."""
        mock_score = PassportScore(
            address=MOCK_ADDRESS_NORMALIZED,
            score=35.0,
            passing=True,
            threshold=20.0,
        )

        with patch.object(
            GitcoinPassportClient,
            'verify_humanity',
            return_value=True,
        ):
            with patch.object(
                GitcoinPassportClient,
                '__aenter__',
                return_value=MagicMock(verify_humanity=AsyncMock(return_value=True)),
            ):
                with patch.object(GitcoinPassportClient, '__aexit__', return_value=None):
                    # The function creates its own client, so we patch the class
                    result = await verify_humanity(
                        MOCK_ADDRESS,
                        threshold=20.0,
                        api_key=MOCK_API_KEY,
                        scorer_id=MOCK_SCORER_ID,
                    )
                    # Note: This test is simplified; in reality we'd need
                    # more sophisticated mocking

    @pytest.mark.asyncio
    async def test_get_passport_score_function(self):
        """Test the get_passport_score convenience function."""
        # Similar pattern to verify_humanity
        pass  # Placeholder for similar test


class TestStampProvider:
    """Tests for the StampProvider enum."""

    def test_all_providers_have_values(self):
        """Test all stamp providers have string values."""
        for provider in StampProvider:
            assert isinstance(provider.value, str)
            assert len(provider.value) > 0

    def test_common_providers_exist(self):
        """Test common providers are defined."""
        assert StampProvider.GOOGLE.value == "Google"
        assert StampProvider.GITHUB.value == "Github"
        assert StampProvider.TWITTER.value == "Twitter"
        assert StampProvider.ENS.value == "Ens"
        assert StampProvider.BRIGHTID.value == "BrightId"


class TestIntegrationWithIdentitySystem:
    """Tests for integration with the broader identity system."""

    def test_gitcoin_passport_factor_contribution(self):
        """Test GitcoinPassportFactor contribution calculation."""
        # Score >= 50 should give 0.4 contribution
        high_score_factor = GitcoinPassportFactor(
            factor_id="test-1",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.ACTIVE,
            passport_address=MOCK_ADDRESS,
            score=55.0,
        )
        assert high_score_factor.get_contribution() == 0.4

        # Score >= 20 but < 50 should give 0.3 contribution
        mid_score_factor = GitcoinPassportFactor(
            factor_id="test-2",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.ACTIVE,
            passport_address=MOCK_ADDRESS,
            score=35.0,
        )
        assert mid_score_factor.get_contribution() == 0.3

        # Score < 20 should give 0 contribution
        low_score_factor = GitcoinPassportFactor(
            factor_id="test-3",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.ACTIVE,
            passport_address=MOCK_ADDRESS,
            score=15.0,
        )
        assert low_score_factor.get_contribution() == 0.0

    def test_inactive_factor_no_contribution(self):
        """Test inactive factor gives no contribution."""
        inactive_factor = GitcoinPassportFactor(
            factor_id="test-inactive",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.INACTIVE,
            passport_address=MOCK_ADDRESS,
            score=100.0,  # High score but inactive
        )
        assert inactive_factor.get_contribution() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

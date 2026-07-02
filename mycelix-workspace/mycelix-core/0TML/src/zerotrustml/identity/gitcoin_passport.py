# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gitcoin Passport Integration for Sybil Resistance in Mycelix

Provides Proof of Personhood verification through Gitcoin Passport stamps
(verifiable credentials). Integrates with the Stamps API v2 for score and
stamp retrieval.

API Reference: https://docs.passport.xyz
Base URL: https://api.passport.xyz

This module enables:
- Score-based humanity verification (0-100 scale)
- Stamp collection and verification
- E2GitcoinPassport assurance level integration
- Sybil-resistant reputation in the Mycelix network

Example:
    >>> from zerotrustml.identity.gitcoin_passport import GitcoinPassportClient
    >>> async with GitcoinPassportClient(api_key="your-key", scorer_id="your-id") as client:
    ...     score = await client.verify_passport("0x...")
    ...     stamps = await client.get_stamps("0x...")
    ...     is_human = await client.check_humanity_threshold("0x...", min_score=20.0)
"""

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import aiohttp

from .factors import FactorStatus, FactorCategory, GitcoinPassportFactor

logger = logging.getLogger(__name__)

# Type variable for generic retry decorator
T = TypeVar('T')


class PassportError(Exception):
    """Base exception for Gitcoin Passport errors."""
    pass


class RateLimitError(PassportError):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(PassportError):
    """Raised when API authentication fails."""
    pass


class AddressNotFoundError(PassportError):
    """Raised when the address is not found in the Passport registry."""
    pass


class RetryExhaustedError(PassportError):
    """Raised when all retry attempts have been exhausted."""
    pass


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior on transient failures.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exceptions that trigger retries
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[type, ...] = (
        RateLimitError,
        aiohttp.ClientError,
        asyncio.TimeoutError,
    )

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given retry attempt with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        return delay


@dataclass
class CacheEntry:
    """
    A single cache entry with TTL support.

    Attributes:
        data: The cached data
        created_at: Timestamp when the entry was created
        ttl_seconds: Time-to-live in seconds
        access_count: Number of times this entry has been accessed
        last_accessed: Timestamp of last access
    """
    data: Any
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return time.time() - self.created_at

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class PassportCache:
    """
    Thread-safe LRU cache for Passport scores with TTL support.

    Provides efficient caching to reduce API calls while respecting
    rate limits. Supports automatic eviction of expired entries and
    LRU eviction when capacity is reached.

    Usage:
        >>> cache = PassportCache(max_size=1000, default_ttl=300)
        >>> cache.set("0x123", score_data)
        >>> cached = cache.get("0x123")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 300,
        cleanup_interval: int = 60,
    ):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds
            cleanup_interval: Interval for automatic cleanup in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def _normalize_key(self, address: str) -> str:
        """Normalize address to lowercase for consistent keys."""
        return address.lower().strip()

    async def get(self, address: str) -> Optional[Any]:
        """
        Get a cached value if it exists and is not expired.

        Args:
            address: Ethereum address

        Returns:
            Cached data or None if not found/expired
        """
        key = self._normalize_key(address)
        async with self._lock:
            await self._maybe_cleanup()

            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["expirations"] += 1
                return None

            entry.touch()
            self._stats["hits"] += 1
            return entry.data

    async def set(
        self,
        address: str,
        data: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a value in the cache.

        Args:
            address: Ethereum address
            data: Data to cache
            ttl: Optional TTL override in seconds
        """
        key = self._normalize_key(address)
        async with self._lock:
            await self._maybe_cleanup()

            # Evict LRU if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()

            self._cache[key] = CacheEntry(
                data=data,
                ttl_seconds=ttl or self.default_ttl,
            )

    async def invalidate(self, address: str) -> bool:
        """
        Remove an entry from the cache.

        Args:
            address: Ethereum address

        Returns:
            True if entry was found and removed
        """
        key = self._normalize_key(address)
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        if (time.time() - self._last_cleanup) > self.cleanup_interval:
            await self._cleanup_expired()
            self._last_cleanup = time.time()

    async def _cleanup_expired(self) -> int:
        """Remove all expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            del self._cache[key]
        self._stats["expirations"] += len(expired_keys)
        return len(expired_keys)

    async def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[lru_key]
        self._stats["evictions"] += 1

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            **self._stats,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)


class StampProvider(Enum):
    """Known Gitcoin Passport stamp providers."""
    # Social Media
    GOOGLE = "Google"
    TWITTER = "Twitter"
    GITHUB = "Github"
    DISCORD = "Discord"
    FACEBOOK = "Facebook"
    LINKEDIN = "Linkedin"

    # Web3 & Blockchain
    ENS = "Ens"
    NFT = "NFT"
    BRIGHTID = "BrightId"
    POH = "Poh"  # Proof of Humanity
    LENS = "Lens"
    GNOSIS_SAFE = "GnosisSafe"
    ETH_BALANCE = "EthBalance"
    ETH_TRANSACTIONS = "EthTransactions"
    ETH_GAS_SPENT = "EthGasSpent"

    # Identity Verification
    COINBASE = "Coinbase"
    HOLONYM = "Holonym"
    CIVIC = "Civic"
    IDENA = "Idena"
    WORLDCOIN = "Worldcoin"

    # Developer
    GITCOIN_GRANTS = "GitcoinGrants"
    GITCOIN_CONTRIBUTOR = "GitcoinContributor"
    STACK_OVERFLOW = "StackOverflow"

    # Education
    GUILD_XYZ = "GuildXYZ"
    SNAPSHOT = "Snapshot"

    # Other
    ZK_SYNC = "ZkSync"
    TRUSTA_LABS = "TrustaLabs"
    HYPERCERTS = "Hypercerts"


@dataclass
class Stamp:
    """
    A verified stamp (credential) from Gitcoin Passport.

    Represents a single proof of identity/action that contributes
    to the overall Passport score.

    Attributes:
        provider: The stamp provider (e.g., "Google", "Github")
        credential_hash: Hash of the verifiable credential
        score: Weight/score contribution of this stamp
        expires_at: When the stamp expires
        metadata: Additional stamp-specific metadata
        is_duplicate: Whether this stamp is deduplicated
    """
    provider: str
    credential_hash: str
    score: float = 0.0
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_duplicate: bool = False

    @property
    def is_expired(self) -> bool:
        """Check if the stamp has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if stamp is valid (not expired and not a duplicate)."""
        return not self.is_expired and not self.is_duplicate

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stamp to dictionary."""
        return {
            "provider": self.provider,
            "credential_hash": self.credential_hash,
            "score": self.score,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_duplicate": self.is_duplicate,
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


@dataclass
class PassportScore:
    """
    Complete Passport score response.

    Contains the overall score, individual stamp scores, and metadata
    about the verification.

    Attributes:
        address: The Ethereum address
        score: Overall Passport score (0-100)
        passing: Whether the score passes the threshold
        threshold: The passing threshold used
        stamps: List of verified stamps
        last_updated: When the score was last calculated
        expiration: When the score expires
        raw_response: Original API response
    """
    address: str
    score: float
    passing: bool
    threshold: float
    stamps: List[Stamp] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    expiration: Optional[datetime] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def stamp_count(self) -> int:
        """Number of valid (non-duplicate) stamps."""
        return len([s for s in self.stamps if s.is_valid])

    @property
    def is_expired(self) -> bool:
        """Check if the score has expired."""
        if self.expiration is None:
            return False
        return datetime.now(timezone.utc) > self.expiration

    def get_stamp(self, provider: str) -> Optional[Stamp]:
        """Get a specific stamp by provider name."""
        for stamp in self.stamps:
            if stamp.provider.lower() == provider.lower():
                return stamp
        return None

    def has_stamp(self, provider: Union[str, StampProvider]) -> bool:
        """Check if a specific stamp is present and valid."""
        provider_name = provider.value if isinstance(provider, StampProvider) else provider
        stamp = self.get_stamp(provider_name)
        return stamp is not None and stamp.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Serialize score to dictionary."""
        return {
            "address": self.address,
            "score": self.score,
            "passing": self.passing,
            "threshold": self.threshold,
            "stamp_count": self.stamp_count,
            "stamps": [s.to_dict() for s in self.stamps],
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "is_expired": self.is_expired,
        }


class GitcoinPassportClient:
    """
    Async client for the Gitcoin Passport Stamps API v2.

    Provides methods to fetch Passport scores, stamps, and verify
    humanity for Sybil resistance. Includes built-in caching and
    retry logic for production reliability.

    Rate Limits (per 15 minutes):
    - Tier 1: 125 requests
    - Tier 2: 350 requests
    - Tier 3: 2000 requests
    - Tier 4: 2000+ requests

    Usage:
        >>> async with GitcoinPassportClient(api_key, scorer_id) as client:
        ...     score = await client.verify_passport("0x1234...")
        ...     print(f"Score: {score.score}, Passing: {score.passing}")
        ...     is_human = await client.check_humanity_threshold("0x...", 20.0)

    Environment Variables:
        GITCOIN_PASSPORT_API_KEY: API key for authentication
        GITCOIN_PASSPORT_SCORER_ID: Default scorer ID
    """

    BASE_URL = "https://api.passport.xyz"
    API_VERSION = "v2"
    DEFAULT_TIMEOUT = 60  # API has 60 second timeout
    DEFAULT_CACHE_TTL = 300  # 5 minutes

    def __init__(
        self,
        api_key: Optional[str] = None,
        scorer_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        cache_max_size: int = 1000,
        retry_config: Optional[RetryConfig] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the Gitcoin Passport client.

        Args:
            api_key: API key for authentication (or GITCOIN_PASSPORT_API_KEY env var)
            scorer_id: Scorer ID for score calculations (or GITCOIN_PASSPORT_SCORER_ID env var)
            base_url: Override base URL for testing
            timeout: Request timeout in seconds (default: 60)
            cache_ttl: Cache time-to-live in seconds (default: 300)
            cache_max_size: Maximum cache entries (default: 1000)
            retry_config: Configuration for retry behavior
            enable_cache: Whether to enable response caching (default: True)

        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("GITCOIN_PASSPORT_API_KEY")
        self.scorer_id = scorer_id or os.getenv("GITCOIN_PASSPORT_SCORER_ID")
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

        # Caching configuration
        self.enable_cache = enable_cache
        self._cache = PassportCache(
            max_size=cache_max_size,
            default_ttl=cache_ttl,
        ) if enable_cache else None

        # Retry configuration
        self.retry_config = retry_config or RetryConfig()

        if not self.api_key:
            raise ValueError(
                "API key is required. Set GITCOIN_PASSPORT_API_KEY environment "
                "variable or pass api_key parameter."
            )

    async def __aenter__(self) -> "GitcoinPassportClient":
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure an aiohttp session exists."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_url(self, *path_parts: str) -> str:
        """Build API URL from path parts."""
        parts = [self.base_url, self.API_VERSION] + list(path_parts)
        return "/".join(p.strip("/") for p in parts)

    async def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retry: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an authenticated API request with retry support.

        Args:
            method: HTTP method
            url: Full URL
            params: Query parameters
            json_data: JSON body data
            retry: Whether to retry on transient failures

        Returns:
            Parsed JSON response

        Raises:
            RateLimitError: If rate limit exceeded after retries
            AuthenticationError: If authentication fails
            AddressNotFoundError: If address not in registry
            RetryExhaustedError: If all retries exhausted
            PassportError: For other API errors
        """
        last_exception: Optional[Exception] = None
        max_attempts = self.retry_config.max_retries + 1 if retry else 1

        for attempt in range(max_attempts):
            try:
                return await self._make_request(method, url, params, json_data)
            except self.retry_config.retryable_exceptions as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    delay = self.retry_config.calculate_delay(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_attempts}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Request failed after {max_attempts} attempts: {e}"
                    )
            except (AuthenticationError, AddressNotFoundError):
                # Don't retry authentication or not-found errors
                raise

        raise RetryExhaustedError(
            f"All {max_attempts} retry attempts exhausted. Last error: {last_exception}"
        )

    async def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single API request without retry logic.

        Args:
            method: HTTP method
            url: Full URL
            params: Query parameters
            json_data: JSON body data

        Returns:
            Parsed JSON response

        Raises:
            RateLimitError: If rate limit exceeded
            AuthenticationError: If authentication fails
            PassportError: For other API errors
        """
        session = await self._ensure_session()

        try:
            async with session.request(
                method,
                url,
                params=params,
                json=json_data,
            ) as response:
                response_text = await response.text()

                if response.status == 401:
                    raise AuthenticationError("Invalid API key")

                if response.status == 429:
                    raise RateLimitError(
                        "Rate limit exceeded. Please wait before making more requests."
                    )

                if response.status == 404:
                    raise AddressNotFoundError(
                        f"Address not found in Passport registry"
                    )

                if response.status >= 400:
                    raise PassportError(
                        f"API error {response.status}: {response_text}"
                    )

                if response_text:
                    return await response.json()
                return {}

        except aiohttp.ClientError as e:
            raise PassportError(f"Network error: {e}")

    async def get_score(
        self,
        address: str,
        scorer_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PassportScore:
        """
        Get the Passport score for an Ethereum address.

        Retrieves the latest calculated score and associated stamps
        for the given address. Results are cached to minimize API calls.

        Args:
            address: Ethereum address (0x prefixed)
            scorer_id: Override the default scorer ID
            use_cache: Whether to use cached results (default: True)

        Returns:
            PassportScore with score details and stamps

        Raises:
            AddressNotFoundError: If address not in registry
            PassportError: For other API errors

        Example:
            >>> score = await client.get_score("0x1234567890123456789012345678901234567890")
            >>> print(f"Score: {score.score}")
        """
        sid = scorer_id or self.scorer_id
        if not sid:
            raise ValueError("Scorer ID is required")

        # Normalize address
        address = address.lower()

        # Check cache first
        cache_key = f"score:{sid}:{address}"
        if use_cache and self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {address}")
                return cached

        url = self._build_url("stamps", sid, "score", address)

        logger.debug(f"Fetching score for {address} with scorer {sid}")

        response = await self._request("GET", url)

        # Parse stamps from response
        stamps = []
        stamps_data = response.get("stamps", {})

        for provider, stamp_info in stamps_data.items():
            if isinstance(stamp_info, dict):
                expires_str = stamp_info.get("expiration_date")
                expires_at = None
                if expires_str:
                    try:
                        expires_at = datetime.fromisoformat(
                            expires_str.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass

                stamps.append(Stamp(
                    provider=provider,
                    credential_hash=stamp_info.get("hash", ""),
                    score=float(stamp_info.get("score", 0)),
                    expires_at=expires_at,
                    is_duplicate=stamp_info.get("dedup", False),
                    metadata=stamp_info.get("metadata", {}),
                ))

        # Parse timestamps
        last_score_ts = response.get("last_score_timestamp")
        last_updated = None
        if last_score_ts:
            try:
                last_updated = datetime.fromisoformat(
                    last_score_ts.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        expiration_ts = response.get("expiration_timestamp")
        expiration = None
        if expiration_ts:
            try:
                expiration = datetime.fromisoformat(
                    expiration_ts.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        score_value = float(response.get("score", 0))
        passing = response.get("passing_score", False)

        result = PassportScore(
            address=address,
            score=score_value,
            passing=passing,
            threshold=20.0,  # Default threshold
            stamps=stamps,
            last_updated=last_updated,
            expiration=expiration,
            raw_response=response,
        )

        # Cache the result
        if self._cache:
            await self._cache.set(cache_key, result)

        return result

    async def verify_passport(
        self,
        wallet_address: str,
        scorer_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PassportScore:
        """
        Verify a wallet's Passport and retrieve its score.

        This is the primary method for retrieving Passport verification
        data. It's an alias for get_score() with a more intuitive name.

        Args:
            wallet_address: Ethereum wallet address (0x prefixed)
            scorer_id: Override the default scorer ID
            use_cache: Whether to use cached results (default: True)

        Returns:
            PassportScore containing:
            - score: Numerical score (0-100)
            - passing: Whether the score passes the threshold
            - stamps: List of verified credentials
            - last_updated: When the score was calculated
            - expiration: When the score expires

        Raises:
            AddressNotFoundError: If wallet not registered with Passport
            AuthenticationError: If API key is invalid
            RetryExhaustedError: If all retries fail
            PassportError: For other API errors

        Example:
            >>> score = await client.verify_passport("0x1234...")
            >>> print(f"Score: {score.score}, Human: {score.passing}")
        """
        return await self.get_score(wallet_address, scorer_id, use_cache)

    async def check_humanity_threshold(
        self,
        wallet_address: str,
        min_score: float = 20.0,
        required_stamps: Optional[List[Union[str, StampProvider]]] = None,
        scorer_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> bool:
        """
        Check if a wallet meets the humanity threshold for Sybil resistance.

        This is the primary method for boolean Sybil resistance checks.
        Returns True if the wallet has sufficient proof of humanity.

        Args:
            wallet_address: Ethereum wallet address (0x prefixed)
            min_score: Minimum required score (default: 20.0)
            required_stamps: Optional list of required stamp providers
            scorer_id: Override the default scorer ID
            use_cache: Whether to use cached results (default: True)

        Returns:
            True if wallet passes humanity verification, False otherwise

        Example:
            >>> is_human = await client.check_humanity_threshold(
            ...     "0x1234...",
            ...     min_score=25.0,
            ...     required_stamps=[StampProvider.GITHUB, StampProvider.GOOGLE]
            ... )
            >>> if is_human:
            ...     print("Verified human - allow governance participation")
        """
        try:
            score = await self.get_score(wallet_address, scorer_id, use_cache)

            # Check score threshold
            if score.score < min_score:
                logger.debug(
                    f"Address {wallet_address} failed: score {score.score} < {min_score}"
                )
                return False

            # Check required stamps if specified
            if required_stamps:
                for required in required_stamps:
                    if not score.has_stamp(required):
                        provider_name = (
                            required.value if isinstance(required, StampProvider)
                            else required
                        )
                        logger.debug(
                            f"Address {wallet_address} missing required stamp: {provider_name}"
                        )
                        return False

            logger.info(
                f"Address {wallet_address} verified as human: "
                f"score={score.score}, stamps={score.stamp_count}"
            )
            return True

        except AddressNotFoundError:
            logger.debug(f"Address {wallet_address} not found in Passport registry")
            return False
        except PassportError as e:
            logger.error(f"Error verifying {wallet_address}: {e}")
            return False

    async def get_stamps(
        self,
        address: str,
        include_metadata: bool = True,
        limit: int = 100,
    ) -> List[Stamp]:
        """
        Get all verified stamps for an Ethereum address.

        Retrieves the full list of stamps with their verification
        details and metadata.

        Args:
            address: Ethereum address (0x prefixed)
            include_metadata: Include additional stamp metadata
            limit: Maximum number of stamps to return

        Returns:
            List of Stamp objects

        Example:
            >>> stamps = await client.get_stamps("0x1234...")
            >>> for stamp in stamps:
            ...     print(f"{stamp.provider}: valid={stamp.is_valid}")
        """
        address = address.lower()

        url = self._build_url("stamps", address)
        params = {
            "include_metadata": str(include_metadata).lower(),
            "limit": limit,
        }

        logger.debug(f"Fetching stamps for {address}")

        response = await self._request("GET", url, params=params)

        stamps = []
        items = response.get("items", [])

        for item in items:
            credential = item.get("credential", {})
            subject = credential.get("credentialSubject", {})

            expires_str = credential.get("expirationDate")
            expires_at = None
            if expires_str:
                try:
                    expires_at = datetime.fromisoformat(
                        expires_str.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            stamps.append(Stamp(
                provider=subject.get("provider", "Unknown"),
                credential_hash=subject.get("hash", ""),
                score=0.0,  # Score not included in this endpoint
                expires_at=expires_at,
                metadata={
                    "id": item.get("id"),
                    "issuer": credential.get("issuer"),
                    "issuance_date": credential.get("issuanceDate"),
                }
            ))

        return stamps

    async def get_stamps_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about all available stamp types.

        Returns information about which stamps are available,
        their weights, and categories.

        Returns:
            Dictionary with stamp metadata
        """
        url = self._build_url("stamps", "metadata")
        return await self._request("GET", url)

    async def invalidate_cache(self, address: Optional[str] = None) -> bool:
        """
        Invalidate cache entries.

        Args:
            address: Specific address to invalidate, or None to clear all

        Returns:
            True if cache was modified
        """
        if not self._cache:
            return False

        if address:
            return await self._cache.invalidate(address)
        else:
            count = await self._cache.clear()
            return count > 0

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hit rate, size, and other stats
        """
        if not self._cache:
            return {"enabled": False}
        return {
            "enabled": True,
            **self._cache.stats,
        }

    async def verify_humanity(
        self,
        address: str,
        threshold: float = 20.0,
        required_stamps: Optional[List[Union[str, StampProvider]]] = None,
        scorer_id: Optional[str] = None,
    ) -> bool:
        """
        Verify if an address passes the humanity threshold.

        This is the primary method for Sybil resistance checks.
        Returns True if the address is likely human based on
        the Passport score and optional stamp requirements.

        Args:
            address: Ethereum address to verify
            threshold: Minimum score required (default: 20.0)
            required_stamps: Optional list of required stamp providers
            scorer_id: Override the default scorer ID

        Returns:
            True if address passes humanity verification

        Example:
            >>> is_human = await client.verify_humanity(
            ...     "0x1234...",
            ...     threshold=20.0,
            ...     required_stamps=[StampProvider.GITHUB]
            ... )
        """
        try:
            score = await self.get_score(address, scorer_id)

            # Check score threshold
            if score.score < threshold:
                logger.debug(
                    f"Address {address} failed: score {score.score} < {threshold}"
                )
                return False

            # Check required stamps if specified
            if required_stamps:
                for required in required_stamps:
                    if not score.has_stamp(required):
                        provider_name = (
                            required.value if isinstance(required, StampProvider)
                            else required
                        )
                        logger.debug(
                            f"Address {address} missing required stamp: {provider_name}"
                        )
                        return False

            logger.info(
                f"Address {address} verified as human: "
                f"score={score.score}, stamps={score.stamp_count}"
            )
            return True

        except AddressNotFoundError:
            logger.debug(f"Address {address} not found in Passport registry")
            return False
        except PassportError as e:
            logger.error(f"Error verifying {address}: {e}")
            return False

    async def get_historical_score(
        self,
        address: str,
        timestamp: datetime,
        scorer_id: Optional[str] = None,
    ) -> PassportScore:
        """
        Get the historical Passport score at a specific point in time.

        Useful for governance snapshots and retroactive verification.

        Args:
            address: Ethereum address
            timestamp: Point in time to query
            scorer_id: Override the default scorer ID

        Returns:
            PassportScore at the specified time
        """
        sid = scorer_id or self.scorer_id
        if not sid:
            raise ValueError("Scorer ID is required")

        address = address.lower()
        url = self._build_url("stamps", sid, "score", address, "history")

        params = {
            "created_at": timestamp.strftime("%Y-%m-%d"),
        }

        response = await self._request("GET", url, params=params)

        # Parse similar to get_score but from historical data
        score_value = float(response.get("score", 0))

        return PassportScore(
            address=address,
            score=score_value,
            passing=score_value >= 20.0,
            threshold=20.0,
            stamps=[],  # Historical endpoint may not include stamps
            raw_response=response,
        )

    def create_passport_factor(
        self,
        address: str,
        score: float,
        stamps: List[Stamp],
        factor_id: Optional[str] = None,
    ) -> GitcoinPassportFactor:
        """
        Create a GitcoinPassportFactor from API response.

        Converts the Passport data into the identity factor format
        used by the Mycelix identity system.

        Args:
            address: Ethereum address
            score: Passport score
            stamps: List of verified stamps
            factor_id: Optional custom factor ID

        Returns:
            Configured GitcoinPassportFactor
        """
        import uuid

        factor = GitcoinPassportFactor(
            factor_id=factor_id or f"gitcoin-{uuid.uuid4().hex[:8]}",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            passport_address=address,
            score=score,
            stamps=[s.provider for s in stamps if s.is_valid],
        )

        # Set status based on score
        if score >= 20.0:
            factor.status = FactorStatus.ACTIVE
            factor.last_verified = datetime.now(timezone.utc)
        else:
            factor.status = FactorStatus.PENDING

        return factor


class GitcoinPassportVerifier:
    """
    High-level verifier for Gitcoin Passport integration with Mycelix.

    Provides Sybil resistance verification with caching and rate
    limit handling. Designed for use in governance and claims systems.

    Usage:
        >>> verifier = GitcoinPassportVerifier(api_key="...", scorer_id="...")
        >>> result = await verifier.verify_for_governance(address)
        >>> if result.passes_e2:
        ...     # Allow governance participation
    """

    # Default thresholds for assurance levels
    E2_THRESHOLD = 20.0  # E2GitcoinPassport assurance level
    E3_THRESHOLD = 50.0  # Higher bar for E3

    def __init__(
        self,
        api_key: Optional[str] = None,
        scorer_id: Optional[str] = None,
        cache_ttl_seconds: int = 300,  # 5 minute cache
    ):
        """
        Initialize the verifier.

        Args:
            api_key: Gitcoin Passport API key
            scorer_id: Scorer ID for score calculation
            cache_ttl_seconds: How long to cache verification results
        """
        self.client = GitcoinPassportClient(api_key, scorer_id)
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple] = {}  # address -> (result, timestamp)

    async def close(self) -> None:
        """Close the underlying client."""
        await self.client.close()

    def _get_cached(self, address: str) -> Optional[PassportScore]:
        """Get cached result if still valid."""
        address = address.lower()
        if address in self._cache:
            result, cached_at = self._cache[address]
            age = (datetime.now(timezone.utc) - cached_at).total_seconds()
            if age < self.cache_ttl:
                return result
            del self._cache[address]
        return None

    def _set_cached(self, address: str, result: PassportScore) -> None:
        """Cache a verification result."""
        self._cache[address.lower()] = (result, datetime.now(timezone.utc))

    async def verify_for_governance(
        self,
        address: str,
        use_cache: bool = True,
    ) -> "GovernanceVerificationResult":
        """
        Verify an address for governance participation.

        Checks if the address meets the E2GitcoinPassport threshold
        required for governance actions like voting and proposals.

        Args:
            address: Ethereum address to verify
            use_cache: Whether to use cached results

        Returns:
            GovernanceVerificationResult with verification details
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached(address)
            if cached:
                return GovernanceVerificationResult(
                    address=address,
                    score=cached,
                    passes_e2=cached.score >= self.E2_THRESHOLD,
                    passes_e3=cached.score >= self.E3_THRESHOLD,
                    from_cache=True,
                )

        try:
            score = await self.client.get_score(address)

            if use_cache:
                self._set_cached(address, score)

            return GovernanceVerificationResult(
                address=address,
                score=score,
                passes_e2=score.score >= self.E2_THRESHOLD,
                passes_e3=score.score >= self.E3_THRESHOLD,
                from_cache=False,
            )

        except AddressNotFoundError:
            return GovernanceVerificationResult(
                address=address,
                score=None,
                passes_e2=False,
                passes_e3=False,
                error="Address not found in Passport registry",
            )
        except PassportError as e:
            return GovernanceVerificationResult(
                address=address,
                score=None,
                passes_e2=False,
                passes_e3=False,
                error=str(e),
            )


@dataclass
class GovernanceVerificationResult:
    """
    Result of governance verification check.

    Contains the full verification result including score,
    assurance level eligibility, and any errors.
    """
    address: str
    score: Optional[PassportScore]
    passes_e2: bool
    passes_e3: bool
    from_cache: bool = False
    error: Optional[str] = None

    @property
    def assurance_level(self) -> str:
        """Get the highest assurance level achieved."""
        if self.passes_e3:
            return "E3_CryptographicallyProven"
        elif self.passes_e2:
            return "E2_PrivatelyVerifiable"
        else:
            return "E1_Testimonial"

    @property
    def vote_weight_multiplier(self) -> float:
        """Get vote weight multiplier based on score."""
        if not self.score:
            return 0.5
        if self.passes_e3:
            return 1.0
        elif self.passes_e2:
            return 0.85
        else:
            return 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "address": self.address,
            "score": self.score.to_dict() if self.score else None,
            "passes_e2": self.passes_e2,
            "passes_e3": self.passes_e3,
            "assurance_level": self.assurance_level,
            "vote_weight_multiplier": self.vote_weight_multiplier,
            "from_cache": self.from_cache,
            "error": self.error,
        }


# Convenience functions for direct usage
async def verify_humanity(
    address: str,
    threshold: float = 20.0,
    api_key: Optional[str] = None,
    scorer_id: Optional[str] = None,
) -> bool:
    """
    Quick verification of humanity for an address.

    Convenience function that creates a client, verifies, and closes.
    For batch operations, use GitcoinPassportClient directly.

    Args:
        address: Ethereum address to verify
        threshold: Minimum score required
        api_key: API key (or from environment)
        scorer_id: Scorer ID (or from environment)

    Returns:
        True if address passes humanity verification

    Example:
        >>> is_human = await verify_humanity("0x1234...")
    """
    async with GitcoinPassportClient(api_key, scorer_id) as client:
        return await client.verify_humanity(address, threshold)


async def get_passport_score(
    address: str,
    api_key: Optional[str] = None,
    scorer_id: Optional[str] = None,
) -> PassportScore:
    """
    Quick retrieval of Passport score for an address.

    Convenience function that creates a client, fetches, and closes.
    For batch operations, use GitcoinPassportClient directly.

    Args:
        address: Ethereum address
        api_key: API key (or from environment)
        scorer_id: Scorer ID (or from environment)

    Returns:
        PassportScore with full score details
    """
    async with GitcoinPassportClient(api_key, scorer_id) as client:
        return await client.get_score(address)


# Example usage
if __name__ == "__main__":
    import os

    async def main():
        # Example: Verify an address
        api_key = os.getenv("GITCOIN_PASSPORT_API_KEY")
        scorer_id = os.getenv("GITCOIN_PASSPORT_SCORER_ID")

        if not api_key or not scorer_id:
            print("Set GITCOIN_PASSPORT_API_KEY and GITCOIN_PASSPORT_SCORER_ID")
            print("\nExample (mock data):")

            # Show example output structure
            example_score = PassportScore(
                address="0x1234567890123456789012345678901234567890",
                score=42.5,
                passing=True,
                threshold=20.0,
                stamps=[
                    Stamp(provider="Google", credential_hash="abc123", score=5.0),
                    Stamp(provider="Github", credential_hash="def456", score=10.0),
                    Stamp(provider="Twitter", credential_hash="ghi789", score=7.5),
                ],
            )

            print(f"Address: {example_score.address}")
            print(f"Score: {example_score.score}")
            print(f"Passing: {example_score.passing}")
            print(f"Stamp Count: {example_score.stamp_count}")
            print(f"Stamps: {[s.provider for s in example_score.stamps]}")
            return

        async with GitcoinPassportClient(api_key, scorer_id) as client:
            # Test address (use a real one with Passport)
            test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # vitalik.eth

            try:
                score = await client.get_score(test_address)
                print(f"Score for {test_address}: {score.score}")
                print(f"Passing: {score.passing}")
                print(f"Stamps: {score.stamp_count}")

                is_human = await client.verify_humanity(test_address)
                print(f"Verified human: {is_human}")

            except PassportError as e:
                print(f"Error: {e}")

    asyncio.run(main())

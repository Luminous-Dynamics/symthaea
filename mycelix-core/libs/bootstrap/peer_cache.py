# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Local Peer Cache Manager
========================

Maintains a local cache of previously discovered peers to provide
resilience when bootstrap servers are unavailable.

Features:
- Persistent storage in JSON format
- Automatic expiration of stale peers
- Region-aware peer selection
- Reputation-based peer ranking
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import threading
import hashlib

logger = logging.getLogger("mycelix.bootstrap.peer_cache")


@dataclass
class CachedPeer:
    """A cached peer entry."""
    agent_pub_key: str
    urls: List[str]
    space_hash: str
    first_seen: float  # Unix timestamp
    last_seen: float   # Unix timestamp
    last_successful_contact: Optional[float] = None
    success_count: int = 0
    failure_count: int = 0
    region: str = "unknown"
    latency_ms: Optional[float] = None
    reputation_score: float = 0.5

    @property
    def age_hours(self) -> float:
        """Get age of this peer entry in hours."""
        return (time.time() - self.first_seen) / 3600

    @property
    def staleness_hours(self) -> float:
        """Get hours since last seen."""
        return (time.time() - self.last_seen) / 3600

    @property
    def success_rate(self) -> float:
        """Calculate success rate for this peer."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Unknown
        return self.success_count / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedPeer":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PeerCacheConfig:
    """Configuration for peer cache."""
    enabled: bool = True
    path: str = "~/.mycelix/peer_cache.json"
    max_peers: int = 500
    max_age_hours: int = 168  # 7 days
    min_peers_for_offline: int = 5
    refresh_interval_hours: int = 1
    prune_interval_seconds: int = 3600  # 1 hour
    backup_enabled: bool = True
    backup_count: int = 3


class PeerCache:
    """
    Thread-safe local peer cache with persistent storage.

    Provides fallback peer discovery when bootstrap servers are unavailable.
    """

    def __init__(self, config: Optional[PeerCacheConfig] = None):
        """
        Initialize the peer cache.

        Args:
            config: Cache configuration (uses defaults if not provided)
        """
        self.config = config or PeerCacheConfig()
        self._lock = threading.RLock()
        self._peers: Dict[str, CachedPeer] = {}
        self._dirty = False
        self._last_prune = 0.0

        # Expand path
        self._cache_path = Path(os.path.expanduser(self.config.path))

        # Load existing cache
        if self.config.enabled:
            self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self._cache_path.exists():
            logger.info(f"No existing peer cache at {self._cache_path}")
            return

        try:
            with open(self._cache_path) as f:
                data = json.load(f)

            peers = data.get("peers", {})
            for key, peer_data in peers.items():
                try:
                    self._peers[key] = CachedPeer.from_dict(peer_data)
                except (TypeError, KeyError) as e:
                    logger.warning(f"Invalid peer entry {key}: {e}")

            logger.info(f"Loaded {len(self._peers)} peers from cache")

            # Prune stale entries
            self._prune_stale()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse peer cache: {e}")
            self._restore_from_backup()
        except Exception as e:
            logger.error(f"Failed to load peer cache: {e}")

    def _save(self) -> None:
        """Save cache to disk."""
        if not self.config.enabled:
            return

        try:
            # Ensure directory exists
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup
            if self.config.backup_enabled and self._cache_path.exists():
                self._create_backup()

            # Prepare data
            data = {
                "version": "1.0",
                "updated": datetime.now(timezone.utc).isoformat(),
                "peer_count": len(self._peers),
                "peers": {k: v.to_dict() for k, v in self._peers.items()},
            }

            # Write atomically
            temp_path = self._cache_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            temp_path.replace(self._cache_path)
            self._dirty = False

            logger.debug(f"Saved {len(self._peers)} peers to cache")

        except Exception as e:
            logger.error(f"Failed to save peer cache: {e}")

    def _create_backup(self) -> None:
        """Create a backup of the current cache file."""
        try:
            backup_dir = self._cache_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)

            # Rotate backups
            backups = sorted(backup_dir.glob("peer_cache_*.json"))
            while len(backups) >= self.config.backup_count:
                backups[0].unlink()
                backups = backups[1:]

            # Create new backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"peer_cache_{timestamp}.json"

            import shutil
            shutil.copy2(self._cache_path, backup_path)

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def _restore_from_backup(self) -> None:
        """Attempt to restore from most recent backup."""
        try:
            backup_dir = self._cache_path.parent / "backups"
            if not backup_dir.exists():
                return

            backups = sorted(backup_dir.glob("peer_cache_*.json"), reverse=True)
            for backup in backups:
                try:
                    with open(backup) as f:
                        data = json.load(f)
                    logger.info(f"Restored from backup: {backup}")
                    return
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Failed to restore from backup: {e}")

    def _prune_stale(self) -> None:
        """Remove stale peer entries."""
        now = time.time()

        # Rate limit pruning
        if now - self._last_prune < self.config.prune_interval_seconds:
            return

        self._last_prune = now
        max_age_seconds = self.config.max_age_hours * 3600
        pruned = 0

        with self._lock:
            stale_keys = [
                key for key, peer in self._peers.items()
                if (now - peer.last_seen) > max_age_seconds
            ]

            for key in stale_keys:
                del self._peers[key]
                pruned += 1

            if pruned > 0:
                self._dirty = True
                logger.info(f"Pruned {pruned} stale peers from cache")

    def _peer_key(self, agent_pub_key: str, space_hash: str) -> str:
        """Generate unique key for a peer."""
        combined = f"{agent_pub_key}:{space_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def add_peer(
        self,
        agent_pub_key: str,
        urls: List[str],
        space_hash: str,
        region: str = "unknown",
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Add or update a peer in the cache.

        Args:
            agent_pub_key: Agent's public key
            urls: List of URLs for this agent
            space_hash: Network/space hash
            region: Geographic region
            latency_ms: Measured latency
        """
        if not self.config.enabled:
            return

        key = self._peer_key(agent_pub_key, space_hash)
        now = time.time()

        with self._lock:
            if key in self._peers:
                # Update existing
                peer = self._peers[key]
                peer.urls = urls
                peer.last_seen = now
                if latency_ms:
                    peer.latency_ms = latency_ms
            else:
                # Add new
                self._peers[key] = CachedPeer(
                    agent_pub_key=agent_pub_key,
                    urls=urls,
                    space_hash=space_hash,
                    first_seen=now,
                    last_seen=now,
                    region=region,
                    latency_ms=latency_ms,
                )

            self._dirty = True

            # Enforce max peers
            if len(self._peers) > self.config.max_peers:
                self._evict_lowest_quality()

    def _evict_lowest_quality(self) -> None:
        """Evict lowest quality peers when cache is full."""
        if not self._peers:
            return

        # Score peers (lower is worse)
        def peer_score(peer: CachedPeer) -> float:
            # Factors: recency, success rate, latency
            recency = 1.0 / (1.0 + peer.staleness_hours)
            success = peer.success_rate
            latency_factor = 1.0
            if peer.latency_ms:
                latency_factor = 1.0 / (1.0 + peer.latency_ms / 1000)
            return recency * 0.4 + success * 0.4 + latency_factor * 0.2

        # Sort by score and remove lowest
        sorted_peers = sorted(
            self._peers.items(),
            key=lambda x: peer_score(x[1])
        )

        evict_count = len(self._peers) - self.config.max_peers
        for key, _ in sorted_peers[:evict_count]:
            del self._peers[key]

    def record_success(self, agent_pub_key: str, space_hash: str) -> None:
        """Record a successful contact with a peer."""
        if not self.config.enabled:
            return

        key = self._peer_key(agent_pub_key, space_hash)
        now = time.time()

        with self._lock:
            if key in self._peers:
                peer = self._peers[key]
                peer.success_count += 1
                peer.last_successful_contact = now
                peer.last_seen = now
                peer.reputation_score = min(1.0, peer.reputation_score + 0.05)
                self._dirty = True

    def record_failure(self, agent_pub_key: str, space_hash: str) -> None:
        """Record a failed contact attempt with a peer."""
        if not self.config.enabled:
            return

        key = self._peer_key(agent_pub_key, space_hash)

        with self._lock:
            if key in self._peers:
                peer = self._peers[key]
                peer.failure_count += 1
                peer.reputation_score = max(0.0, peer.reputation_score - 0.1)
                self._dirty = True

    def get_peers(
        self,
        space_hash: str,
        limit: int = 10,
        region: Optional[str] = None,
        min_success_rate: float = 0.0,
    ) -> List[CachedPeer]:
        """
        Get cached peers for a space.

        Args:
            space_hash: Network/space hash to filter by
            limit: Maximum number of peers to return
            region: Optional region filter
            min_success_rate: Minimum success rate filter

        Returns:
            List of cached peers, sorted by quality
        """
        if not self.config.enabled:
            return []

        self._prune_stale()

        with self._lock:
            # Filter peers
            candidates = [
                peer for peer in self._peers.values()
                if peer.space_hash == space_hash
                and peer.success_rate >= min_success_rate
                and (region is None or peer.region == region)
            ]

            # Score and sort
            def score(peer: CachedPeer) -> float:
                recency = 1.0 / (1.0 + peer.staleness_hours)
                success = peer.success_rate
                reputation = peer.reputation_score
                latency = 1.0 if not peer.latency_ms else 1.0 / (1.0 + peer.latency_ms / 100)
                return recency * 0.25 + success * 0.3 + reputation * 0.25 + latency * 0.2

            sorted_peers = sorted(candidates, key=score, reverse=True)
            return sorted_peers[:limit]

    def get_all_spaces(self) -> Set[str]:
        """Get all space hashes in the cache."""
        with self._lock:
            return {peer.space_hash for peer in self._peers.values()}

    def has_sufficient_peers(self, space_hash: str) -> bool:
        """Check if cache has enough peers for offline operation."""
        if not self.config.enabled:
            return False

        with self._lock:
            count = sum(
                1 for peer in self._peers.values()
                if peer.space_hash == space_hash
                and peer.success_rate >= 0.5
            )
            return count >= self.config.min_peers_for_offline

    def flush(self) -> None:
        """Force save cache to disk."""
        with self._lock:
            if self._dirty:
                self._save()

    def clear(self, space_hash: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            space_hash: If provided, only clear entries for this space
        """
        with self._lock:
            if space_hash:
                keys_to_remove = [
                    key for key, peer in self._peers.items()
                    if peer.space_hash == space_hash
                ]
                for key in keys_to_remove:
                    del self._peers[key]
            else:
                self._peers.clear()

            self._dirty = True

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_peers = len(self._peers)
            spaces = self.get_all_spaces()

            if not self._peers:
                return {
                    "total_peers": 0,
                    "spaces": 0,
                    "avg_success_rate": 0.0,
                    "avg_age_hours": 0.0,
                }

            avg_success = sum(p.success_rate for p in self._peers.values()) / total_peers
            avg_age = sum(p.age_hours for p in self._peers.values()) / total_peers

            return {
                "total_peers": total_peers,
                "spaces": len(spaces),
                "avg_success_rate": round(avg_success, 3),
                "avg_age_hours": round(avg_age, 1),
                "cache_path": str(self._cache_path),
                "enabled": self.config.enabled,
            }

    def __enter__(self) -> "PeerCache":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - auto-save."""
        self.flush()

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Query API
"""

from typing import List, Optional, TYPE_CHECKING

from ..models import QueryResult, EpistemicTier

if TYPE_CHECKING:
    from ..client import MycelixClient, AsyncMycelixClient


class QueryAPI:
    """Synchronous Query API"""

    def __init__(self, client: "MycelixClient"):
        self.client = client

    def search(
        self,
        category: Optional[str] = None,
        tier: Optional[EpistemicTier] = None,
        keywords: Optional[List[str]] = None,
        creator: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """
        Search for claims.

        Args:
            category: Filter by category
            tier: Filter by epistemic tier
            keywords: Filter by keywords
            creator: Filter by creator
            limit: Maximum results
            offset: Results offset

        Returns:
            Query results

        Example:
            >>> results = client.query.search(
            ...     category="longevity",
            ...     tier=EpistemicTier.E3,
            ...     keywords=["NAD+", "aging"]
            ... )
            >>> print(f"Found {results.total_count} claims")
        """
        params = {
            "limit": limit,
            "offset": offset,
        }

        if category:
            params["category"] = category
        if tier:
            params["tier"] = tier.value
        if keywords:
            params["keywords"] = ",".join(keywords)
        if creator:
            params["creator"] = creator

        data = self.client._request("GET", "/api/v1/query", params=params)
        return QueryResult(**data)

    def filter(
        self,
        min_tier: Optional[EpistemicTier] = None,
        categories: Optional[List[str]] = None,
        limit: int = 100,
    ) -> QueryResult:
        """Filter claims by criteria"""
        params = {"limit": limit}

        if min_tier:
            params["min_tier"] = min_tier.value
        if categories:
            params["categories"] = ",".join(categories)

        data = self.client._request("GET", "/api/v1/query/filter", params=params)
        return QueryResult(**data)


class AsyncQueryAPI:
    """Asynchronous Query API"""

    def __init__(self, client: "AsyncMycelixClient"):
        self.client = client

    async def search(
        self,
        category: Optional[str] = None,
        tier: Optional[EpistemicTier] = None,
        keywords: Optional[List[str]] = None,
        creator: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """Search for claims (async)"""
        params = {
            "limit": limit,
            "offset": offset,
        }

        if category:
            params["category"] = category
        if tier:
            params["tier"] = tier.value
        if keywords:
            params["keywords"] = ",".join(keywords)
        if creator:
            params["creator"] = creator

        data = await self.client._request("GET", "/api/v1/query", params=params)
        return QueryResult(**data)

    async def filter(
        self,
        min_tier: Optional[EpistemicTier] = None,
        categories: Optional[List[str]] = None,
        limit: int = 100,
    ) -> QueryResult:
        """Filter claims (async)"""
        params = {"limit": limit}

        if min_tier:
            params["min_tier"] = min_tier.value
        if categories:
            params["categories"] = ",".join(categories)

        data = await self.client._request("GET", "/api/v1/query/filter", params=params)
        return QueryResult(**data)

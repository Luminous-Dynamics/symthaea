# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Trust API
"""

from typing import Dict, Any, TYPE_CHECKING

from ..models import TrustScore, UpdateTrustScoreRequest

if TYPE_CHECKING:
    from ..client import MycelixClient, AsyncMycelixClient


class TrustAPI:
    """Synchronous Trust API"""

    def __init__(self, client: "MycelixClient"):
        self.client = client

    def get_score(self, participant_id: str) -> TrustScore:
        """
        Get trust score for a participant.

        Args:
            participant_id: Participant identifier

        Returns:
            Trust score

        Example:
            >>> score = client.trust.get_score("researcher@uni.edu")
            >>> print(f"Trust: {score.score:.2f} (confidence: {score.confidence:.2f})")
        """
        data = self.client._request("GET", f"/api/v1/trust/{participant_id}")
        return TrustScore(**data)

    def update_score(
        self,
        participant_id: str,
        score: float,
        confidence: float,
    ) -> TrustScore:
        """
        Update trust score.

        Args:
            participant_id: Participant identifier
            score: New trust score (0.0-1.0)
            confidence: Confidence in score (0.0-1.0)

        Returns:
            Updated trust score
        """
        request = UpdateTrustScoreRequest(score=score, confidence=confidence)
        data = self.client._request(
            "PUT",
            f"/api/v1/trust/{participant_id}",
            json=request.model_dump(mode="json"),
        )
        return TrustScore(**data)

    def get_stats(self) -> Dict[str, Any]:
        """Get trust system statistics"""
        return self.client._request("GET", "/api/v1/trust/stats")


class AsyncTrustAPI:
    """Asynchronous Trust API"""

    def __init__(self, client: "AsyncMycelixClient"):
        self.client = client

    async def get_score(self, participant_id: str) -> TrustScore:
        """Get trust score (async)"""
        data = await self.client._request("GET", f"/api/v1/trust/{participant_id}")
        return TrustScore(**data)

    async def update_score(
        self,
        participant_id: str,
        score: float,
        confidence: float,
    ) -> TrustScore:
        """Update trust score (async)"""
        request = UpdateTrustScoreRequest(score=score, confidence=confidence)
        data = await self.client._request(
            "PUT",
            f"/api/v1/trust/{participant_id}",
            json=request.model_dump(mode="json"),
        )
        return TrustScore(**data)

    async def get_stats(self) -> Dict[str, Any]:
        """Get trust statistics (async)"""
        return await self.client._request("GET", "/api/v1/trust/stats")

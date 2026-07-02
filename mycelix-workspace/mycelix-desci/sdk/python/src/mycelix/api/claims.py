# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Claims API
"""

from typing import List, Optional, TYPE_CHECKING
from uuid import UUID

from ..models import (
    Claim,
    CreateClaimRequest,
    AddVerificationRequest,
    AddProvenanceRequest,
    EpistemicTier,
    ClaimContent,
)

if TYPE_CHECKING:
    from ..client import MycelixClient, AsyncMycelixClient


class ClaimsAPI:
    """Synchronous Claims API"""

    def __init__(self, client: "MycelixClient"):
        self.client = client

    def create(
        self,
        tier: EpistemicTier,
        content: ClaimContent,
        creator: str,
    ) -> Claim:
        """
        Create a new claim.

        Args:
            tier: Epistemic tier
            content: Claim content
            creator: Creator identifier

        Returns:
            Created claim

        Example:
            >>> claim = client.claims.create(
            ...     tier=EpistemicTier.E0,
            ...     content=ClaimContent(
            ...         dataset_hash="blake3:abc123",
            ...         description="Novel finding",
            ...         category="longevity",
            ...         keywords=["NAD+"]
            ...     ),
            ...     creator="researcher@uni.edu"
            ... )
        """
        request = CreateClaimRequest(tier=tier, content=content, creator=creator)
        data = self.client._request(
            "POST",
            "/api/v1/claims",
            json=request.model_dump(mode="json"),
        )
        return Claim(**data)

    def get(self, claim_id: UUID) -> Claim:
        """
        Get claim by ID.

        Args:
            claim_id: Claim UUID

        Returns:
            Claim

        Raises:
            ClaimNotFoundError: If claim not found
        """
        data = self.client._request("GET", f"/api/v1/claims/{claim_id}")
        return Claim(**data)

    def list(self, limit: int = 100, offset: int = 0) -> List[Claim]:
        """
        List claims with pagination.

        Args:
            limit: Maximum claims to return
            offset: Number of claims to skip

        Returns:
            List of claims
        """
        data = self.client._request(
            "GET",
            "/api/v1/claims",
            params={"limit": limit, "offset": offset},
        )
        return [Claim(**claim) for claim in data.get("claims", [])]

    def add_verification(
        self,
        claim_id: UUID,
        verifier: str,
        signature: str,
        comments: Optional[str] = None,
    ) -> Claim:
        """Add verification to claim"""
        request = AddVerificationRequest(
            verifier=verifier, signature=signature, comments=comments
        )
        data = self.client._request(
            "POST",
            f"/api/v1/claims/{claim_id}/verifications",
            json=request.model_dump(mode="json"),
        )
        return Claim(**data)

    def add_provenance(
        self,
        claim_id: UUID,
        source: str,
        relationship: str,
    ) -> Claim:
        """Add provenance to claim"""
        request = AddProvenanceRequest(source=source, relationship=relationship)
        data = self.client._request(
            "POST",
            f"/api/v1/claims/{claim_id}/provenance",
            json=request.model_dump(mode="json"),
        )
        return Claim(**data)

    def delete(self, claim_id: UUID) -> None:
        """Delete claim"""
        self.client._request("DELETE", f"/api/v1/claims/{claim_id}")


class AsyncClaimsAPI:
    """Asynchronous Claims API"""

    def __init__(self, client: "AsyncMycelixClient"):
        self.client = client

    async def create(
        self,
        tier: EpistemicTier,
        content: ClaimContent,
        creator: str,
    ) -> Claim:
        """Create a new claim (async)"""
        request = CreateClaimRequest(tier=tier, content=content, creator=creator)
        data = await self.client._request(
            "POST",
            "/api/v1/claims",
            json=request.model_dump(mode="json"),
        )
        return Claim(**data)

    async def get(self, claim_id: UUID) -> Claim:
        """Get claim by ID (async)"""
        data = await self.client._request("GET", f"/api/v1/claims/{claim_id}")
        return Claim(**data)

    async def list(self, limit: int = 100, offset: int = 0) -> List[Claim]:
        """List claims (async)"""
        data = await self.client._request(
            "GET",
            "/api/v1/claims",
            params={"limit": limit, "offset": offset},
        )
        return [Claim(**claim) for claim in data.get("claims", [])]

    async def add_verification(
        self,
        claim_id: UUID,
        verifier: str,
        signature: str,
        comments: Optional[str] = None,
    ) -> Claim:
        """Add verification (async)"""
        request = AddVerificationRequest(
            verifier=verifier, signature=signature, comments=comments
        )
        data = await self.client._request(
            "POST",
            f"/api/v1/claims/{claim_id}/verifications",
            json=request.model_dump(mode="json"),
        )
        return Claim(**data)

    async def add_provenance(
        self,
        claim_id: UUID,
        source: str,
        relationship: str,
    ) -> Claim:
        """Add provenance (async)"""
        request = AddProvenanceRequest(source=source, relationship=relationship)
        data = await self.client._request(
            "POST",
            f"/api/v1/claims/{claim_id}/provenance",
            json=request.model_dump(mode="json"),
        )
        return Claim(**data)

    async def delete(self, claim_id: UUID) -> None:
        """Delete claim (async)"""
        await self.client._request("DELETE", f"/api/v1/claims/{claim_id}")

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix-DeSci Data Models
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID


class EpistemicTier(str, Enum):
    """Epistemic verification tier"""

    E0 = "E0"  # Unverified
    E1 = "E1"  # Basic provenance
    E2 = "E2"  # Peer-verified
    E3 = "E3"  # Reproducible
    E4 = "E4"  # Peer-reviewed


class ClaimContent(BaseModel):
    """Scientific claim content"""

    dataset_hash: str = Field(..., description="BLAKE3 hash of the dataset")
    description: str = Field(..., description="Human-readable description")
    category: str = Field(..., description="Research category")
    keywords: List[str] = Field(default_factory=list, description="Keywords for discovery")
    storage_ref: Optional[str] = Field(None, description="IPFS/Arweave reference")
    reproducibility_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    license: Optional[str] = Field(None, description="Data license (e.g., CC-BY-4.0)")


class Provenance(BaseModel):
    """Data provenance record"""

    source: str = Field(..., description="Source identifier")
    relationship: str = Field(..., description="Relationship type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Verification(BaseModel):
    """Peer verification record"""

    verifier: str = Field(..., description="Verifier identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signature: str = Field(..., description="Cryptographic signature")
    comments: Optional[str] = None


class Claim(BaseModel):
    """Scientific claim"""

    id: UUID = Field(..., description="Unique claim identifier")
    tier: EpistemicTier = Field(..., description="Epistemic tier")
    content: ClaimContent = Field(..., description="Claim content")
    creator: str = Field(..., description="Creator identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    provenance: List[Provenance] = Field(default_factory=list)
    verifications: List[Verification] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class TrustScore(BaseModel):
    """Trust score for a participant"""

    score: float = Field(..., ge=0.0, le=1.0, description="Trust score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in score")
    interaction_count: int = Field(..., ge=0, description="Number of interactions")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class QueryResult(BaseModel):
    """Query result"""

    claims: List[Claim] = Field(default_factory=list)
    total_count: int = Field(..., ge=0, description="Total matching claims")
    page: int = Field(default=0, ge=0, description="Current page number")
    execution_time_ms: int = Field(..., ge=0, description="Query execution time (ms)")


class CreateClaimRequest(BaseModel):
    """Request to create a claim"""

    tier: EpistemicTier
    content: ClaimContent
    creator: str


class AddVerificationRequest(BaseModel):
    """Request to add verification"""

    verifier: str
    signature: str
    comments: Optional[str] = None


class AddProvenanceRequest(BaseModel):
    """Request to add provenance"""

    source: str
    relationship: str


class UpdateTrustScoreRequest(BaseModel):
    """Request to update trust score"""

    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)

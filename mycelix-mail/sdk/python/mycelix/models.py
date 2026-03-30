# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Data models for Mycelix Mail SDK
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Email:
    """Email message model."""

    id: str
    message_id: str = field(metadata={"alias": "messageId"})
    from_addr: str = field(metadata={"alias": "from"})
    to: List[str] = field(default_factory=list)
    subject: str = ""
    body_text: str = field(default="", metadata={"alias": "bodyText"})
    snippet: str = ""
    date: str = ""
    received_at: str = field(default="", metadata={"alias": "receivedAt"})
    folder: str = "inbox"
    is_read: bool = field(default=False, metadata={"alias": "isRead"})
    is_starred: bool = field(default=False, metadata={"alias": "isStarred"})
    has_attachments: bool = field(default=False, metadata={"alias": "hasAttachments"})

    thread_id: Optional[str] = field(default=None, metadata={"alias": "threadId"})
    from_name: Optional[str] = field(default=None, metadata={"alias": "fromName"})
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    body_html: Optional[str] = field(default=None, metadata={"alias": "bodyHtml"})
    labels: List[str] = field(default_factory=list)
    attachments: Optional[List["Attachment"]] = None
    trust_score: Optional[float] = field(default=None, metadata={"alias": "trustScore"})
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self):
        # Handle alias mapping from API response
        if hasattr(self, "messageId"):
            self.message_id = self.messageId
        if hasattr(self, "bodyText"):
            self.body_text = self.bodyText


@dataclass
class Attachment:
    """Email attachment model."""

    id: str
    filename: str
    content_type: str = field(metadata={"alias": "contentType"})
    size: int = 0
    content_id: Optional[str] = field(default=None, metadata={"alias": "contentId"})


@dataclass
class Contact:
    """Contact model."""

    id: str
    email: str
    created_at: str = field(metadata={"alias": "createdAt"})
    updated_at: str = field(metadata={"alias": "updatedAt"})
    name: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    notes: Optional[str] = None
    trust_score: Optional[float] = field(default=None, metadata={"alias": "trustScore"})
    is_favorite: bool = field(default=False, metadata={"alias": "isFavorite"})


@dataclass
class TrustAttestation:
    """Trust attestation model."""

    id: str
    from_email: str = field(metadata={"alias": "fromEmail"})
    to_email: str = field(metadata={"alias": "toEmail"})
    level: int = 0
    context: str = ""
    created_at: str = field(metadata={"alias": "createdAt"})
    expires_at: Optional[str] = field(default=None, metadata={"alias": "expiresAt"})
    revoked_at: Optional[str] = field(default=None, metadata={"alias": "revokedAt"})


@dataclass
class TrustScore:
    """Trust score model."""

    email: str
    score: float
    level: str  # very_high, high, medium, low, very_low, unknown
    direct_attestations: int = field(default=0, metadata={"alias": "directAttestations"})
    indirect_path: Optional[List[str]] = field(default=None, metadata={"alias": "indirectPath"})


@dataclass
class Folder:
    """Email folder model."""

    name: str
    count: int = 0
    unread_count: int = field(default=0, metadata={"alias": "unreadCount"})
    parent: Optional[str] = None


@dataclass
class Label:
    """Email label model."""

    name: str
    color: str = "#808080"
    count: int = 0


@dataclass
class WebhookConfig:
    """Webhook configuration model."""

    url: str
    events: List[str] = field(default_factory=list)
    id: Optional[str] = None
    secret: Optional[str] = None
    active: bool = True
    created_at: Optional[str] = field(default=None, metadata={"alias": "createdAt"})


@dataclass
class MigrationJob:
    """Migration job status model."""

    id: str
    source_type: str = field(metadata={"alias": "sourceType"})
    status: str = "pending"  # pending, running, paused, completed, failed, cancelled
    total_items: int = field(default=0, metadata={"alias": "totalItems"})
    processed_items: int = field(default=0, metadata={"alias": "processedItems"})
    failed_items: int = field(default=0, metadata={"alias": "failedItems"})
    current_phase: str = field(default="", metadata={"alias": "currentPhase"})
    error_log: List[str] = field(default_factory=list, metadata={"alias": "errorLog"})
    created_at: str = field(default="", metadata={"alias": "createdAt"})
    started_at: Optional[str] = field(default=None, metadata={"alias": "startedAt"})
    completed_at: Optional[str] = field(default=None, metadata={"alias": "completedAt"})

    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def is_complete(self) -> bool:
        """Check if migration is complete."""
        return self.status in ("completed", "failed", "cancelled")


@dataclass
class User:
    """User account model."""

    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = field(default=None, metadata={"alias": "avatarUrl"})
    created_at: str = field(default="", metadata={"alias": "createdAt"})
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Thread:
    """Email thread model."""

    id: str
    subject: str
    messages: List[Email] = field(default_factory=list)
    participant_count: int = field(default=0, metadata={"alias": "participantCount"})
    last_activity: str = field(default="", metadata={"alias": "lastActivity"})
    is_read: bool = field(default=False, metadata={"alias": "isRead"})


@dataclass
class SearchResult:
    """Search result with highlights."""

    email: Email
    highlights: Dict[str, List[str]] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class CalendarEvent:
    """Calendar event extracted from email."""

    id: str
    title: str
    start_time: str = field(metadata={"alias": "startTime"})
    end_time: str = field(metadata={"alias": "endTime"})
    location: Optional[str] = None
    attendees: List[str] = field(default_factory=list)
    source_email_id: Optional[str] = field(default=None, metadata={"alias": "sourceEmailId"})
    status: str = "tentative"  # tentative, confirmed, cancelled


@dataclass
class ExportJob:
    """Data export job model."""

    id: str
    status: str = "pending"
    format: str = "mbox"  # mbox, eml, json
    include_attachments: bool = field(default=True, metadata={"alias": "includeAttachments"})
    created_at: str = field(default="", metadata={"alias": "createdAt"})
    completed_at: Optional[str] = field(default=None, metadata={"alias": "completedAt"})
    download_url: Optional[str] = field(default=None, metadata={"alias": "downloadUrl"})
    expires_at: Optional[str] = field(default=None, metadata={"alias": "expiresAt"})

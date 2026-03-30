# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Mail API Client
"""

import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import Email, Contact, TrustAttestation, TrustScore, Folder, Label, WebhookConfig
from .exceptions import MycelixError, AuthenticationError, RateLimitError, NotFoundError


class MycelixClient:
    """
    Main client for interacting with the Mycelix Mail API.

    Args:
        api_key: Your API key
        base_url: API base URL (default: https://api.mycelix.mail)
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> client = MycelixClient(api_key="your-api-key")
        >>> emails = client.emails.list(folder="inbox")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.mycelix.mail",
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "mycelix-sdk-python/1.0.0",
                "Content-Type": "application/json",
            },
        )

        # API namespaces
        self.emails = EmailsAPI(self)
        self.contacts = ContactsAPI(self)
        self.trust = TrustAPI(self)
        self.webhooks = WebhooksAPI(self)
        self.folders = FoldersAPI(self)
        self.labels = LabelsAPI(self)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an API request."""
        # Filter out None values from params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        response = self._client.request(
            method=method,
            url=path,
            params=params,
            json=json,
        )

        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 404:
            raise NotFoundError(f"Resource not found: {path}")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif response.status_code >= 400:
            error_data = response.json()
            raise MycelixError(
                code=error_data.get("code", "unknown"),
                message=error_data.get("message", "Unknown error"),
                status=response.status_code,
            )

        if response.status_code == 204:
            return None

        data = response.json()
        return data.get("data", data)

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EmailsAPI:
    """Email operations."""

    def __init__(self, client: MycelixClient):
        self._client = client

    def list(
        self,
        folder: Optional[str] = None,
        labels: Optional[List[str]] = None,
        is_read: Optional[bool] = None,
        is_starred: Optional[bool] = None,
        from_addr: Optional[str] = None,
        to_addr: Optional[str] = None,
        subject: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Email]:
        """List emails with optional filters."""
        params = {
            "folder": folder,
            "labels": ",".join(labels) if labels else None,
            "isRead": is_read,
            "isStarred": is_starred,
            "from": from_addr,
            "to": to_addr,
            "subject": subject,
            "after": after.isoformat() if after else None,
            "before": before.isoformat() if before else None,
            "limit": limit,
            "offset": offset,
        }
        data = self._client._request("GET", "/v1/emails", params=params)
        return [Email(**email) for email in data]

    def get(self, email_id: str) -> Email:
        """Get a single email by ID."""
        data = self._client._request("GET", f"/v1/emails/{email_id}")
        return Email(**data)

    def send(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        body_html: Optional[str] = None,
        reply_to: Optional[str] = None,
        scheduled_at: Optional[datetime] = None,
    ) -> Email:
        """Send an email."""
        payload = {
            "to": to,
            "subject": subject,
            "body": body,
            "cc": cc,
            "bcc": bcc,
            "bodyHtml": body_html,
            "replyTo": reply_to,
            "scheduledAt": scheduled_at.isoformat() if scheduled_at else None,
        }
        data = self._client._request("POST", "/v1/emails/send", json=payload)
        return Email(**data)

    def reply(self, email_id: str, body: str, reply_all: bool = False) -> Email:
        """Reply to an email."""
        data = self._client._request(
            "POST",
            f"/v1/emails/{email_id}/reply",
            json={"body": body, "replyAll": reply_all},
        )
        return Email(**data)

    def forward(self, email_id: str, to: List[str], body: Optional[str] = None) -> Email:
        """Forward an email."""
        data = self._client._request(
            "POST",
            f"/v1/emails/{email_id}/forward",
            json={"to": to, "body": body},
        )
        return Email(**data)

    def mark_read(self, email_id: str, read: bool = True) -> None:
        """Mark an email as read/unread."""
        self._client._request("PATCH", f"/v1/emails/{email_id}", json={"isRead": read})

    def star(self, email_id: str, starred: bool = True) -> None:
        """Star/unstar an email."""
        self._client._request("PATCH", f"/v1/emails/{email_id}", json={"isStarred": starred})

    def move(self, email_id: str, folder: str) -> None:
        """Move an email to a folder."""
        self._client._request("POST", f"/v1/emails/{email_id}/move", json={"folder": folder})

    def delete(self, email_id: str, permanent: bool = False) -> None:
        """Delete an email."""
        self._client._request("DELETE", f"/v1/emails/{email_id}", params={"permanent": permanent})

    def search(
        self,
        query: str,
        folders: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        has_attachments: Optional[bool] = None,
        limit: int = 50,
    ) -> List[Email]:
        """Search emails."""
        params = {
            "query": query,
            "folders": ",".join(folders) if folders else None,
            "dateFrom": date_from.isoformat() if date_from else None,
            "dateTo": date_to.isoformat() if date_to else None,
            "hasAttachments": has_attachments,
            "limit": limit,
        }
        data = self._client._request("GET", "/v1/emails/search", params=params)
        return [Email(**email) for email in data]


class ContactsAPI:
    """Contact operations."""

    def __init__(self, client: MycelixClient):
        self._client = client

    def list(
        self,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Contact]:
        """List contacts."""
        data = self._client._request(
            "GET",
            "/v1/contacts",
            params={"search": search, "limit": limit, "offset": offset},
        )
        return [Contact(**contact) for contact in data]

    def get(self, contact_id: str) -> Contact:
        """Get a contact by ID."""
        data = self._client._request("GET", f"/v1/contacts/{contact_id}")
        return Contact(**data)

    def create(
        self,
        email: str,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        company: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Contact:
        """Create a new contact."""
        data = self._client._request(
            "POST",
            "/v1/contacts",
            json={
                "email": email,
                "name": name,
                "phone": phone,
                "company": company,
                "notes": notes,
            },
        )
        return Contact(**data)

    def update(self, contact_id: str, **kwargs) -> Contact:
        """Update a contact."""
        data = self._client._request("PATCH", f"/v1/contacts/{contact_id}", json=kwargs)
        return Contact(**data)

    def delete(self, contact_id: str) -> None:
        """Delete a contact."""
        self._client._request("DELETE", f"/v1/contacts/{contact_id}")


class TrustAPI:
    """Trust and attestation operations."""

    def __init__(self, client: MycelixClient):
        self._client = client

    def get_score(self, email: str) -> TrustScore:
        """Get trust score for an email address."""
        data = self._client._request("GET", f"/v1/trust/score/{email}")
        return TrustScore(**data)

    def get_scores(self, emails: List[str]) -> Dict[str, TrustScore]:
        """Get trust scores for multiple email addresses."""
        data = self._client._request("POST", "/v1/trust/scores", json={"emails": emails})
        return {email: TrustScore(**score) for email, score in data.items()}

    def create_attestation(
        self,
        to_email: str,
        level: int,
        context: str,
        expires_at: Optional[datetime] = None,
    ) -> TrustAttestation:
        """Create a trust attestation."""
        data = self._client._request(
            "POST",
            "/v1/trust/attestations",
            json={
                "toEmail": to_email,
                "level": level,
                "context": context,
                "expiresAt": expires_at.isoformat() if expires_at else None,
            },
        )
        return TrustAttestation(**data)

    def revoke_attestation(self, attestation_id: str) -> None:
        """Revoke a trust attestation."""
        self._client._request("DELETE", f"/v1/trust/attestations/{attestation_id}")

    def list_attestations(self, type: str = "given") -> List[TrustAttestation]:
        """List attestations (given or received)."""
        data = self._client._request("GET", "/v1/trust/attestations", params={"type": type})
        return [TrustAttestation(**att) for att in data]

    def get_trust_path(self, from_email: str, to_email: str) -> List[str]:
        """Get trust path between two email addresses."""
        return self._client._request(
            "GET",
            "/v1/trust/path",
            params={"from": from_email, "to": to_email},
        )


class WebhooksAPI:
    """Webhook management."""

    def __init__(self, client: MycelixClient):
        self._client = client

    def list(self) -> List[WebhookConfig]:
        """List all webhooks."""
        data = self._client._request("GET", "/v1/webhooks")
        return [WebhookConfig(**webhook) for webhook in data]

    def create(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        active: bool = True,
    ) -> WebhookConfig:
        """Create a webhook."""
        data = self._client._request(
            "POST",
            "/v1/webhooks",
            json={"url": url, "events": events, "secret": secret, "active": active},
        )
        return WebhookConfig(**data)

    def update(self, webhook_id: str, **kwargs) -> WebhookConfig:
        """Update a webhook."""
        data = self._client._request("PATCH", f"/v1/webhooks/{webhook_id}", json=kwargs)
        return WebhookConfig(**data)

    def delete(self, webhook_id: str) -> None:
        """Delete a webhook."""
        self._client._request("DELETE", f"/v1/webhooks/{webhook_id}")

    def test(self, webhook_id: str) -> Dict[str, Any]:
        """Test a webhook."""
        return self._client._request("POST", f"/v1/webhooks/{webhook_id}/test")


class FoldersAPI:
    """Folder management."""

    def __init__(self, client: MycelixClient):
        self._client = client

    def list(self) -> List[Folder]:
        """List all folders."""
        data = self._client._request("GET", "/v1/folders")
        return [Folder(**folder) for folder in data]

    def create(self, name: str, parent: Optional[str] = None) -> Folder:
        """Create a folder."""
        data = self._client._request("POST", "/v1/folders", json={"name": name, "parent": parent})
        return Folder(**data)

    def rename(self, name: str, new_name: str) -> None:
        """Rename a folder."""
        self._client._request("PATCH", f"/v1/folders/{name}", json={"name": new_name})

    def delete(self, name: str) -> None:
        """Delete a folder."""
        self._client._request("DELETE", f"/v1/folders/{name}")


class LabelsAPI:
    """Label management."""

    def __init__(self, client: MycelixClient):
        self._client = client

    def list(self) -> List[Label]:
        """List all labels."""
        data = self._client._request("GET", "/v1/labels")
        return [Label(**label) for label in data]

    def create(self, name: str, color: str) -> Label:
        """Create a label."""
        data = self._client._request("POST", "/v1/labels", json={"name": name, "color": color})
        return Label(**data)

    def update(self, name: str, **kwargs) -> None:
        """Update a label."""
        self._client._request("PATCH", f"/v1/labels/{name}", json=kwargs)

    def delete(self, name: str) -> None:
        """Delete a label."""
        self._client._request("DELETE", f"/v1/labels/{name}")

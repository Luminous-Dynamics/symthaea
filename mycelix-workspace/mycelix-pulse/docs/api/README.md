# Mycelix Mail API Reference

## Overview

Mycelix Mail provides a RESTful API for all email operations, trust management, and system configuration.

**Base URL:** `https://your-domain.com/api`

**Authentication:** Bearer token (JWT) in Authorization header

## Authentication

### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "your-password"
}
```

Response:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 3600,
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "name": "User Name"
  }
}
```

### WebAuthn/Passkey Login
```http
POST /api/auth/webauthn/login/start
POST /api/auth/webauthn/login/finish
```

## Email Operations

### List Emails
```http
GET /api/emails?folder=inbox&limit=50&offset=0
Authorization: Bearer {token}
```

Query Parameters:
- `folder` - Folder name (default: inbox)
- `limit` - Max results (default: 50, max: 100)
- `offset` - Pagination offset
- `unread` - Filter unread only (boolean)
- `starred` - Filter starred only (boolean)

### Get Email
```http
GET /api/emails/{id}
Authorization: Bearer {token}
```

### Send Email
```http
POST /api/emails/send
Authorization: Bearer {token}
Content-Type: application/json

{
  "to": ["recipient@example.com"],
  "cc": [],
  "bcc": [],
  "subject": "Hello",
  "body": "Email body text",
  "body_html": "<p>Email body HTML</p>",
  "attachments": [],
  "reply_to": "optional-message-id",
  "encrypt": false,
  "sign": true
}
```

### Search Emails
```http
GET /api/emails/search?q=search+query
Authorization: Bearer {token}
```

Search supports:
- Full-text search: `meeting notes`
- Field search: `from:alice@example.com`
- Date ranges: `after:2024-01-01 before:2024-12-31`
- Boolean operators: `important AND unread`
- Labels: `label:work`
- Attachments: `has:attachment`

### Delete Email
```http
DELETE /api/emails/{id}
Authorization: Bearer {token}
```

### Move Email
```http
POST /api/emails/{id}/move
Authorization: Bearer {token}
Content-Type: application/json

{
  "folder": "archive"
}
```

### Bulk Actions
```http
POST /api/emails/bulk
Authorization: Bearer {token}
Content-Type: application/json

{
  "ids": ["id1", "id2", "id3"],
  "action": "archive",
  "params": {}
}
```

Actions: `archive`, `delete`, `mark_read`, `mark_unread`, `star`, `unstar`, `move`, `label`

## Folders

### List Folders
```http
GET /api/folders
Authorization: Bearer {token}
```

### Create Folder
```http
POST /api/folders
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "Projects",
  "parent": "optional-parent-id"
}
```

### Delete Folder
```http
DELETE /api/folders/{id}
Authorization: Bearer {token}
```

## Trust & Attestations

### Get Trust Score
```http
GET /api/trust/{email}
Authorization: Bearer {token}
```

Response:
```json
{
  "email": "contact@example.com",
  "score": 0.85,
  "level": "high",
  "attestations": [
    {
      "id": "uuid",
      "from": "you@example.com",
      "trust_level": 0.9,
      "context": "colleague",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "sources": {
    "direct": 0.9,
    "web_of_trust": 0.8,
    "interaction": 0.85
  }
}
```

### Create Attestation
```http
POST /api/trust/attestations
Authorization: Bearer {token}
Content-Type: application/json

{
  "target_email": "contact@example.com",
  "trust_level": 0.9,
  "context": "colleague",
  "expires_at": "2025-01-15T00:00:00Z"
}
```

### Revoke Attestation
```http
DELETE /api/trust/attestations/{id}
Authorization: Bearer {token}
```

## Accounts

### List Accounts
```http
GET /api/accounts
Authorization: Bearer {token}
```

### Add Account
```http
POST /api/accounts
Authorization: Bearer {token}
Content-Type: application/json

{
  "email": "another@example.com",
  "imap_host": "imap.example.com",
  "imap_port": 993,
  "smtp_host": "smtp.example.com",
  "smtp_port": 587,
  "username": "another@example.com",
  "password": "password",
  "oauth2": null
}
```

### Sync Account
```http
POST /api/accounts/{id}/sync
Authorization: Bearer {token}
```

## AI Copilot

### Summarize Thread
```http
POST /api/copilot/summarize
Authorization: Bearer {token}
Content-Type: application/json

{
  "thread_id": "thread-id"
}
```

### Extract Actions
```http
POST /api/copilot/actions
Authorization: Bearer {token}
Content-Type: application/json

{
  "email_id": "email-id"
}
```

### Smart Reply
```http
POST /api/copilot/smart-reply
Authorization: Bearer {token}
Content-Type: application/json

{
  "email_id": "email-id",
  "tone": "professional"
}
```

## Webhooks

### Register Webhook
```http
POST /api/webhooks
Authorization: Bearer {token}
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["email.received", "email.sent"],
  "secret": "your-webhook-secret"
}
```

Events:
- `email.received` - New email received
- `email.sent` - Email sent
- `email.deleted` - Email deleted
- `sync.completed` - Sync completed
- `trust.attestation_created` - New attestation

### Webhook Payload
```json
{
  "event": "email.received",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "id": "email-id",
    "from": "sender@example.com",
    "subject": "Email subject"
  },
  "signature": "sha256=..."
}
```

## Rate Limits

| Endpoint | Rate Limit |
|----------|------------|
| Authentication | 10/minute |
| Email send | 100/hour |
| Search | 60/minute |
| API (general) | 1000/hour |

Rate limit headers:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

## Error Responses

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email address",
    "details": {
      "field": "to",
      "reason": "Invalid format"
    }
  }
}
```

Error codes:
- `UNAUTHORIZED` - Invalid or missing token
- `FORBIDDEN` - Insufficient permissions
- `NOT_FOUND` - Resource not found
- `VALIDATION_ERROR` - Invalid request data
- `RATE_LIMITED` - Too many requests
- `INTERNAL_ERROR` - Server error

## SDKs

- **JavaScript/TypeScript:** `npm install @mycelix/mail-sdk`
- **Python:** `pip install mycelix-mail`
- **Rust:** `cargo add mycelix-mail-client`
- **Go:** `go get github.com/mycelix/mail-go`

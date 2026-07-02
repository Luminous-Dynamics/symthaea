# Upload Validation Cheatsheet

Current behavior:
- Size limit: 100 MB.
- MIME sniff + allowlist: `audio/mpeg`, `audio/wav`, `audio/flac`, `image/png`, `image/jpeg` (can be disabled with `ENABLE_UPLOAD_VALIDATION=false`).
- Quotas: per IP and per wallet (limits configured via env).

To harden further (next steps):
- Enforce required headers (e.g., `Content-Length`, `Content-Type`).
- Add checksum (SHA-256) validation on upload; require hash in finalize step.
- Add max duration for audio (if codec parsing available).
- Add allowlist for image dimensions (if parsing available).
- Add preflight endpoint for validation without upload.

Config:
- `ENABLE_UPLOAD_VALIDATION=true|false` (default true).
- Quota envs: `UPLOAD_QUOTA_WINDOW_SECONDS`, `UPLOAD_QUOTA_MAX`, `UPLOAD_QUOTA_WALLET_MAX`.

Recommended rollout:
- Keep validation on in staging/prod; allow disabling in dev via env.
- Log validation failures with requestId and reason; avoid logging file contents.

-- SFEB MON-004 PostgreSQL outbox/claim contract v1.
-- REFERENCE ONLY: not executed in this environment (psql/PostgreSQL unavailable).
-- This is an adapter contract, not a complete migration or a production proof.
-- Important distinctions:
--   * objective_fencing_epoch: authority at the external mutation boundary.
--   * claim_epoch/claim_token: temporary ownership of this outbox row.
--     They are NOT interchangeable.
--   * remote_outcome='unknown' is a durable reconciliation state, not failure.
--   * a successful local DB transaction does not atomically commit a remote API.

CREATE SCHEMA IF NOT EXISTS sfeb;

CREATE TABLE IF NOT EXISTS sfeb.tenant_policy (
    tenant_id              text PRIMARY KEY,
    permission_revision    bigint NOT NULL CHECK (permission_revision > 0),
    action_permitted       boolean NOT NULL,
    updated_at             timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS sfeb.objective_fences (
    tenant_id              text NOT NULL,
    objective_id           text NOT NULL,
    fencing_epoch          bigint NOT NULL CHECK (fencing_epoch > 0),
    worker_id              text NOT NULL,
    lease_expires_at       timestamptz NOT NULL,
    policy_revision        bigint NOT NULL CHECK (policy_revision > 0),
    PRIMARY KEY (tenant_id, objective_id)
);

CREATE TABLE IF NOT EXISTS sfeb.source_receipts (
    receipt_id             text PRIMARY KEY,
    tenant_id              text NOT NULL,
    source_uri             text NOT NULL,
    event_id               text NOT NULL,
    source_sequence        bigint NOT NULL CHECK (source_sequence > 0),
    payload_sha256         bytea NOT NULL CHECK (octet_length(payload_sha256) = 32),
    resource_id             text NOT NULL,
    action_type             text NOT NULL,
    issuer_id               text NOT NULL,
    issued_at               timestamptz NOT NULL,
    expires_at              timestamptz NOT NULL,
    CHECK (expires_at > issued_at),
    UNIQUE (tenant_id, source_uri, event_id)
);

CREATE TABLE IF NOT EXISTS sfeb.source_receipt_revocations (
    receipt_id             text PRIMARY KEY REFERENCES sfeb.source_receipts(receipt_id),
    tenant_id              text NOT NULL,
    revoked_at              timestamptz NOT NULL,
    reason                  text NOT NULL,
    authority_id            text NOT NULL,
    authority_receipt_sha256 bytea NOT NULL CHECK (octet_length(authority_receipt_sha256) = 32)
);
-- Receipt rows and revocations are append-only in the application contract.
-- Production should grant INSERT/SELECT only to the intake role and deny
-- UPDATE/DELETE; stronger guarantees require separate database roles/ownership.

CREATE TABLE IF NOT EXISTS sfeb.resources (
    tenant_id              text NOT NULL,
    resource_id            text NOT NULL,
    state                  text NOT NULL CHECK (state IN ('available','pending_delivery','reserved')),
    version                bigint NOT NULL DEFAULT 0 CHECK (version >= 0),
    PRIMARY KEY (tenant_id, resource_id)
);

CREATE TABLE IF NOT EXISTS sfeb.operations (
    tenant_id              text NOT NULL,
    objective_id           text NOT NULL,
    action_key             text NOT NULL,
    request_digest         bytea NOT NULL CHECK (octet_length(request_digest) = 32),
    operation_id           uuid NOT NULL UNIQUE,
    objective_version      bigint NOT NULL CHECK (objective_version > 0),
    receipt_id             text NOT NULL REFERENCES sfeb.source_receipts(receipt_id),
    objective_fencing_epoch bigint NOT NULL CHECK (objective_fencing_epoch > 0),
    permission_revision    bigint NOT NULL CHECK (permission_revision > 0),
    status                 text NOT NULL CHECK (status IN ('prepared','outcome_unknown','remote_committed','rejected')),
    created_at             timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, action_key),
    FOREIGN KEY (tenant_id, objective_id) REFERENCES sfeb.objective_fences(tenant_id, objective_id)
);

CREATE TABLE IF NOT EXISTS sfeb.outbox (
    tenant_id              text NOT NULL,
    event_id               uuid NOT NULL,
    objective_id           text NOT NULL,
    action_key             text NOT NULL,
    operation_id           uuid NOT NULL REFERENCES sfeb.operations(operation_id),
    request_digest         bytea NOT NULL CHECK (octet_length(request_digest) = 32),
    resource_id            text NOT NULL,
    aggregate_key          text NOT NULL,
    aggregate_sequence     bigint NOT NULL CHECK (aggregate_sequence > 0),
    objective_fencing_epoch bigint NOT NULL CHECK (objective_fencing_epoch > 0),
    permission_revision    bigint NOT NULL CHECK (permission_revision > 0),
    payload                 jsonb NOT NULL,
    status                  text NOT NULL CHECK (status IN ('pending','leased','delivered','quarantined')),
    remote_outcome          text NOT NULL DEFAULT 'not_attempted'
                                CHECK (remote_outcome IN ('not_attempted','unknown','committed','rejected')),
    attempts                integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    claim_epoch             bigint NOT NULL DEFAULT 0 CHECK (claim_epoch >= 0),
    claim_token             uuid,
    claimed_by              text,
    claim_expires_at        timestamptz,
    created_at              timestamptz NOT NULL DEFAULT clock_timestamp(),
    delivered_at            timestamptz,
    postcondition_sha256    bytea,
    PRIMARY KEY (tenant_id, event_id),
    UNIQUE (tenant_id, action_key),
    UNIQUE (tenant_id, aggregate_key, aggregate_sequence),
    FOREIGN KEY (tenant_id, action_key) REFERENCES sfeb.operations(tenant_id, action_key),
    CHECK (
      (status = 'leased' AND claim_token IS NOT NULL AND claimed_by IS NOT NULL AND claim_expires_at IS NOT NULL)
      OR
      (status <> 'leased' AND claim_token IS NULL AND claimed_by IS NULL AND claim_expires_at IS NULL)
    ),
    CHECK (status <> 'delivered' OR remote_outcome = 'committed'),
    CHECK (remote_outcome <> 'committed' OR (postcondition_sha256 IS NOT NULL AND octet_length(postcondition_sha256) = 32))
);

CREATE INDEX IF NOT EXISTS sfeb_outbox_pending_idx
    ON sfeb.outbox (created_at, tenant_id, event_id)
    WHERE status IN ('pending','leased');

CREATE INDEX IF NOT EXISTS sfeb_outbox_aggregate_order_idx
    ON sfeb.outbox (tenant_id, aggregate_key, aggregate_sequence, status);

CREATE TABLE IF NOT EXISTS sfeb.evidence_chain_heads (
    tenant_id              text PRIMARY KEY,
    sequence               bigint NOT NULL DEFAULT 0 CHECK (sequence >= 0),
    head_sha256             bytea NOT NULL CHECK (octet_length(head_sha256) = 32)
);

CREATE TABLE IF NOT EXISTS sfeb.evidence_events (
    tenant_id              text NOT NULL,
    sequence               bigint NOT NULL CHECK (sequence > 0),
    subject_id              text NOT NULL,
    event_type              text NOT NULL,
    occurred_at             timestamptz NOT NULL,
    payload_jcs             bytea NOT NULL,
    previous_sha256         bytea NOT NULL CHECK (octet_length(previous_sha256) = 32),
    event_sha256            bytea NOT NULL CHECK (octet_length(event_sha256) = 32),
    PRIMARY KEY (tenant_id, sequence),
    UNIQUE (tenant_id, event_sha256)
);
-- Append protocol: SELECT the tenant's evidence_chain_heads row FOR UPDATE,
-- canonicalize the event payload with RFC 8785 outside SQL, compute the
-- event hash over a documented, domain-separated byte frame, insert sequence
-- + previous hash + new hash, and update the chain head in the same transaction.
-- Do not use JSONB text rendering as a cross-language canonicalization contract.
-- External signed checkpoints are required to detect privileged whole-DB rewrite.

-- Atomic outbox claim protocol.
-- Run this as a short transaction; never hold row locks while calling a network API.
-- $1 = batch size; $2 = worker identity; $3 = fresh claim UUID; $4 = lease interval.
WITH eligible AS (
    SELECT o.tenant_id, o.event_id
    FROM sfeb.outbox AS o
    WHERE (
        o.status = 'pending'
        OR (o.status = 'leased' AND o.claim_expires_at <= clock_timestamp())
    )
    -- Do not claim a later item in an aggregate while an earlier item is
    -- pending, leased, quarantined, or not confirmed delivered.
    AND NOT EXISTS (
        SELECT 1
        FROM sfeb.outbox AS prior
        WHERE prior.tenant_id = o.tenant_id
          AND prior.aggregate_key = o.aggregate_key
          AND prior.aggregate_sequence < o.aggregate_sequence
          AND prior.status <> 'delivered'
    )
    ORDER BY o.created_at, o.tenant_id, o.event_id
    LIMIT $1
    FOR UPDATE OF o SKIP LOCKED
)
UPDATE sfeb.outbox AS o
SET status = 'leased',
    claim_epoch = o.claim_epoch + 1,
    claim_token = $3::uuid,
    claimed_by = $2,
    claim_expires_at = clock_timestamp() + $4::interval,
    attempts = o.attempts + 1,
    remote_outcome = CASE
        WHEN o.remote_outcome = 'not_attempted' THEN 'unknown'
        ELSE o.remote_outcome
    END
FROM eligible AS e
WHERE o.tenant_id = e.tenant_id
  AND o.event_id = e.event_id
RETURNING o.*;

-- After the separate sink has committed AND independent readback has
-- verified the operation key, request digest and postcondition, complete
-- only if this worker still owns the exact claim lease.
-- $1 tenant; $2 event; $3 worker; $4 claim epoch; $5 claim token; $6 postcondition digest.
UPDATE sfeb.outbox
SET status = 'delivered',
    remote_outcome = 'committed',
    delivered_at = clock_timestamp(),
    postcondition_sha256 = $6,
    claimed_by = NULL,
    claim_token = NULL,
    claim_expires_at = NULL
WHERE tenant_id = $1
  AND event_id = $2
  AND status = 'leased'
  AND claimed_by = $3
  AND claim_epoch = $4
  AND claim_token = $5
  AND claim_expires_at > clock_timestamp()
RETURNING tenant_id, event_id, action_key, operation_id;

-- If completion returns no row, this worker lost the claim. It MUST NOT
-- mint a new action key or infer failure. A later owner reconciles the same
-- idempotency key with the external sink. The remote sink must itself enforce
-- tenant + action_key + request_digest uniqueness, current permission and
-- objective fence at mutation time; this local PostgreSQL lock cannot protect
-- an unrelated remote service.

-- Release a claim only after a known pre-commit failure. For a timeout or
-- lost acknowledgement keep remote_outcome='unknown'. The claim may expire
-- and be re-acquired, but the action key and request digest never change.
-- $1 tenant; $2 event; $3 worker; $4 epoch; $5 token.
UPDATE sfeb.outbox
SET status = 'pending',
    claimed_by = NULL,
    claim_token = NULL,
    claim_expires_at = NULL
WHERE tenant_id = $1
  AND event_id = $2
  AND status = 'leased'
  AND claimed_by = $3
  AND claim_epoch = $4
  AND claim_token = $5
  AND remote_outcome = 'not_attempted'
RETURNING event_id;

-- Contract invariant:
-- Outbox claim_epoch/token fences only delivery workers. objective_fencing_epoch
-- is a distinct, tenant/objective-scoped authority token that must be checked by
-- the external mutation sink. SKIP LOCKED is for this queue only; PostgreSQL
-- documents that it yields an inconsistent view and is unsuitable for general
-- purpose reads.

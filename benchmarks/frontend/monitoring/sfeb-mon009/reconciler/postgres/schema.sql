-- MON-009 PostgreSQL conditional-mutation reference sink.
-- Apply only to a disposable database. This is a test profile, not production DDL.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
DROP SCHEMA IF EXISTS sfeb CASCADE;
CREATE SCHEMA sfeb;

DO $$ BEGIN CREATE ROLE sfeb_writer NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE sfeb_observer NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
ALTER ROLE sfeb_writer NOLOGIN;
ALTER ROLE sfeb_observer NOLOGIN;

CREATE TABLE sfeb.resource_state (
    tenant_id text NOT NULL,
    resource_id text NOT NULL,
    state_payload jsonb NOT NULL,
    revision bigint NOT NULL CHECK (revision >= 0),
    state_digest text NOT NULL CHECK (state_digest ~ '^[0-9a-f]{64}$'),
    accepted_authority_revision bigint NOT NULL CHECK (accepted_authority_revision >= 0),
    authority_updated_at timestamptz NOT NULL,
    authority_valid_until timestamptz NOT NULL,
    authority_max_age_seconds bigint NOT NULL CHECK (authority_max_age_seconds > 0),
    fencing_high_water bigint NOT NULL CHECK (fencing_high_water >= 0),
    event_sequence bigint NOT NULL CHECK (event_sequence >= 0),
    PRIMARY KEY (tenant_id, resource_id)
);

-- One immutable outcome per logical operation identity. Replays with the same
-- immutable binding return this original outcome; changed bindings conflict.
CREATE TABLE sfeb.operation_ledger (
    tenant_id text NOT NULL,
    operation_id uuid NOT NULL,
    resource_id text NOT NULL,
    request_digest text NOT NULL CHECK (request_digest ~ '^[0-9a-f]{64}$'),
    authority_revision bigint NOT NULL,
    fence_token bigint NOT NULL,
    expected_prior_revision bigint NOT NULL,
    outcome text NOT NULL CHECK (outcome IN ('committed', 'rejected')),
    rejection_reason text,
    event_sequence bigint NOT NULL,
    prior_revision bigint NOT NULL,
    result_revision bigint NOT NULL,
    prior_digest text NOT NULL CHECK (prior_digest ~ '^[0-9a-f]{64}$'),
    result_digest text NOT NULL CHECK (result_digest ~ '^[0-9a-f]{64}$'),
    committed_at timestamptz NOT NULL,
    receipt_payload jsonb NOT NULL,
    PRIMARY KEY (tenant_id, operation_id),
    UNIQUE (tenant_id, resource_id, event_sequence)
);

-- Every invocation reaching the sink procedure is recorded separately from
-- the one durable logical operation outcome. This prevents retries, blocks,
-- refusals and identity-conflict attempts from being mistaken for extra effects.
CREATE TABLE sfeb.attempt_ledger (
    attempt_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id text,
    resource_id text,
    operation_id uuid,
    request_digest text CHECK (request_digest IS NULL OR request_digest ~ '^[0-9a-f]{64}$'),
    outcome text NOT NULL,
    event_sequence bigint,
    session_principal text NOT NULL,
    active_role text NOT NULL,
    attempted_at timestamptz NOT NULL
);

CREATE FUNCTION sfeb.reject_attempt_mutation() RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
BEGIN
    RAISE EXCEPTION 'MON-009 attempt ledger is append-only';
END;
$$;
CREATE TRIGGER attempt_ledger_append_only
BEFORE UPDATE OR DELETE ON sfeb.attempt_ledger
FOR EACH ROW EXECUTE FUNCTION sfeb.reject_attempt_mutation();

CREATE FUNCTION sfeb.reject_ledger_mutation() RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
BEGIN
    RAISE EXCEPTION 'MON-009 operation ledger is append-only';
END;
$$;
CREATE TRIGGER operation_ledger_append_only
BEFORE UPDATE OR DELETE ON sfeb.operation_ledger
FOR EACH ROW EXECUTE FUNCTION sfeb.reject_ledger_mutation();

-- Mutation API:
-- * request_canonical is the exact RFC 8785/JCS byte sequence supplied by the
--   caller; the sink hashes those exact bytes using pgcrypto.
-- * request_payload must parse to the same JSON value as those bytes.
-- * next_state is the requested destination state. This generic test sink
--   enforces a conditional set, not domain-specific business correctness.
-- * receipt_payload is unsigned source evidence derived by the destination.
--   A separate signer/SCITT adapter must authenticate it after commit.
CREATE FUNCTION sfeb.apply_action(
    p_tenant_id text,
    p_resource_id text,
    p_operation_id uuid,
    p_request_canonical bytea,
    p_request_payload jsonb,
    p_next_state jsonb,
    p_authority_revision bigint,
    p_fence_token bigint,
    p_expected_revision bigint
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
DECLARE
    v_existing sfeb.operation_ledger%ROWTYPE;
    v_state sfeb.resource_state%ROWTYPE;
    v_request_digest text;
    v_prior_digest text;
    v_result_digest text;
    v_prior_revision bigint;
    v_result_revision bigint;
    v_next_sequence bigint;
    v_now timestamptz := clock_timestamp();
    v_outcome text := 'committed';
    v_rejection text := NULL;
    v_receipt jsonb;
    v_parsed jsonb;
    v_attempt_id bigint;
    v_role text := COALESCE(NULLIF(current_setting('role', true), 'none'), current_user);
BEGIN
    IF p_tenant_id IS NULL OR p_tenant_id = '' OR
       p_resource_id IS NULL OR p_resource_id = '' OR
       p_operation_id IS NULL OR p_request_canonical IS NULL OR
       p_request_payload IS NULL OR p_next_state IS NULL OR
       p_authority_revision IS NULL OR p_fence_token IS NULL OR p_expected_revision IS NULL OR
       p_authority_revision < 0 OR p_fence_token < 0 OR p_expected_revision < 0 THEN
        INSERT INTO sfeb.attempt_ledger (tenant_id, resource_id, operation_id, outcome, session_principal, active_role, attempted_at)
        VALUES (p_tenant_id, p_resource_id, p_operation_id, 'invalid_input', session_user, v_role, v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status', 'invalid_input', 'attempt_id', v_attempt_id);
    END IF;

    v_request_digest := encode(public.digest(p_request_canonical, 'sha256'), 'hex');
    BEGIN
        v_parsed := convert_from(p_request_canonical, 'UTF8')::jsonb;
    EXCEPTION WHEN OTHERS THEN
        INSERT INTO sfeb.attempt_ledger (tenant_id, resource_id, operation_id, request_digest, outcome, session_principal, active_role, attempted_at)
        VALUES (p_tenant_id, p_resource_id, p_operation_id, v_request_digest, 'invalid_request_encoding', session_user, v_role, v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status', 'invalid_request_encoding', 'attempt_id', v_attempt_id);
    END;
    IF v_parsed IS DISTINCT FROM p_request_payload
       OR p_request_payload->'next_state' IS DISTINCT FROM p_next_state THEN
        INSERT INTO sfeb.attempt_ledger (tenant_id, resource_id, operation_id, request_digest, outcome, session_principal, active_role, attempted_at)
        VALUES (p_tenant_id, p_resource_id, p_operation_id, v_request_digest, 'request_payload_mismatch', session_user, v_role, v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status', 'request_payload_mismatch', 'attempt_id', v_attempt_id);
    END IF;

    v_result_digest := encode(public.digest(convert_to(p_next_state::text, 'UTF8'), 'sha256'), 'hex');

    -- Serialize all attempts sharing a tenant-scoped operation ID, including
    -- conflicting attempts that target different resources. Hash collisions
    -- only add serialization; the ledger primary key remains authoritative.
    PERFORM pg_advisory_xact_lock(hashtextextended(p_tenant_id || ':' || p_operation_id::text, 0));

    SELECT * INTO v_existing
    FROM sfeb.operation_ledger
    WHERE tenant_id = p_tenant_id AND operation_id = p_operation_id;
    IF FOUND THEN
        IF v_existing.resource_id = p_resource_id
           AND v_existing.request_digest = v_request_digest
           AND v_existing.authority_revision = p_authority_revision
           AND v_existing.fence_token = p_fence_token
           AND v_existing.expected_prior_revision = p_expected_revision THEN
            INSERT INTO sfeb.attempt_ledger (tenant_id, resource_id, operation_id, request_digest, outcome, event_sequence, session_principal, active_role, attempted_at)
            VALUES (p_tenant_id, p_resource_id, p_operation_id, v_request_digest, 'replayed', v_existing.event_sequence, session_user, v_role, v_now)
            RETURNING attempt_id INTO v_attempt_id;
            RETURN jsonb_build_object('status', 'replayed', 'attempt_id', v_attempt_id, 'receipt', v_existing.receipt_payload);
        END IF;
        INSERT INTO sfeb.attempt_ledger (tenant_id, resource_id, operation_id, request_digest, outcome, event_sequence, session_principal, active_role, attempted_at)
        VALUES (p_tenant_id, p_resource_id, p_operation_id, v_request_digest, 'identity_conflict', v_existing.event_sequence, session_user, v_role, v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status', 'identity_conflict', 'attempt_id', v_attempt_id);
    END IF;

    SELECT * INTO v_state
    FROM sfeb.resource_state
    WHERE tenant_id = p_tenant_id AND resource_id = p_resource_id
    FOR UPDATE;
    IF NOT FOUND THEN
        INSERT INTO sfeb.attempt_ledger (tenant_id, resource_id, operation_id, request_digest, outcome, session_principal, active_role, attempted_at)
        VALUES (p_tenant_id, p_resource_id, p_operation_id, v_request_digest, 'unknown_resource', session_user, v_role, v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status', 'unknown_resource', 'attempt_id', v_attempt_id);
    END IF;

    v_prior_revision := v_state.revision;
    v_prior_digest := v_state.state_digest;
    v_result_revision := v_prior_revision;

    IF p_expected_revision <> v_state.revision THEN
        v_outcome := 'rejected';
        v_rejection := 'revision_conflict';
    ELSIF p_authority_revision <> v_state.accepted_authority_revision
       OR v_now > v_state.authority_valid_until
       OR v_state.authority_updated_at > v_now
       OR v_now - v_state.authority_updated_at > make_interval(secs => v_state.authority_max_age_seconds::double precision) THEN
        v_outcome := 'rejected';
        v_rejection := 'stale_or_invalid_authority';
    ELSIF p_fence_token < v_state.fencing_high_water THEN
        v_outcome := 'rejected';
        v_rejection := 'stale_fence';
    ELSE
        v_result_revision := v_prior_revision + 1;
    END IF;

    IF v_state.event_sequence = 9223372036854775807 THEN
        RAISE EXCEPTION 'event sequence exhausted';
    END IF;
    v_next_sequence := v_state.event_sequence + 1;

    IF v_outcome = 'committed' THEN
        UPDATE sfeb.resource_state
        SET state_payload = p_next_state,
            revision = v_result_revision,
            state_digest = v_result_digest,
            fencing_high_water = GREATEST(fencing_high_water, p_fence_token),
            event_sequence = v_next_sequence
        WHERE tenant_id = p_tenant_id AND resource_id = p_resource_id;
    ELSE
        UPDATE sfeb.resource_state
        SET event_sequence = v_next_sequence
        WHERE tenant_id = p_tenant_id AND resource_id = p_resource_id;
        v_result_digest := v_prior_digest;
    END IF;

    v_receipt := jsonb_build_object(
        'profile', 'sfeb-postgres-sink-v1',
        'destination', 'sfeb-postgres-test-sink-v1',
        'tenant_id', p_tenant_id,
        'resource_id', p_resource_id,
        'operation_id', p_operation_id::text,
        'request_digest', v_request_digest,
        'request_digest_algorithm', 'SHA-256',
        'request_encoding_profile', 'RFC8785-validated-by-adapter',
        'result_digest_profile', 'postgres-jsonb-text-v1',
        'authority_revision', p_authority_revision,
        'fence_token', p_fence_token,
        'expected_prior_revision', p_expected_revision,
        'prior_revision', v_prior_revision,
        'result_revision', v_result_revision,
        'prior_digest', v_prior_digest,
        'result_digest', v_result_digest,
        'outcome', v_outcome,
        'rejection_reason', v_rejection,
        'event_sequence', v_next_sequence,
        'committed_at', CASE WHEN v_outcome = 'committed' THEN to_jsonb(v_now) ELSE 'null'::jsonb END
    );

    INSERT INTO sfeb.operation_ledger (
        tenant_id, operation_id, resource_id, request_digest,
        authority_revision, fence_token, expected_prior_revision,
        outcome, rejection_reason, event_sequence,
        prior_revision, result_revision, prior_digest, result_digest,
        committed_at, receipt_payload
    ) VALUES (
        p_tenant_id, p_operation_id, p_resource_id, v_request_digest,
        p_authority_revision, p_fence_token, p_expected_revision,
        v_outcome, v_rejection, v_next_sequence,
        v_prior_revision, v_result_revision, v_prior_digest, v_result_digest,
        v_now, v_receipt
    );

    INSERT INTO sfeb.attempt_ledger (tenant_id, resource_id, operation_id, request_digest, outcome, event_sequence, session_principal, active_role, attempted_at)
    VALUES (p_tenant_id, p_resource_id, p_operation_id, v_request_digest, v_outcome, v_next_sequence, session_user, v_role, v_now)
    RETURNING attempt_id INTO v_attempt_id;

    RETURN jsonb_build_object('status', v_outcome, 'attempt_id', v_attempt_id, 'receipt', v_receipt);
END;
$$;

CREATE VIEW sfeb.observer_readback WITH (security_barrier = true) AS
SELECT r.tenant_id, r.resource_id, r.revision, r.state_digest,
       r.state_payload, r.event_sequence, r.fencing_high_water,
       r.accepted_authority_revision
FROM sfeb.resource_state AS r;

REVOKE ALL ON ALL TABLES IN SCHEMA sfeb FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA sfeb FROM PUBLIC;
GRANT USAGE ON SCHEMA sfeb TO sfeb_writer, sfeb_observer;
GRANT EXECUTE ON FUNCTION sfeb.apply_action(text,text,uuid,bytea,jsonb,jsonb,bigint,bigint,bigint) TO sfeb_writer;
GRANT SELECT ON sfeb.observer_readback TO sfeb_observer;

COMMENT ON SCHEMA sfeb IS 'MON-009 reversible test sink; privileged database administrators remain outside this boundary.';
COMMENT ON FUNCTION sfeb.apply_action(text,text,uuid,bytea,jsonb,jsonb,bigint,bigint,bigint) IS
'Atomic idempotent conditional state set. Returns an unsigned destination-derived receipt payload; it is not a signature, SCITT receipt, or physical-effect claim.';

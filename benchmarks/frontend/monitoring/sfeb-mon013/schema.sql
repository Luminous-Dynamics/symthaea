-- MON-013 PostgreSQL trust-boundary reference profile v1.
-- Disposable database only. Run as a DBA/superuser in an EMPTY database.
-- Requires pgcrypto. This is a reference qualification target, not production DDL.
\set ON_ERROR_STOP on
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $roles$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'sfeb_owner') THEN
        CREATE ROLE sfeb_owner NOLOGIN NOSUPERUSER NOBYPASSRLS;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'sfeb_writer') THEN
        CREATE ROLE sfeb_writer NOLOGIN NOSUPERUSER NOBYPASSRLS;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'sfeb_observer') THEN
        CREATE ROLE sfeb_observer NOLOGIN NOSUPERUSER NOBYPASSRLS;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'sfeb_grant_admin') THEN
        CREATE ROLE sfeb_grant_admin NOLOGIN NOSUPERUSER NOBYPASSRLS;
    END IF;
END
$roles$;

ALTER ROLE sfeb_owner NOLOGIN NOSUPERUSER NOBYPASSRLS;
ALTER ROLE sfeb_writer NOLOGIN NOSUPERUSER NOBYPASSRLS;
ALTER ROLE sfeb_observer NOLOGIN NOSUPERUSER NOBYPASSRLS;
ALTER ROLE sfeb_grant_admin NOLOGIN NOSUPERUSER NOBYPASSRLS;

CREATE SCHEMA sfeb AUTHORIZATION sfeb_owner;
SET ROLE sfeb_owner;

CREATE TABLE sfeb.destination_config (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    destination_id text NOT NULL UNIQUE
);
INSERT INTO sfeb.destination_config(singleton, destination_id)
VALUES (true, 'sfeb-postgres-reference-destination-v1');

CREATE TABLE sfeb.principal_scope_grant (
    grant_id uuid PRIMARY KEY DEFAULT public.gen_random_uuid(),
    db_principal name NOT NULL,
    principal_id text NOT NULL,
    tenant_id text NOT NULL,
    resource_id text NOT NULL,
    action text NOT NULL CHECK (action IN ('mutate', 'readback')),
    policy_digest text NOT NULL CHECK (policy_digest ~ '^[0-9a-f]{64}$'),
    authority_digest text NOT NULL CHECK (authority_digest ~ '^[0-9a-f]{64}$'),
    authority_revision bigint NOT NULL CHECK (authority_revision >= 0),
    issuer text NOT NULL,
    issuer_key_id text NOT NULL,
    evidence_digest text NOT NULL CHECK (evidence_digest ~ '^[0-9a-f]{64}$'),
    verified_at timestamptz NOT NULL,
    not_before timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    max_age interval NOT NULL CHECK (max_age > interval '0 seconds'),
    revoked_at timestamptz,
    CHECK (expires_at > not_before)
);
CREATE INDEX principal_scope_grant_lookup
ON sfeb.principal_scope_grant
(db_principal, tenant_id, resource_id, action, policy_digest, authority_digest);

CREATE TABLE sfeb.resource_state (
    tenant_id text NOT NULL,
    resource_id text NOT NULL,
    state_payload jsonb NOT NULL,
    revision bigint NOT NULL CHECK (revision >= 0),
    state_digest text NOT NULL CHECK (state_digest ~ '^[0-9a-f]{64}$'),
    accepted_policy_digest text NOT NULL CHECK (accepted_policy_digest ~ '^[0-9a-f]{64}$'),
    accepted_authority_digest text NOT NULL CHECK (accepted_authority_digest ~ '^[0-9a-f]{64}$'),
    accepted_authority_revision bigint NOT NULL CHECK (accepted_authority_revision >= 0),
    authority_updated_at timestamptz NOT NULL,
    authority_valid_until timestamptz NOT NULL,
    authority_max_age interval NOT NULL CHECK (authority_max_age > interval '0 seconds'),
    fencing_high_water bigint NOT NULL CHECK (fencing_high_water >= 0),
    event_sequence bigint NOT NULL CHECK (event_sequence >= 0),
    PRIMARY KEY (tenant_id, resource_id)
);

CREATE TABLE sfeb.state_history (
    tenant_id text NOT NULL,
    resource_id text NOT NULL,
    revision bigint NOT NULL CHECK (revision >= 0),
    state_digest text NOT NULL CHECK (state_digest ~ '^[0-9a-f]{64}$'),
    state_payload jsonb NOT NULL,
    operation_id uuid,
    event_sequence bigint NOT NULL CHECK (event_sequence >= 0),
    recorded_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, resource_id, revision),
    FOREIGN KEY (tenant_id, resource_id)
        REFERENCES sfeb.resource_state(tenant_id, resource_id)
);

-- One configured destination per database. A multi-destination deployment
-- uses the composite (destination_id, operation_id) identity unchanged.
CREATE TABLE sfeb.operation_ledger (
    destination_id text NOT NULL,
    operation_id uuid NOT NULL,
    principal_id text NOT NULL,
    db_principal name NOT NULL,
    tenant_id text NOT NULL,
    resource_id text NOT NULL,
    action text NOT NULL,
    request_digest text NOT NULL CHECK (request_digest ~ '^[0-9a-f]{64}$'),
    policy_digest text NOT NULL CHECK (policy_digest ~ '^[0-9a-f]{64}$'),
    authority_digest text NOT NULL CHECK (authority_digest ~ '^[0-9a-f]{64}$'),
    authority_revision bigint NOT NULL,
    fence_token bigint NOT NULL,
    expected_prior_revision bigint NOT NULL,
    outcome text NOT NULL CHECK (outcome IN ('committed', 'refused')),
    rejection_reason text,
    event_sequence bigint NOT NULL,
    prior_revision bigint NOT NULL,
    result_revision bigint NOT NULL,
    prior_digest text NOT NULL CHECK (prior_digest ~ '^[0-9a-f]{64}$'),
    result_digest text NOT NULL CHECK (result_digest ~ '^[0-9a-f]{64}$'),
    committed_at timestamptz,
    receipt_payload jsonb NOT NULL,
    PRIMARY KEY (destination_id, operation_id)
);

CREATE TABLE sfeb.attempt_ledger (
    attempt_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    destination_id text,
    operation_id uuid,
    db_principal name NOT NULL,
    tenant_id text,
    resource_id text,
    request_digest text,
    outcome text NOT NULL,
    detail text,
    attempted_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE FUNCTION sfeb.reject_ledger_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
BEGIN
    RAISE EXCEPTION 'MON-013 evidence ledger is append-only';
END;
$$;

CREATE TRIGGER operation_ledger_append_only
BEFORE UPDATE OR DELETE ON sfeb.operation_ledger
FOR EACH ROW EXECUTE FUNCTION sfeb.reject_ledger_mutation();
CREATE TRIGGER attempt_ledger_append_only
BEFORE UPDATE OR DELETE ON sfeb.attempt_ledger
FOR EACH ROW EXECUTE FUNCTION sfeb.reject_ledger_mutation();
CREATE TRIGGER state_history_append_only
BEFORE UPDATE OR DELETE ON sfeb.state_history
FOR EACH ROW EXECUTE FUNCTION sfeb.reject_ledger_mutation();

-- Used by RLS. Identity is session_user, never an application-set GUC.
CREATE FUNCTION sfeb._has_scope(
    p_tenant_id text, p_resource_id text, p_action text
) RETURNS boolean
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
DECLARE
    v_now timestamptz := clock_timestamp();
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles r
        WHERE r.rolname = session_user
          AND (r.rolsuper OR r.rolbypassrls)
    ) THEN
        RETURN false;
    END IF;

    RETURN EXISTS (
        SELECT 1
        FROM sfeb.principal_scope_grant g
        WHERE g.db_principal = session_user
          AND g.tenant_id = p_tenant_id
          AND g.resource_id = p_resource_id
          AND g.action = p_action
          AND g.revoked_at IS NULL
          AND v_now >= g.not_before
          AND v_now < g.expires_at
          AND v_now >= g.verified_at
          AND v_now - g.verified_at <= g.max_age
    );
END;
$$;

-- The mutation wrapper locks this grant FOR SHARE. Concurrent revocation
-- therefore serializes before or after the transaction containing the commit.
CREATE FUNCTION sfeb._lock_scope_grant(
    p_tenant_id text,
    p_resource_id text,
    p_action text,
    p_policy_digest text,
    p_authority_digest text,
    p_authority_revision bigint
) RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
DECLARE
    v_grant_id uuid;
    v_now timestamptz := clock_timestamp();
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles r
        WHERE r.rolname = session_user
          AND (r.rolsuper OR r.rolbypassrls)
    ) THEN
        RETURN NULL;
    END IF;

    SELECT g.grant_id
      INTO v_grant_id
      FROM sfeb.principal_scope_grant g
     WHERE g.db_principal = session_user
       AND g.tenant_id = p_tenant_id
       AND g.resource_id = p_resource_id
       AND g.action = p_action
       AND g.policy_digest = p_policy_digest
       AND g.authority_digest = p_authority_digest
       AND g.authority_revision = p_authority_revision
       AND g.revoked_at IS NULL
       AND v_now >= g.not_before
       AND v_now < g.expires_at
       AND v_now >= g.verified_at
       AND v_now - g.verified_at <= g.max_age
     ORDER BY g.verified_at DESC
     LIMIT 1
     FOR SHARE;

    RETURN v_grant_id;
END;
$$;

ALTER TABLE sfeb.resource_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE sfeb.resource_state FORCE ROW LEVEL SECURITY;
ALTER TABLE sfeb.state_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE sfeb.state_history FORCE ROW LEVEL SECURITY;

CREATE POLICY resource_owner_select ON sfeb.resource_state
FOR SELECT TO sfeb_owner
USING (sfeb._has_scope(tenant_id, resource_id, 'mutate')
    OR sfeb._has_scope(tenant_id, resource_id, 'readback'));
CREATE POLICY resource_owner_update ON sfeb.resource_state
FOR UPDATE TO sfeb_owner
USING (sfeb._has_scope(tenant_id, resource_id, 'mutate'))
WITH CHECK (sfeb._has_scope(tenant_id, resource_id, 'mutate'));
CREATE POLICY resource_observer_select ON sfeb.resource_state
FOR SELECT TO sfeb_observer
USING (sfeb._has_scope(tenant_id, resource_id, 'readback'));

CREATE POLICY history_owner_select ON sfeb.state_history
FOR SELECT TO sfeb_owner
USING (sfeb._has_scope(tenant_id, resource_id, 'readback'));
CREATE POLICY history_owner_insert ON sfeb.state_history
FOR INSERT TO sfeb_owner
WITH CHECK (sfeb._has_scope(tenant_id, resource_id, 'mutate'));
CREATE POLICY history_observer_select ON sfeb.state_history
FOR SELECT TO sfeb_observer
USING (sfeb._has_scope(tenant_id, resource_id, 'readback'));

CREATE FUNCTION sfeb.apply_action_authorized(
    p_tenant_id text,
    p_resource_id text,
    p_operation_id uuid,
    p_request_canonical bytea,
    p_request_payload jsonb,
    p_next_state jsonb,
    p_policy_digest text,
    p_authority_digest text,
    p_authority_revision bigint,
    p_fence_token bigint,
    p_expected_revision bigint
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
DECLARE
    v_destination text;
    v_principal text;
    v_grant_id uuid;
    v_existing sfeb.operation_ledger%ROWTYPE;
    v_state sfeb.resource_state%ROWTYPE;
    v_request_digest text;
    v_result_digest text;
    v_prior_revision bigint;
    v_result_revision bigint;
    v_event_sequence bigint;
    v_now timestamptz := clock_timestamp();
    v_outcome text := 'committed';
    v_rejection text := NULL;
    v_receipt jsonb;
    v_parsed jsonb;
    v_attempt_id bigint;
BEGIN
    SELECT destination_id INTO STRICT v_destination
      FROM sfeb.destination_config WHERE singleton;

    IF p_tenant_id IS NULL OR p_resource_id IS NULL OR p_operation_id IS NULL
       OR p_request_canonical IS NULL OR p_request_payload IS NULL
       OR p_next_state IS NULL OR p_policy_digest IS NULL OR p_authority_digest IS NULL
       OR p_authority_revision IS NULL OR p_fence_token IS NULL OR p_expected_revision IS NULL
       OR p_authority_revision < 0 OR p_fence_token < 0 OR p_expected_revision < 0
       OR p_policy_digest !~ '^[0-9a-f]{64}$'
       OR p_authority_digest !~ '^[0-9a-f]{64}$' THEN
        INSERT INTO sfeb.attempt_ledger
            (destination_id, operation_id, db_principal, tenant_id, resource_id, outcome, detail, attempted_at)
        VALUES
            (v_destination, p_operation_id, session_user, p_tenant_id, p_resource_id,
             'invalid_input', 'required field or digest malformed', v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status','invalid_input','attempt_id',v_attempt_id);
    END IF;

    v_request_digest := encode(public.digest(p_request_canonical, 'sha256'), 'hex');
    BEGIN
        v_parsed := convert_from(p_request_canonical, 'UTF8')::jsonb;
    EXCEPTION WHEN OTHERS THEN
        INSERT INTO sfeb.attempt_ledger
            (destination_id, operation_id, db_principal, tenant_id, resource_id, request_digest, outcome, detail, attempted_at)
        VALUES
            (v_destination, p_operation_id, session_user, p_tenant_id, p_resource_id,
             v_request_digest, 'invalid_request_encoding', 'request is not UTF-8 JSON', v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status','invalid_request_encoding','attempt_id',v_attempt_id);
    END;

    IF v_parsed IS DISTINCT FROM p_request_payload
       OR p_request_payload->>'tenant_id' IS DISTINCT FROM p_tenant_id
       OR p_request_payload->>'resource_id' IS DISTINCT FROM p_resource_id
       OR p_request_payload->>'operation_id' IS DISTINCT FROM p_operation_id::text
       OR p_request_payload->>'policy_digest' IS DISTINCT FROM p_policy_digest
       OR p_request_payload->>'authority_digest' IS DISTINCT FROM p_authority_digest
       OR p_request_payload->>'action' IS DISTINCT FROM 'set_state'
       OR p_request_payload->'next_state' IS DISTINCT FROM p_next_state
       OR p_request_payload->>'authority_revision' IS DISTINCT FROM p_authority_revision::text
       OR p_request_payload->>'fence_token' IS DISTINCT FROM p_fence_token::text
       OR p_request_payload->>'expected_revision' IS DISTINCT FROM p_expected_revision::text THEN
        INSERT INTO sfeb.attempt_ledger
            (destination_id, operation_id, db_principal, tenant_id, resource_id, request_digest, outcome, detail, attempted_at)
        VALUES
            (v_destination, p_operation_id, session_user, p_tenant_id, p_resource_id,
             v_request_digest, 'request_binding_mismatch', 'request bytes do not bind supplied mutation arguments', v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status','request_binding_mismatch','attempt_id',v_attempt_id);
    END IF;

    -- Caller adapters must validate the selected canonicalization profile
    -- before invoking this function. PostgreSQL jsonb parsing is not JCS.
    v_grant_id := sfeb._lock_scope_grant(
        p_tenant_id, p_resource_id, 'mutate',
        p_policy_digest, p_authority_digest, p_authority_revision
    );
    IF v_grant_id IS NULL THEN
        INSERT INTO sfeb.attempt_ledger
            (destination_id, operation_id, db_principal, tenant_id, resource_id, request_digest, outcome, detail, attempted_at)
        VALUES
            (v_destination, p_operation_id, session_user, p_tenant_id, p_resource_id,
             v_request_digest, 'unauthorized', 'no current authenticated session scope grant', v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status','unauthorized','attempt_id',v_attempt_id);
    END IF;

    SELECT principal_id INTO STRICT v_principal
      FROM sfeb.principal_scope_grant
     WHERE grant_id = v_grant_id;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(v_destination || ':' || p_operation_id::text, 0)
    );

    SELECT * INTO v_existing
      FROM sfeb.operation_ledger
     WHERE destination_id = v_destination
       AND operation_id = p_operation_id;

    IF FOUND THEN
        IF v_existing.db_principal = session_user
           AND v_existing.principal_id = v_principal
           AND v_existing.tenant_id = p_tenant_id
           AND v_existing.resource_id = p_resource_id
           AND v_existing.action = 'set_state'
           AND v_existing.request_digest = v_request_digest
           AND v_existing.policy_digest = p_policy_digest
           AND v_existing.authority_digest = p_authority_digest
           AND v_existing.authority_revision = p_authority_revision
           AND v_existing.fence_token = p_fence_token
           AND v_existing.expected_prior_revision = p_expected_revision THEN
            INSERT INTO sfeb.attempt_ledger
                (destination_id, operation_id, db_principal, tenant_id, resource_id, request_digest, outcome, detail, attempted_at)
            VALUES
                (v_destination, p_operation_id, session_user, p_tenant_id, p_resource_id,
                 v_request_digest, 'replayed', 'returned immutable original operation outcome', v_now)
            RETURNING attempt_id INTO v_attempt_id;
            RETURN jsonb_build_object(
                'status','replayed','attempt_id',v_attempt_id,
                'original_outcome',v_existing.outcome,
                'receipt',v_existing.receipt_payload
            );
        END IF;

        INSERT INTO sfeb.attempt_ledger
            (destination_id, operation_id, db_principal, tenant_id, resource_id, request_digest, outcome, detail, attempted_at)
        VALUES
            (v_destination, p_operation_id, session_user, p_tenant_id, p_resource_id,
             v_request_digest, 'identity_conflict', 'destination-global operation ID already has a different immutable binding', v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status','identity_conflict','attempt_id',v_attempt_id);
    END IF;

    SELECT * INTO v_state
      FROM sfeb.resource_state
     WHERE tenant_id = p_tenant_id AND resource_id = p_resource_id
     FOR UPDATE;

    IF NOT FOUND THEN
        INSERT INTO sfeb.attempt_ledger
            (destination_id, operation_id, db_principal, tenant_id, resource_id, request_digest, outcome, detail, attempted_at)
        VALUES
            (v_destination, p_operation_id, session_user, p_tenant_id, p_resource_id,
             v_request_digest, 'resource_unavailable', 'authorized resource not found at destination', v_now)
        RETURNING attempt_id INTO v_attempt_id;
        RETURN jsonb_build_object('status','resource_unavailable','attempt_id',v_attempt_id);
    END IF;

    v_prior_revision := v_state.revision;
    v_result_revision := v_prior_revision;
    v_result_digest := v_state.state_digest;

    IF p_expected_revision <> v_state.revision THEN
        v_outcome := 'refused';
        v_rejection := 'revision_conflict';
    ELSIF p_policy_digest <> v_state.accepted_policy_digest
       OR p_authority_digest <> v_state.accepted_authority_digest
       OR p_authority_revision <> v_state.accepted_authority_revision
       OR v_now > v_state.authority_valid_until
       OR v_state.authority_updated_at > v_now
       OR v_now - v_state.authority_updated_at > v_state.authority_max_age THEN
        v_outcome := 'refused';
        v_rejection := 'stale_or_mismatched_authority_or_policy';
    ELSIF p_fence_token < v_state.fencing_high_water THEN
        v_outcome := 'refused';
        v_rejection := 'stale_fence';
    ELSE
        v_result_revision := v_prior_revision + 1;
        v_result_digest := encode(
            public.digest(convert_to(p_next_state::text, 'UTF8'), 'sha256'), 'hex'
        );
    END IF;

    IF v_state.event_sequence = 9223372036854775807 THEN
        RAISE EXCEPTION 'event sequence exhausted';
    END IF;
    v_event_sequence := v_state.event_sequence + 1;

    IF v_outcome = 'committed' THEN
        UPDATE sfeb.resource_state
           SET state_payload = p_next_state,
               revision = v_result_revision,
               state_digest = v_result_digest,
               fencing_high_water = GREATEST(fencing_high_water, p_fence_token),
               event_sequence = v_event_sequence
         WHERE tenant_id = p_tenant_id AND resource_id = p_resource_id;

        INSERT INTO sfeb.state_history
            (tenant_id, resource_id, revision, state_digest, state_payload, operation_id, event_sequence, recorded_at)
        VALUES
            (p_tenant_id, p_resource_id, v_result_revision, v_result_digest,
             p_next_state, p_operation_id, v_event_sequence, v_now);
    ELSE
        UPDATE sfeb.resource_state
           SET event_sequence = v_event_sequence
         WHERE tenant_id = p_tenant_id AND resource_id = p_resource_id;
    END IF;

    v_receipt := jsonb_build_object(
        'profile','sfeb-postgres-authorized-sink-v1',
        'destination_id',v_destination,
        'operation_id',p_operation_id::text,
        'principal_id',v_principal,
        'db_principal',session_user::text,
        'tenant_id',p_tenant_id,
        'resource_id',p_resource_id,
        'action','set_state',
        'request_digest',v_request_digest,
        'request_digest_algorithm','SHA-256',
        'request_encoding','exact UTF-8 JSON bytes; canonicalization verified by caller adapter',
        'policy_digest',p_policy_digest,
        'authority_digest',p_authority_digest,
        'authority_revision',p_authority_revision,
        'fence_token',p_fence_token,
        'expected_prior_revision',p_expected_revision,
        'prior_revision',v_prior_revision,
        'result_revision',v_result_revision,
        'prior_digest',v_state.state_digest,
        'result_digest',v_result_digest,
        'state_digest_profile','postgres-jsonb-text-v1',
        'outcome',v_outcome,
        'rejection_reason',v_rejection,
        'event_sequence',v_event_sequence,
        'committed_at',CASE WHEN v_outcome = 'committed' THEN to_jsonb(v_now) ELSE 'null'::jsonb END
    );

    INSERT INTO sfeb.operation_ledger (
        destination_id, operation_id, principal_id, db_principal,
        tenant_id, resource_id, action, request_digest, policy_digest,
        authority_digest, authority_revision, fence_token, expected_prior_revision,
        outcome, rejection_reason, event_sequence, prior_revision, result_revision,
        prior_digest, result_digest, committed_at, receipt_payload
    ) VALUES (
        v_destination, p_operation_id, v_principal, session_user,
        p_tenant_id, p_resource_id, 'set_state', v_request_digest, p_policy_digest,
        p_authority_digest, p_authority_revision, p_fence_token, p_expected_revision,
        v_outcome, v_rejection, v_event_sequence, v_prior_revision, v_result_revision,
        v_state.state_digest, v_result_digest,
        CASE WHEN v_outcome = 'committed' THEN v_now ELSE NULL END,
        v_receipt
    );

    INSERT INTO sfeb.attempt_ledger
        (destination_id, operation_id, db_principal, tenant_id, resource_id, request_digest, outcome, detail, attempted_at)
    VALUES
        (v_destination, p_operation_id, session_user, p_tenant_id, p_resource_id,
         v_request_digest, v_outcome, v_rejection, v_now)
    RETURNING attempt_id INTO v_attempt_id;

    RETURN jsonb_build_object('status',v_outcome,'attempt_id',v_attempt_id,'receipt',v_receipt);
END;
$$;

CREATE FUNCTION sfeb.readback_authorized(
    p_tenant_id text,
    p_resource_id text,
    p_revision bigint,
    p_policy_digest text,
    p_authority_digest text,
    p_authority_revision bigint
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
DECLARE
    v_grant_id uuid;
    v_observed sfeb.state_history%ROWTYPE;
BEGIN
    v_grant_id := sfeb._lock_scope_grant(
        p_tenant_id, p_resource_id, 'readback',
        p_policy_digest, p_authority_digest, p_authority_revision
    );
    IF v_grant_id IS NULL THEN
        RETURN jsonb_build_object('status','unauthorized');
    END IF;

    SELECT * INTO v_observed
      FROM sfeb.state_history
     WHERE tenant_id = p_tenant_id
       AND resource_id = p_resource_id
       AND revision = p_revision;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('status','not_observed','requested_revision',p_revision);
    END IF;

    RETURN jsonb_build_object(
        'status','observed',
        'tenant_id',v_observed.tenant_id,
        'resource_id',v_observed.resource_id,
        'revision',v_observed.revision,
        'state_digest',v_observed.state_digest,
        'state_digest_profile','postgres-jsonb-text-v1',
        'state_payload',v_observed.state_payload,
        'operation_id',v_observed.operation_id,
        'event_sequence',v_observed.event_sequence,
        'recorded_at',v_observed.recorded_at
    );
END;
$$;

CREATE FUNCTION sfeb.revoke_scope_grant(p_grant_id uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
DECLARE
    v_rows integer;
BEGIN
    IF NOT pg_catalog.pg_has_role(session_user, 'sfeb_grant_admin', 'MEMBER') THEN
        RAISE EXCEPTION 'caller is not a MON grant administrator';
    END IF;
    UPDATE sfeb.principal_scope_grant
       SET revoked_at = COALESCE(revoked_at, clock_timestamp())
     WHERE grant_id = p_grant_id;
    GET DIAGNOSTICS v_rows = ROW_COUNT;
    RETURN v_rows = 1;
END;
$$;

RESET ROLE;

REVOKE ALL ON ALL TABLES IN SCHEMA sfeb FROM PUBLIC, sfeb_writer, sfeb_observer, sfeb_grant_admin;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA sfeb FROM PUBLIC;
GRANT USAGE ON SCHEMA sfeb TO sfeb_writer, sfeb_observer, sfeb_grant_admin;

GRANT EXECUTE ON FUNCTION sfeb.apply_action_authorized(text,text,uuid,bytea,jsonb,jsonb,text,text,bigint,bigint,bigint) TO sfeb_writer;
GRANT EXECUTE ON FUNCTION sfeb.readback_authorized(text,text,bigint,text,text,bigint) TO sfeb_observer;
GRANT EXECUTE ON FUNCTION sfeb.revoke_scope_grant(uuid) TO sfeb_grant_admin;

GRANT SELECT ON sfeb.resource_state, sfeb.state_history TO sfeb_observer;

REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA sfeb
FROM PUBLIC, sfeb_writer, sfeb_observer, sfeb_grant_admin;

COMMENT ON SCHEMA sfeb IS
'MON-013 disposable reference profile. Superusers, database owners, host administrators and privileged bypass paths remain explicit external trust roots.';
COMMENT ON FUNCTION sfeb.apply_action_authorized(text,text,uuid,bytea,jsonb,jsonb,text,text,bigint,bigint,bigint) IS
'Principal-scoped conditional mutation. Uses session_user and a locked scope grant. Receipt payload is unsigned; caller adapters must validate canonical request bytes and sign/register the record separately.';
COMMENT ON FUNCTION sfeb.readback_authorized(text,text,bigint,text,text,bigint) IS
'Read-only exact-revision observation scoped to the authenticated database session principal. It does not prove physical effect.';

-- MON-013 commit-time observation extension.
-- Apply after schema.sql and before integration.sql in a disposable database.
-- Configure track_commit_timestamp=on in postgresql.conf and restart PostgreSQL.

ALTER TABLE sfeb.operation_ledger
  ADD COLUMN authority_updated_at_at_mutation timestamptz,
  ADD COLUMN authority_valid_until_at_mutation timestamptz,
  ADD COLUMN authority_max_age_at_mutation interval;

CREATE FUNCTION sfeb._capture_authority_snapshot()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
DECLARE v_state sfeb.resource_state%ROWTYPE;
BEGIN
  SELECT * INTO v_state
    FROM sfeb.resource_state
   WHERE tenant_id=NEW.tenant_id AND resource_id=NEW.resource_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'MON-013 cannot capture authority snapshot for missing resource';
  END IF;
  NEW.authority_updated_at_at_mutation := v_state.authority_updated_at;
  NEW.authority_valid_until_at_mutation := v_state.authority_valid_until;
  NEW.authority_max_age_at_mutation := v_state.authority_max_age;
  RETURN NEW;
END;
$$;
ALTER FUNCTION sfeb._capture_authority_snapshot() OWNER TO sfeb_owner;
REVOKE ALL ON FUNCTION sfeb._capture_authority_snapshot() FROM PUBLIC;

CREATE TRIGGER operation_capture_authority_snapshot
BEFORE INSERT ON sfeb.operation_ledger
FOR EACH ROW EXECUTE FUNCTION sfeb._capture_authority_snapshot();

ALTER TABLE sfeb.operation_ledger
  ALTER COLUMN authority_updated_at_at_mutation SET NOT NULL,
  ALTER COLUMN authority_valid_until_at_mutation SET NOT NULL,
  ALTER COLUMN authority_max_age_at_mutation SET NOT NULL;

CREATE OR REPLACE FUNCTION sfeb.readback_authorized(
  p_tenant text,p_resource text,p_operation uuid,
  p_expected_revision bigint,p_expected_state_digest text,p_expected_request_digest text,
  p_policy text,p_authority text,p_authority_revision bigint
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, sfeb, pg_temp
AS $$
DECLARE
  v_destination text;
  v_grant_id uuid;
  v_op sfeb.operation_ledger%ROWTYPE;
  v_observed sfeb.state_history%ROWTYPE;
  v_grant sfeb.principal_scope_grant%ROWTYPE;
  v_commit_time timestamptz;
  v_authority_at_commit text := 'uncheckable';
  v_qualification text := 'indeterminate';
BEGIN
  SELECT destination_id INTO STRICT v_destination
    FROM sfeb.destination_config WHERE singleton;

  v_grant_id := sfeb._lock_scope_grant(
    p_tenant,p_resource,'readback',p_policy,p_authority,p_authority_revision
  );
  IF v_grant_id IS NULL THEN
    RETURN jsonb_build_object('status','unauthorized');
  END IF;

  IF p_operation IS NULL OR p_expected_revision IS NULL
     OR p_expected_state_digest IS NULL OR p_expected_request_digest IS NULL
     OR p_policy IS NULL OR p_authority IS NULL
     OR p_expected_state_digest !~ '^[0-9a-f]{64}$'
     OR p_expected_request_digest !~ '^[0-9a-f]{64}$' THEN
    RETURN jsonb_build_object('status','indeterminate','reason','missing_or_malformed_expected_binding');
  END IF;

  SELECT * INTO v_op
    FROM sfeb.operation_ledger
   WHERE destination_id=v_destination AND operation_id=p_operation;
  IF NOT FOUND THEN
    RETURN jsonb_build_object('status','not_observed','reason','operation_not_visible_after_commit');
  END IF;

  IF v_op.tenant_id<>p_tenant OR v_op.resource_id<>p_resource
     OR v_op.request_digest<>p_expected_request_digest
     OR v_op.policy_digest<>p_policy OR v_op.authority_digest<>p_authority
     OR v_op.authority_revision<>p_authority_revision
     OR v_op.result_revision<>p_expected_revision
     OR v_op.result_digest<>p_expected_state_digest THEN
    RETURN jsonb_build_object('status','contradicted','reason','operation_ledger_binding_mismatch');
  END IF;

  IF v_op.outcome='refused' THEN
    RETURN jsonb_build_object('status','refused','reason',v_op.rejection_reason);
  END IF;

  SELECT * INTO v_observed
    FROM sfeb.state_history
   WHERE tenant_id=p_tenant AND resource_id=p_resource
     AND revision=p_expected_revision AND operation_id=p_operation;
  IF NOT FOUND THEN
    RETURN jsonb_build_object('status','not_observed','reason','exact_revision_not_visible');
  END IF;
  IF v_observed.state_digest<>p_expected_state_digest THEN
    RETURN jsonb_build_object('status','contradicted','reason','readback_digest_mismatch');
  END IF;

  IF pg_catalog.current_setting('track_commit_timestamp',true)='on' THEN
    SELECT pg_catalog.pg_xact_commit_timestamp(o.xmin)
      INTO v_commit_time
      FROM sfeb.operation_ledger o
     WHERE o.destination_id=v_destination AND o.operation_id=p_operation;
  END IF;

  IF v_commit_time IS NOT NULL THEN
    SELECT * INTO v_grant
      FROM sfeb.principal_scope_grant
     WHERE grant_id=v_op.grant_id;

    IF FOUND
       AND v_grant.db_principal=v_op.db_principal
       AND v_grant.principal_id=v_op.principal_id
       AND v_grant.tenant_id=v_op.tenant_id
       AND v_grant.resource_id=v_op.resource_id
       AND v_grant.action='mutate'
       AND v_grant.policy_digest=v_op.policy_digest
       AND v_grant.authority_digest=v_op.authority_digest
       AND v_grant.authority_revision=v_op.authority_revision
       AND v_commit_time>=v_grant.not_before
       AND v_commit_time<v_grant.expires_at
       AND v_commit_time>=v_grant.verified_at
       AND v_commit_time-v_grant.verified_at<=v_grant.max_age
       AND (v_grant.revoked_at IS NULL OR v_grant.revoked_at>v_commit_time)
       AND v_commit_time>=v_op.authority_updated_at_at_mutation
       AND v_commit_time<v_op.authority_valid_until_at_mutation
       AND v_commit_time-v_op.authority_updated_at_at_mutation<=v_op.authority_max_age_at_mutation THEN
      v_authority_at_commit := 'valid';
      v_qualification := 'qualified';
    ELSE
      v_authority_at_commit := 'invalid';
      v_qualification := 'not_qualified';
    END IF;
  END IF;

  RETURN jsonb_build_object(
    'status','observed_committed',
    'qualification',v_qualification,
    'authority_at_commit',v_authority_at_commit,
    'database_commit_time',v_commit_time,
    'commit_time_source','pg_xact_commit_timestamp(operation_ledger.xmin)',
    'commit_time_retention','requires track_commit_timestamp=on; VACUUM may remove historical values',
    'destination_id',v_destination,
    'operation_id',p_operation::text,
    'tenant_id',p_tenant,
    'resource_id',p_resource,
    'revision',v_observed.revision,
    'state_digest',v_observed.state_digest,
    'state_digest_profile','postgres-jsonb-text-v1',
    'state_payload',v_observed.state_payload,
    'event_sequence',v_observed.event_sequence,
    'recorded_at',v_observed.recorded_at
  );
END;
$$;

COMMENT ON FUNCTION sfeb.readback_authorized(text,text,uuid,bigint,text,text,text,text,bigint) IS
'Separate post-commit exact readback. Observed state is distinct from authority_at_commit. Authority can be graded only when PostgreSQL commit timestamp exists; missing/aged-out timestamps yield indeterminate qualification.';

-- MON-013 disposable PostgreSQL integration smoke/negative controls.
-- Run as superuser after schema.sql and commit-time-readback.sql.
\set ON_ERROR_STOP on

DO $roles$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='mon013_writer_a') THEN CREATE ROLE mon013_writer_a LOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='mon013_writer_b') THEN CREATE ROLE mon013_writer_b LOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='mon013_observer_a') THEN CREATE ROLE mon013_observer_a LOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='mon013_observer_b') THEN CREATE ROLE mon013_observer_b LOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='mon013_grant_admin') THEN CREATE ROLE mon013_grant_admin LOGIN; END IF;
END
$roles$;
ALTER ROLE mon013_writer_a LOGIN INHERIT NOSUPERUSER NOBYPASSRLS;
ALTER ROLE mon013_writer_b LOGIN INHERIT NOSUPERUSER NOBYPASSRLS;
ALTER ROLE mon013_observer_a LOGIN INHERIT NOSUPERUSER NOBYPASSRLS;
ALTER ROLE mon013_observer_b LOGIN INHERIT NOSUPERUSER NOBYPASSRLS;
ALTER ROLE mon013_grant_admin LOGIN INHERIT NOSUPERUSER NOBYPASSRLS;

GRANT sfeb_writer TO mon013_writer_a, mon013_writer_b;
GRANT sfeb_observer TO mon013_observer_a, mon013_observer_b;
GRANT sfeb_grant_admin TO mon013_grant_admin;

-- Fixed test grants make the integration reproducible. In deployment, these
-- rows are populated only by the trusted grant-provisioning authority.
INSERT INTO sfeb.resource_state(
  tenant_id,resource_id,state_payload,revision,state_digest,
  accepted_policy_digest,accepted_authority_digest,accepted_authority_revision,
  authority_updated_at,authority_valid_until,authority_max_age,fencing_high_water,event_sequence
) VALUES
  ('tenant-a','resource-a','{"v": 0}'::jsonb,0,
   encode(public.digest(convert_to('{"v": 0}','UTF8'),'sha256'),'hex'),
   repeat('a',64),repeat('b',64),4,clock_timestamp(),
   clock_timestamp()+interval '1 hour',interval '1 hour',0,0),
  ('tenant-b','resource-b','{"v": 0}'::jsonb,0,
   encode(public.digest(convert_to('{"v": 0}','UTF8'),'sha256'),'hex'),
   repeat('a',64),repeat('b',64),4,clock_timestamp(),
   clock_timestamp()+interval '1 hour',interval '1 hour',0,0);

INSERT INTO sfeb.state_history(
  tenant_id,resource_id,revision,state_digest,state_payload,operation_id,event_sequence,recorded_at
)
SELECT tenant_id,resource_id,revision,state_digest,state_payload,NULL,event_sequence,clock_timestamp()
FROM sfeb.resource_state;

INSERT INTO sfeb.principal_scope_grant(
  grant_id,db_principal,principal_id,tenant_id,resource_id,action,
  policy_digest,authority_digest,authority_revision,issuer,issuer_key_id,evidence_digest,
  verified_at,not_before,expires_at,max_age
) VALUES
  ('10000000-0000-0000-0000-000000000001','mon013_writer_a','principal-a','tenant-a','resource-a','mutate',
   repeat('a',64),repeat('b',64),4,'test-authority','test-key',repeat('c',64),
   clock_timestamp(),clock_timestamp()-interval '1 minute',clock_timestamp()+interval '1 hour',interval '1 hour'),
  ('10000000-0000-0000-0000-000000000002','mon013_observer_a','observer-a','tenant-a','resource-a','readback',
   repeat('a',64),repeat('b',64),4,'test-authority','test-key',repeat('d',64),
   clock_timestamp(),clock_timestamp()-interval '1 minute',clock_timestamp()+interval '1 hour',interval '1 hour'),
  ('20000000-0000-0000-0000-000000000001','mon013_writer_b','principal-b','tenant-b','resource-b','mutate',
   repeat('a',64),repeat('b',64),4,'test-authority','test-key',repeat('e',64),
   clock_timestamp(),clock_timestamp()-interval '1 minute',clock_timestamp()+interval '1 hour',interval '1 hour'),
  ('20000000-0000-0000-0000-000000000002','mon013_observer_b','observer-b','tenant-b','resource-b','readback',
   repeat('a',64),repeat('b',64),4,'test-authority','test-key',repeat('f',64),
   clock_timestamp(),clock_timestamp()-interval '1 minute',clock_timestamp()+interval '1 hour',interval '1 hour');

-- Commit a valid action. The function must return commit_pending, not committed.
SET SESSION AUTHORIZATION mon013_writer_a;
DO $test$
DECLARE
  r jsonb;
  body text := '{"action":"set_state","authority_digest":"' || repeat('b',64) ||
    '","authority_revision":4,"expected_revision":0,"fence_token":7,"next_state":{"v":1},' ||
    '"operation_id":"00000000-0000-0000-0000-000000000001","policy_digest":"' || repeat('a',64) ||
    '","resource_id":"resource-a","tenant_id":"tenant-a"}';
BEGIN
  r := sfeb.apply_action_authorized(
    'tenant-a','resource-a','00000000-0000-0000-0000-000000000001',
    convert_to(body,'UTF8'),body::jsonb,'{"v":1}'::jsonb,
    repeat('a',64),repeat('b',64),4,7,0
  );
  IF r->>'status' <> 'commit_pending' THEN
    RAISE EXCEPTION 'expected commit_pending, got %',r;
  END IF;
END
$test$;
RESET SESSION AUTHORIZATION;

-- Lost-ack retry after the first transaction committed returns the original
-- immutable outcome and cannot increment the resource revision again.
SET SESSION AUTHORIZATION mon013_writer_a;
DO $test$
DECLARE
  r jsonb;
  body text := '{"action":"set_state","authority_digest":"' || repeat('b',64) ||
    '","authority_revision":4,"expected_revision":0,"fence_token":7,"next_state":{"v":1},' ||
    '"operation_id":"00000000-0000-0000-0000-000000000001","policy_digest":"' || repeat('a',64) ||
    '","resource_id":"resource-a","tenant_id":"tenant-a"}';
BEGIN
  r := sfeb.apply_action_authorized(
    'tenant-a','resource-a','00000000-0000-0000-0000-000000000001',
    convert_to(body,'UTF8'),body::jsonb,'{"v":1}'::jsonb,
    repeat('a',64),repeat('b',64),4,7,0
  );
  IF r->>'status' <> 'replayed' OR r->>'original_outcome' <> 'accepted_for_commit' THEN
    RAISE EXCEPTION 'expected original replay, got %',r;
  END IF;
END
$test$;
RESET SESSION AUTHORIZATION;

-- Another authenticated principal with a valid grant for another tenant
-- cannot reuse the destination-global operation ID for a different action.
SET SESSION AUTHORIZATION mon013_writer_b;
DO $test$
DECLARE
  r jsonb;
  body text := '{"action":"set_state","authority_digest":"' || repeat('b',64) ||
    '","authority_revision":4,"expected_revision":0,"fence_token":8,"next_state":{"v":2},' ||
    '"operation_id":"00000000-0000-0000-0000-000000000001","policy_digest":"' || repeat('a',64) ||
    '","resource_id":"resource-b","tenant_id":"tenant-b"}';
BEGIN
  r := sfeb.apply_action_authorized(
    'tenant-b','resource-b','00000000-0000-0000-0000-000000000001',
    convert_to(body,'UTF8'),body::jsonb,'{"v":2}'::jsonb,
    repeat('a',64),repeat('b',64),4,8,0
  );
  IF r->>'status' <> 'identity_conflict' THEN
    RAISE EXCEPTION 'expected destination-global identity_conflict, got %',r;
  END IF;
END
$test$;
RESET SESSION AUTHORIZATION;

-- A principal cannot use its valid tenant-A grant to mutate tenant B.
SET SESSION AUTHORIZATION mon013_writer_a;
DO $test$
DECLARE
  r jsonb;
  body text := '{"action":"set_state","authority_digest":"' || repeat('b',64) ||
    '","authority_revision":4,"expected_revision":0,"fence_token":9,"next_state":{"v":3},' ||
    '"operation_id":"00000000-0000-0000-0000-000000000003","policy_digest":"' || repeat('a',64) ||
    '","resource_id":"resource-b","tenant_id":"tenant-b"}';
BEGIN
  r := sfeb.apply_action_authorized(
    'tenant-b','resource-b','00000000-0000-0000-0000-000000000003',
    convert_to(body,'UTF8'),body::jsonb,'{"v":3}'::jsonb,
    repeat('a',64),repeat('b',64),4,9,0
  );
  IF r->>'status' <> 'unauthorized' THEN
    RAISE EXCEPTION 'expected cross-tenant unauthorized, got %',r;
  END IF;
END
$test$;
RESET SESSION AUTHORIZATION;

-- A separate readback principal must observe the exact operation, revision,
-- state digest, request digest and transaction commit time.
SET SESSION AUTHORIZATION mon013_observer_a;
DO $test$
DECLARE
  r jsonb;
  body text := '{"action":"set_state","authority_digest":"' || repeat('b',64) ||
    '","authority_revision":4,"expected_revision":0,"fence_token":7,"next_state":{"v":1},' ||
    '"operation_id":"00000000-0000-0000-0000-000000000001","policy_digest":"' || repeat('a',64) ||
    '","resource_id":"resource-a","tenant_id":"tenant-a"}';
  request_digest text := encode(public.digest(convert_to(body,'UTF8'),'sha256'),'hex');
  state_digest text := encode(public.digest(convert_to('{"v": 1}','UTF8'),'sha256'),'hex');
BEGIN
  r := sfeb.readback_authorized(
    'tenant-a','resource-a','00000000-0000-0000-0000-000000000001',
    1,state_digest,request_digest,repeat('a',64),repeat('b',64),4
  );
  IF r->>'status' <> 'observed_committed' THEN
    RAISE EXCEPTION 'expected observed_committed, got %',r;
  END IF;
  IF r->>'qualification' <> 'qualified' OR r->>'authority_at_commit' <> 'valid' THEN
    RAISE EXCEPTION 'expected qualified authority at commit, got %',r;
  END IF;
  IF r->>'database_commit_time' IS NULL OR r->>'database_commit_time' = 'null' THEN
    RAISE EXCEPTION 'expected tracked database commit timestamp, got %',r;
  END IF;
END
$test$;
RESET SESSION AUTHORIZATION;

-- Observer B has no grant for tenant A. Both the readback function and direct
-- RLS-protected SELECT must refuse cross-tenant disclosure.
SET SESSION AUTHORIZATION mon013_observer_b;
DO $test$
DECLARE
  r jsonb;
  n bigint;
BEGIN
  r := sfeb.readback_authorized(
    'tenant-a','resource-a','00000000-0000-0000-0000-000000000001',
    1,repeat('0',64),repeat('0',64),repeat('a',64),repeat('b',64),4
  );
  IF r->>'status' <> 'unauthorized' THEN
    RAISE EXCEPTION 'expected readback unauthorized, got %',r;
  END IF;
  SELECT count(*) INTO n FROM sfeb.resource_state WHERE tenant_id='tenant-a';
  IF n <> 0 THEN RAISE EXCEPTION 'observer B disclosed tenant A rows: %',n; END IF;
END
$test$;
RESET SESSION AUTHORIZATION;

-- Revoke writer A's grant through the grant-admin function; subsequent
-- mutation must fail closed. The revoke transaction waits for any active
-- mutation holding the grant row FOR SHARE.
SET SESSION AUTHORIZATION mon013_grant_admin;
SELECT sfeb.revoke_scope_grant('10000000-0000-0000-0000-000000000001');
RESET SESSION AUTHORIZATION;

SET SESSION AUTHORIZATION mon013_writer_a;
DO $test$
DECLARE
  r jsonb;
  body text := '{"action":"set_state","authority_digest":"' || repeat('b',64) ||
    '","authority_revision":4,"expected_revision":1,"fence_token":10,"next_state":{"v":4},' ||
    '"operation_id":"00000000-0000-0000-0000-000000000004","policy_digest":"' || repeat('a',64) ||
    '","resource_id":"resource-a","tenant_id":"tenant-a"}';
BEGIN
  r := sfeb.apply_action_authorized(
    'tenant-a','resource-a','00000000-0000-0000-0000-000000000004',
    convert_to(body,'UTF8'),body::jsonb,'{"v":4}'::jsonb,
    repeat('a',64),repeat('b',64),4,10,1
  );
  IF r->>'status' <> 'unauthorized' THEN
    RAISE EXCEPTION 'expected revoked grant rejection, got %',r;
  END IF;
END
$test$;
RESET SESSION AUTHORIZATION;

DO $assert$
DECLARE n bigint;
BEGIN
  IF (SELECT revision FROM sfeb.resource_state WHERE tenant_id='tenant-a' AND resource_id='resource-a') <> 1 THEN
    RAISE EXCEPTION 'tenant A resource revision should be exactly 1';
  END IF;
  IF (SELECT revision FROM sfeb.resource_state WHERE tenant_id='tenant-b' AND resource_id='resource-b') <> 0 THEN
    RAISE EXCEPTION 'tenant B resource changed after cross-tenant replay attempts';
  END IF;
  SELECT count(*) INTO n FROM sfeb.operation_ledger
   WHERE operation_id='00000000-0000-0000-0000-000000000001';
  IF n <> 1 THEN RAISE EXCEPTION 'destination-global key should have one operation row, found %',n; END IF;
  IF has_table_privilege('sfeb_writer','sfeb.resource_state','UPDATE') THEN
    RAISE EXCEPTION 'writer role has direct UPDATE privilege';
  END IF;
  IF NOT (SELECT relforcerowsecurity FROM pg_catalog.pg_class WHERE oid='sfeb.resource_state'::regclass) THEN
    RAISE EXCEPTION 'resource_state FORCE ROW LEVEL SECURITY not enabled';
  END IF;
  IF (SELECT rolsuper OR rolbypassrls FROM pg_catalog.pg_roles WHERE rolname='sfeb_owner') THEN
    RAISE EXCEPTION 'sfeb_owner must be NOBYPASSRLS and non-superuser';
  END IF;
END
$assert$;

SELECT 'MON-013 PostgreSQL integration smoke/negative controls: PASS' AS result;

-- Run after schema.sql in a disposable PostgreSQL database as a role that can
-- SET ROLE to sfeb_writer and sfeb_observer.
INSERT INTO sfeb.resource_state (
    tenant_id, resource_id, state_payload, revision, state_digest,
    accepted_authority_revision, authority_updated_at, authority_valid_until,
    authority_max_age_seconds, fencing_high_water, event_sequence
) VALUES (
    'tenant-a', 'resource-1', '{"value":0}'::jsonb, 0,
    encode(public.digest(convert_to('{"value": 0}', 'UTF8'), 'sha256'), 'hex'),
    4, clock_timestamp(), clock_timestamp() + interval '1 hour', 3600, 0, 0
);

SET ROLE sfeb_writer;
DO $$
DECLARE r jsonb;
BEGIN
    r := sfeb.apply_action('tenant-a','resource-1','00000000-0000-0000-0000-000000000001',
        convert_to('{"next_state":{"value":1}}','UTF8'), '{"next_state":{"value":1}}'::jsonb, '{"value":1}'::jsonb, 4, 7, 0);
    IF r->>'status' <> 'committed' THEN RAISE EXCEPTION 'first mutation failed: %', r; END IF;

    r := sfeb.apply_action('tenant-a','resource-1','00000000-0000-0000-0000-000000000001',
        convert_to('{"next_state":{"value":1}}','UTF8'), '{"next_state":{"value":1}}'::jsonb, '{"value":1}'::jsonb, 4, 7, 0);
    IF r->>'status' <> 'replayed' THEN RAISE EXCEPTION 'same-ID retry was not replayed: %', r; END IF;

    r := sfeb.apply_action('tenant-a','resource-1','00000000-0000-0000-0000-000000000001',
        convert_to('{"next_state":{"value":2}}','UTF8'), '{"next_state":{"value":2}}'::jsonb, '{"value":2}'::jsonb, 4, 7, 0);
    IF r->>'status' <> 'identity_conflict' THEN RAISE EXCEPTION 'changed binding not rejected: %', r; END IF;

    r := sfeb.apply_action('tenant-a','resource-1','00000000-0000-0000-0000-000000000002',
        convert_to('{"next_state":{"value":2}}','UTF8'), '{"next_state":{"value":2}}'::jsonb, '{"value":2}'::jsonb, 4, 6, 1);
    IF r->'receipt'->>'rejection_reason' <> 'stale_fence' THEN RAISE EXCEPTION 'stale fence not rejected: %', r; END IF;

    r := sfeb.apply_action('tenant-a','resource-1','00000000-0000-0000-0000-000000000003',
        convert_to('{"next_state":{"value":3}}','UTF8'), '{"next_state":{"value":3}}'::jsonb, '{"value":3}'::jsonb, 3, 7, 1);
    IF r->'receipt'->>'rejection_reason' <> 'stale_or_invalid_authority' THEN RAISE EXCEPTION 'stale authority not rejected: %', r; END IF;

    r := sfeb.apply_action('tenant-a','resource-1','00000000-0000-0000-0000-000000000004',
        convert_to('{"next_state":{"value":4}}','UTF8'), '{"next_state":{"value":4}}'::jsonb, '{"value":4}'::jsonb, 4, 8, 1);
    IF r->>'status' <> 'committed' THEN RAISE EXCEPTION 'valid follow-up failed: %', r; END IF;
END $$;
RESET ROLE;

-- Observer role can read the declared view but has no direct mutation rights.
SET ROLE sfeb_observer;
DO $$
DECLARE n bigint;
BEGIN
    SELECT revision INTO n FROM sfeb.observer_readback
    WHERE tenant_id='tenant-a' AND resource_id='resource-1';
    IF n <> 2 THEN RAISE EXCEPTION 'expected exact final revision 2, got %', n; END IF;
END $$;
RESET ROLE;

DO $$
BEGIN
    IF has_table_privilege('sfeb_writer', 'sfeb.resource_state', 'UPDATE') THEN
        RAISE EXCEPTION 'writer unexpectedly has direct UPDATE privilege';
    END IF;
    IF has_table_privilege('sfeb_observer', 'sfeb.resource_state', 'UPDATE') THEN
        RAISE EXCEPTION 'observer unexpectedly has direct UPDATE privilege';
    END IF;
    IF (SELECT count(*) FROM sfeb.operation_ledger WHERE tenant_id='tenant-a') <> 4 THEN
        RAISE EXCEPTION 'expected four immutable first outcomes';
    END IF;
    IF (SELECT count(*) FROM sfeb.attempt_ledger WHERE tenant_id='tenant-a') <> 6 THEN
        RAISE EXCEPTION 'expected every verdict and replay/conflict attempt to be recorded';
    END IF;
    IF (SELECT count(*) FROM sfeb.attempt_ledger WHERE outcome='identity_conflict') <> 1 THEN
        RAISE EXCEPTION 'changed-binding attempt was not separately recorded';
    END IF;
    IF (SELECT revision FROM sfeb.resource_state WHERE tenant_id='tenant-a' AND resource_id='resource-1') <> 2 THEN
        RAISE EXCEPTION 'duplicate or rejected request changed resource revision';
    END IF;
    IF (SELECT event_sequence FROM sfeb.resource_state WHERE tenant_id='tenant-a' AND resource_id='resource-1') <> 4 THEN
        RAISE EXCEPTION 'expected event sequence 4, including two rejections';
    END IF;
END $$;

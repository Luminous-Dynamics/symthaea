\set ON_ERROR_STOP on
WITH
ledger_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(to_jsonb(l)::text, E'\n' ORDER BY event_id), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM public.ledger AS l
),
schema_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(
            concat_ws('|', c.relname, a.attnum, a.attname, a.atttypid::regtype::text,
                      a.attnotnull, a.attgenerated, a.attidentity,
                      coalesce(pg_get_expr(d.adbin, d.adrelid), '')),
            E'\n' ORDER BY c.relname, a.attnum
        ), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM pg_class AS c
    JOIN pg_namespace AS n ON n.oid = c.relnamespace
    JOIN pg_attribute AS a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
    LEFT JOIN pg_attrdef AS d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
    WHERE n.nspname = 'public'
      AND c.relname IN ('ledger', 'sequence_probe', 'ledger_partitioned',
                        'ledger_partitioned_2026', 'ledger_partitioned_default',
                        'volatile_cache', 'large_object_ref', 'ledger_summary')
),
index_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(pg_get_indexdef(i.indexrelid), E'\n' ORDER BY t.relname, ic.relname), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM pg_index AS i
    JOIN pg_class AS t ON t.oid = i.indrelid
    JOIN pg_class AS ic ON ic.oid = i.indexrelid
    JOIN pg_namespace AS n ON n.oid = t.relnamespace
    WHERE n.nspname = 'public'
),
constraint_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(
            concat_ws('|', c.conrelid::regclass::text, c.conname, c.contype, pg_get_constraintdef(c.oid, true)),
            E'\n' ORDER BY c.conrelid::regclass::text, c.conname
        ), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM pg_constraint AS c
    JOIN pg_class AS t ON t.oid = c.conrelid
    JOIN pg_namespace AS n ON n.oid = t.relnamespace
    WHERE n.nspname = 'public'
),
trigger_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(
            pg_get_triggerdef(t.oid, true),
            E'\n' ORDER BY c.relname, t.tgname
        ), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM pg_trigger AS t
    JOIN pg_class AS c ON c.oid = t.tgrelid
    JOIN pg_namespace AS n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND NOT t.tgisinternal
),
routine_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(
            pg_get_functiondef(p.oid),
            E'\n' ORDER BY p.proname, pg_get_function_identity_arguments(p.oid)
        ), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM pg_proc AS p
    JOIN pg_namespace AS n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public' AND p.prokind IN ('f', 'p')
),
extension_state AS (
    SELECT coalesce(jsonb_agg(jsonb_build_object(
        'name', e.extname,
        'version', e.extversion,
        'schema', n.nspname
    ) ORDER BY e.extname), '[]'::jsonb) AS state
    FROM pg_extension AS e
    JOIN pg_namespace AS n ON n.oid = e.extnamespace
),
schema_acl_state AS (
    SELECT coalesce(jsonb_agg(jsonb_build_object(
        'schema', n.nspname,
        'acl', coalesce(n.nspacl::text, '')
    ) ORDER BY n.nspname), '[]'::jsonb) AS state
    FROM pg_namespace AS n
    WHERE n.nspname = 'public'
),
database_acl_state AS (
    SELECT coalesce(datacl::text, '') AS acl
    FROM pg_database
    WHERE datname = current_database()
),
role_membership_state AS (
    SELECT coalesce(jsonb_agg(jsonb_build_object(
        'member', member.rolname,
        'granted_role', granted.rolname,
        'admin_option', m.admin_option,
        'inherit_option', m.inherit_option,
        'set_option', m.set_option
    ) ORDER BY member.rolname, granted.rolname), '[]'::jsonb) AS state
    FROM pg_auth_members AS m
    JOIN pg_roles AS member ON member.oid = m.member
    JOIN pg_roles AS granted ON granted.oid = m.roleid
    WHERE member.rolname IN ('mon025', 'mon025_auditor', 'mon025_repl')
       OR granted.rolname IN ('mon025', 'mon025_auditor', 'mon025_repl')
),
sequence_probe_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(to_jsonb(s)::text, E'\n' ORDER BY id), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM public.sequence_probe AS s
),
sequence_state AS (
    SELECT jsonb_build_object('last_value', last_value, 'is_called', is_called) AS state
    FROM public.operation_sequence
),
large_object_state AS (
    SELECT encode(digest(lo_get(loid), 'sha256'), 'hex') AS digest
    FROM public.large_object_ref
    WHERE label = 'fixture-v1'
),
materialized_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(to_jsonb(m)::text, E'\n' ORDER BY account_id), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM public.ledger_summary AS m
),
partition_rows_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(to_jsonb(p)::text, E'\n' ORDER BY occurred_on, id), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM public.ledger_partitioned AS p
),
unlogged_state AS (
    SELECT encode(
        digest(convert_to(coalesce(string_agg(to_jsonb(u)::text, E'\n' ORDER BY cache_key), ''), 'UTF8'), 'sha256'),
        'hex'
    ) AS digest
    FROM public.volatile_cache AS u
),
role_state AS (
    SELECT coalesce(jsonb_agg(jsonb_build_object(
        'role', r.rolname,
        'superuser', r.rolsuper,
        'inherit', r.rolinherit,
        'create_role', r.rolcreaterole,
        'create_db', r.rolcreatedb,
        'replication', r.rolreplication,
        'login', r.rolcanlogin
    ) ORDER BY r.rolname), '[]'::jsonb) AS state
    FROM pg_roles AS r
    WHERE r.rolname IN ('mon025', 'mon025_auditor', 'mon025_repl')
),
partition_state AS (
    SELECT coalesce(jsonb_agg(jsonb_build_object(
        'relation', c.relname,
        'bound', coalesce(pg_get_expr(c.relpartbound, c.oid), '')
    ) ORDER BY c.relname), '[]'::jsonb) AS state
    FROM pg_class AS c
    JOIN pg_namespace AS n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relname LIKE 'ledger_partitioned%'
),
acl_state AS (
    SELECT coalesce(jsonb_agg(jsonb_build_object(
        'relation', c.relname,
        'acl', coalesce(c.relacl::text, '')
    ) ORDER BY c.relname), '[]'::jsonb) AS state
    FROM pg_class AS c
    JOIN pg_namespace AS n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relname IN ('ledger', 'ledger_partitioned', 'ledger_summary')
)
SELECT jsonb_build_object(
    'database', current_database(),
    'ledger_digest', (SELECT digest FROM ledger_state),
    'schema_digest', (SELECT digest FROM schema_state),
    'index_digest', (SELECT digest FROM index_state),
    'constraint_digest', (SELECT digest FROM constraint_state),
    'trigger_digest', (SELECT digest FROM trigger_state),
    'routine_digest', (SELECT digest FROM routine_state),
    'extension_state', (SELECT state FROM extension_state),
    'schema_acl_state', (SELECT state FROM schema_acl_state),
    'database_acl', (SELECT acl FROM database_acl_state),
    'role_membership_state', (SELECT state FROM role_membership_state),
    'sequence_state', (SELECT state FROM sequence_state),
    'sequence_probe_digest', (SELECT digest FROM sequence_probe_state),
    'large_object_digest', (SELECT digest FROM large_object_state),
    'materialized_view_digest', (SELECT digest FROM materialized_state),
    'partition_state', (SELECT state FROM partition_state),
    'partition_rows_digest', (SELECT digest FROM partition_rows_state),
    'unlogged_digest', (SELECT digest FROM unlogged_state),
    'role_state', (SELECT state FROM role_state),
    'acl_state', (SELECT state FROM acl_state)
)::text;

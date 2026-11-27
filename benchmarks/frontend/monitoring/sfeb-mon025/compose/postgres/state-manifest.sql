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
                        'volatile_cache', 'large_object_ref')
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
    'sequence_state', (SELECT state FROM sequence_state),
    'large_object_digest', (SELECT digest FROM large_object_state),
    'materialized_view_digest', (SELECT digest FROM materialized_state),
    'partition_state', (SELECT state FROM partition_state),
    'acl_state', (SELECT state FROM acl_state),
    'unlogged_row_count', (SELECT count(*) FROM public.volatile_cache)
)::text;

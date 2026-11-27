\set ON_ERROR_STOP on
SELECT jsonb_build_object(
    'database', current_database(),
    'server_version', current_setting('server_version'),
    'server_version_num', current_setting('server_version_num'),
    'wal_level', current_setting('wal_level'),
    'system_identifier', (SELECT system_identifier::text FROM pg_control_system()),
    'timeline_id', (
        SELECT timeline_id::text FROM pg_split_walfile_name(pg_walfile_name(pg_current_wal_lsn()))
    ),
    'publications', (
        SELECT coalesce(jsonb_agg(jsonb_build_object(
            'name', p.pubname,
            'all_tables', p.puballtables,
            'insert', p.pubinsert,
            'update', p.pubupdate,
            'delete', p.pubdelete,
            'truncate', p.pubtruncate
        ) ORDER BY p.pubname), '[]'::jsonb)
        FROM pg_publication AS p
    ),
    'slots', (
        SELECT coalesce(jsonb_agg(jsonb_build_object(
            'name', s.slot_name,
            'plugin', s.plugin,
            'database', s.database,
            'active', s.active,
            'restart_lsn', s.restart_lsn::text,
            'confirmed_flush_lsn', s.confirmed_flush_lsn::text,
            'wal_status', s.wal_status
        ) ORDER BY s.slot_name), '[]'::jsonb)
        FROM pg_replication_slots AS s
    )
)::text;

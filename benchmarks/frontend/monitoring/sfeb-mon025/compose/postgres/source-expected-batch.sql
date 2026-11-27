\set ON_ERROR_STOP on
SELECT COALESCE(
    json_agg(
        jsonb_build_object(
            'event_id', event_id,
            'batch_id', batch_id,
            'account_id', account_id,
            'amount', amount::text
        )
        ORDER BY event_id
    ),
    '[]'::json
)::text
FROM public.ledger
WHERE batch_id = :'batch_id';

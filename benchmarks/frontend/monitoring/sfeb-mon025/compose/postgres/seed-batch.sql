\set ON_ERROR_STOP on
BEGIN;
INSERT INTO public.ledger(event_id, batch_id, account_id, amount)
SELECT
    'mon025:' || :'batch_id' || ':' || lpad(n::text, 6, '0'),
    :'batch_id',
    'account-' || ((n % 7) + 1)::text,
    ((n % 10000)::numeric / 100) - 20
FROM generate_series(1, :batch_size::integer) AS g(n);
SELECT pg_logical_emit_message(true, 'mon025_transaction', :'batch_id');
COMMIT;

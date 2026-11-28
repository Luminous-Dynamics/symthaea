\set ON_ERROR_STOP on
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE public.ledger (
    event_id text PRIMARY KEY,
    batch_id text NOT NULL,
    account_id text NOT NULL,
    amount numeric(14,2) NOT NULL,
    amount_cents bigint GENERATED ALWAYS AS (round(amount * 100)::bigint) STORED,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX ledger_batch_idx ON public.ledger (batch_id, event_id);

CREATE SEQUENCE public.operation_sequence START 1000 CACHE 20;
CREATE TABLE public.sequence_probe (
    id bigint PRIMARY KEY DEFAULT nextval('public.operation_sequence'),
    probe_label text NOT NULL UNIQUE
);
ALTER SEQUENCE public.operation_sequence OWNED BY public.sequence_probe.id;

CREATE TABLE public.ledger_partitioned (
    id bigint NOT NULL,
    occurred_on date NOT NULL,
    amount numeric(14,2) NOT NULL,
    PRIMARY KEY (id, occurred_on)
) PARTITION BY RANGE (occurred_on);
CREATE TABLE public.ledger_partitioned_2026
    PARTITION OF public.ledger_partitioned
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
CREATE TABLE public.ledger_partitioned_default
    PARTITION OF public.ledger_partitioned DEFAULT;

CREATE UNLOGGED TABLE public.volatile_cache (
    cache_key text PRIMARY KEY,
    cache_value text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE public.large_object_ref (
    label text PRIMARY KEY,
    loid oid NOT NULL
);

CREATE MATERIALIZED VIEW public.ledger_summary AS
SELECT account_id, count(*) AS entry_count, sum(amount) AS total_amount
FROM public.ledger GROUP BY account_id;

CREATE ROLE mon025_auditor NOLOGIN;
GRANT USAGE ON SCHEMA public TO mon025_auditor;
GRANT SELECT ON public.ledger, public.ledger_partitioned, public.ledger_summary TO mon025_auditor;

INSERT INTO public.ledger(event_id, batch_id, account_id, amount)
VALUES
    ('baseline-001', 'baseline', 'account-a', 10.25),
    ('baseline-002', 'baseline', 'account-b', -3.50);
INSERT INTO public.sequence_probe(probe_label) VALUES ('baseline-sequence-probe');
INSERT INTO public.ledger_partitioned(id, occurred_on, amount)
VALUES (1, DATE '2026-11-27', 10.25);
INSERT INTO public.volatile_cache(cache_key, cache_value)
VALUES ('baseline-cache-key', 'baseline-cache-value');

DO $$
DECLARE
    object_id oid;
BEGIN
    object_id := lo_from_bytea(0, convert_to('mon025-large-object-fixture-v1', 'UTF8'));
    INSERT INTO public.large_object_ref(label, loid) VALUES ('fixture-v1', object_id);
END
$$;

REFRESH MATERIALIZED VIEW public.ledger_summary;

CREATE PUBLICATION mon025_pub FOR TABLE public.ledger;

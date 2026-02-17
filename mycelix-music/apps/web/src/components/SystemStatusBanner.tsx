import React, { useEffect, useState } from 'react';
import { StatusBanner } from './StatusBanner';

type Health = {
  db?: string;
  redis?: string;
  ipfs?: string;
  ceramic?: string;
  clock_skew_ms?: number;
  indexer_last_block?: number | string | null;
  indexer_lag_blocks?: number | string | null;
};

export function SystemStatusBanner({ className = '' }: { className?: string }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [indexer, setIndexer] = useState<{ last_block?: number | null; lag_blocks?: number | null; error?: string } | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch(`/health/details?client_ts=${Date.now()}`);
        const d = await resp.json();
        if (!cancelled) setHealth(d);
      } catch {
        // ignore errors
      }
      try {
        const resp = await fetch('/health/indexer');
        if (resp.ok) {
          const d = await resp.json();
          if (!cancelled) {
            setIndexer({
              last_block: typeof d?.last_block === 'number' ? d.last_block : null,
              lag_blocks: typeof d?.lag_blocks === 'number' ? d.lag_blocks : null,
              error: d?.error || undefined,
            });
          }
        }
      } catch {
        // ignore errors
      }
    })();
    return () => { cancelled = true; };
  }, []);

  if (dismissed) return null;
  if (!health) return null;

  const warnings: string[] = [];
  if (health.db !== 'ok') warnings.push('Database');
  if (health.redis !== 'ok') warnings.push('Redis');
  if (health.ipfs !== 'ok') warnings.push('IPFS');
  if (health.ceramic !== 'configured') warnings.push('Ceramic');
  const skewMs = typeof health.clock_skew_ms === 'number' ? Math.abs(health.clock_skew_ms) : 0;
  const skewSeconds = Math.round(skewMs / 1000);
  const skewWarning = skewSeconds > 5 ? `Clock skew ~${skewSeconds}s; signed requests may expire` : '';
  const indexerLag = typeof indexer?.lag_blocks === 'number'
    ? indexer?.lag_blocks
    : typeof health.indexer_lag_blocks === 'number'
      ? health.indexer_lag_blocks
      : null;
  const indexerErr = indexer?.error || (typeof health.indexer_last_block === 'string' && health.indexer_last_block.startsWith('error') ? 'Indexer last block unavailable' : undefined);
  const indexerWarning = indexerErr ? indexerErr : indexerLag && indexerLag > 5 ? `Indexer is ${indexerLag} blocks behind` : undefined;

  if (warnings.length === 0 && !skewWarning && !indexerWarning) return null;

  return (
    <StatusBanner variant="warning" dismissible onDismiss={() => setDismissed(true)} className={className}>
      System notice: {warnings.length > 0 ? `${warnings.join(', ')} ${warnings.length === 1 ? 'is' : 'are'} not fully configured.` : null}
      {skewWarning && ` ${skewWarning}.`}
      {indexerWarning && ` ${indexerWarning}.`}
    </StatusBanner>
  );
}

import type { Listing, RiskSignal } from '$types';

export async function analyzeRisk(listings: Listing[]): Promise<Record<string, RiskSignal>> {
  if (typeof Worker === 'undefined' || listings.length === 0) {
    return fallbackScore(listings);
  }

  try {
    const worker = new Worker(new URL('./workers/risk-analyzer.ts', import.meta.url), {
      type: 'module',
    });

    const response = await new Promise<Record<string, RiskSignal>>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Risk worker timeout')), 2000);
      worker.onmessage = (event: MessageEvent<{ scores: Record<string, RiskSignal> }>) => {
        clearTimeout(timer);
        resolve(event.data.scores);
        worker.terminate();
      };
      worker.onerror = (err) => {
        clearTimeout(timer);
        worker.terminate();
        reject(err as any);
      };

      worker.postMessage({ listings });
    });

    return response;
  } catch (error) {
    console.warn('Risk worker failed, falling back', error);
    return fallbackScore(listings);
  }
}

function fallbackScore(listings: Listing[]): Record<string, RiskSignal> {
  const scores: Record<string, RiskSignal> = {};
  listings.forEach((l) => {
    const id = l.listing_hash || l.id;
    const flags: string[] = [];
    let score = 0;
    if (!l.photos_ipfs_cids || l.photos_ipfs_cids.length === 0) {
      score += 0.3;
      flags.push('No photos provided');
    }
    scores[id] = { score, flags };
  });
  return scores;
}

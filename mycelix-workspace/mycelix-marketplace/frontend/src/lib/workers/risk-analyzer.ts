import type { Listing, RiskSignal } from '$types';

interface RiskRequest {
  listings: Listing[];
}

interface RiskResponse {
  scores: Record<string, RiskSignal>;
}

function scoreListings(listings: Listing[]): Record<string, RiskSignal> {
  if (!listings.length) return {};

  const prices = listings.map((l) => l.price);
  const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
  const variance = prices.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / prices.length;
  const std = Math.sqrt(variance || 1);

  const descriptions = listings.map((l) => ({
    id: l.listing_hash || l.id,
    text: (l.description || '').toLowerCase(),
  }));

  const descIndex = new Map<string, string[]>();
  descriptions.forEach((d) => {
    const tokens = d.text.split(/\s+/).filter(Boolean);
    tokens.forEach((t) => {
      const arr = descIndex.get(t) || [];
      arr.push(d.id);
      descIndex.set(t, arr);
    });
  });

  const result: Record<string, RiskSignal> = {};

  listings.forEach((listing) => {
    const id = listing.listing_hash || listing.id;
    const flags: string[] = [];
    let score = 0;

    // Price outlier
    const z = (listing.price - mean) / std;
    if (Math.abs(z) > 2.5) {
      score += 0.4;
      flags.push('Price outlier vs. marketplace average');
    }

    // Missing photos
    if (!listing.photos_ipfs_cids || listing.photos_ipfs_cids.length === 0) {
      score += 0.3;
      flags.push('No photos provided');
    }

    // Duplicate description tokens
    const tokens = (listing.description || '').toLowerCase().split(/\s+/).filter(Boolean);
    const duplicateHits = new Set<string>();
    tokens.forEach((t) => {
      const hits = descIndex.get(t) || [];
      if (hits.length > 2) duplicateHits.add(t);
    });
    if (duplicateHits.size > 5) {
      score += 0.2;
      flags.push('Description shares many tokens with other listings');
    }

    score = Math.min(1, score);
    result[id] = { score, flags };
  });

  return result;
}

self.onmessage = (event: MessageEvent<RiskRequest>) => {
  const { listings } = event.data;
  const scores = scoreListings(listings);
  const response: RiskResponse = { scores };
  (self as any).postMessage(response);
};

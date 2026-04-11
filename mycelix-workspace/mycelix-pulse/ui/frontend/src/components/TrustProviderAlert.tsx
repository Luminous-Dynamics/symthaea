import { useEffect, useState } from 'react';
import { api } from '@/services/api';

export default function TrustProviderAlert() {
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await api.getTrustHealth();
      } catch {
        if (!cancelled) setUnreachable(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (!unreachable) return null;

  return (
    <div className="mb-2" role="status" aria-live="polite">
      <div className="flex items-center space-x-2 px-3 py-2 bg-rose-50 dark:bg-rose-900/20 text-rose-700 dark:text-rose-100 border border-rose-200 dark:border-rose-800 rounded-md text-sm">
        <span className="font-semibold">Trust provider unreachable</span>
        <span className="text-xs">Using fallback trust; check TRUST_PROVIDER_URL or network.</span>
      </div>
    </div>
  );
}

import type { TrustSummary } from '@/store/trustStore';
import { toast } from '@/store/toastStore';
import { useRef } from 'react';

interface TrustGraphDrawerProps {
  open: boolean;
  onClose: () => void;
  sender: string;
  summary: TrustSummary & { attestations?: Attestation[] };
}

interface Attestation {
  from: string;
  to: string;
  weight: number;
  reason?: string;
}

const tierLabels: Record<string, string> = {
  high: 'High trust',
  medium: 'Neutral trust',
  low: 'Low trust',
  unknown: 'Unknown trust',
};

export function TrustGraphDrawer({ open, onClose, sender, summary }: TrustGraphDrawerProps) {
  if (!open) return null;

  const svgRef = useRef<SVGSVGElement | null>(null);
  const hopCount = summary.pathLength && summary.pathLength > 1 ? summary.pathLength : 2;
  const hops = Array.from({ length: hopCount }).map((_, idx) => ({
    id: idx + 1,
    label: idx === 0 ? 'You' : idx === hopCount - 1 ? sender : `Peer ${idx}`,
    trust: summary.score || 50 - idx * 5,
  }));
  const linkWeights = Array.from({ length: Math.max(hops.length - 1, 0) }).map((_, idx) => {
    const att = summary.attestations?.[idx];
    return att?.weight ?? 0.5;
  });

  const copyReport = async () => {
    const lines = [
      `Sender: ${sender}`,
      `Tier: ${summary.tier || 'unknown'}`,
      `Score: ${summary.score ?? '—'}`,
      `Path length: ${summary.pathLength ?? '—'}`,
      `Decay: ${summary.decayAt ? new Date(summary.decayAt).toLocaleString() : '—'}`,
      `Reasons: ${summary.reasons?.length ? summary.reasons.join(', ') : 'None provided'}`,
    ];
    if (summary.attestations && summary.attestations.length > 0) {
      lines.push('Attestations:');
      summary.attestations.forEach((att) => {
        lines.push(`- ${att.from} -> ${att.to} (weight ${att.weight}${att.reason ? `, ${att.reason}` : ''})`);
      });
    }
    try {
      await navigator.clipboard.writeText(lines.join('\n'));
      toast.success('Trust report copied');
    } catch (err) {
      toast.error('Failed to copy trust report');
    }
  };

  const exportJson = () => {
    const payload = {
      sender,
      summary,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trust-summary-${sender.replace(/[^a-zA-Z0-9-_]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Trust JSON exported');
  };

  const exportMarkdown = () => {
    const lines = [
      `# Trust Report for ${sender}`,
      '',
      `- Tier: ${summary.tier || 'unknown'}`,
      `- Score: ${summary.score ?? '–'}`,
      `- Path length: ${summary.pathLength ?? '–'}`,
      `- Decay: ${summary.decayAt ? new Date(summary.decayAt).toLocaleString() : '–'}`,
      `- Reasons: ${summary.reasons?.length ? summary.reasons.join(', ') : 'None provided'}`,
    ];
    if (summary.attestations && summary.attestations.length > 0) {
      lines.push(`- Attestations:`);
      summary.attestations.forEach((att) => {
        lines.push(`  - ${att.from} -> ${att.to} (weight ${att.weight}${att.reason ? `, ${att.reason}` : ''})`);
      });
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trust-summary-${sender.replace(/[^a-zA-Z0-9-_]/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Trust Markdown exported');
  };

  const exportPng = () => {
    const svg = svgRef.current;
    if (!svg) {
      toast.error('Unable to export graph');
      return;
    }
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svg);
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 500;
      canvas.height = 120;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((pngBlob) => {
        if (!pngBlob) {
          URL.revokeObjectURL(url);
          return;
        }
        const pngUrl = URL.createObjectURL(pngBlob);
        const a = document.createElement('a');
        a.href = pngUrl;
        a.download = `trust-graph-${sender.replace(/[^a-zA-Z0-9-_]/g, '_')}.png`;
        a.click();
        URL.revokeObjectURL(pngUrl);
      });
      URL.revokeObjectURL(url);
      toast.success('Trust graph PNG exported');
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      toast.error('Failed to export graph');
    };
    image.src = url;
  };

  return (
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="absolute right-0 top-0 h-full w-full max-w-md bg-white dark:bg-gray-900 shadow-xl p-6 overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Trust Path</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">{sender}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
            aria-label="Close trust drawer"
          >
            ✕
          </button>
        </div>

        <div className="mb-3">
          <button
            onClick={copyReport}
            className="px-3 py-1.5 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600"
          >
            Copy trust report
          </button>
          <button
            onClick={exportJson}
            className="ml-2 px-3 py-1.5 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600"
          >
            Export JSON
          </button>
          <button
            onClick={exportMarkdown}
            className="ml-2 px-3 py-1.5 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600"
          >
            Export Markdown
          </button>
          <button
            onClick={exportPng}
            className="ml-2 px-3 py-1.5 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600"
          >
            Export PNG
          </button>
        </div>

        <div className="space-y-3 mb-4">
          <div className="text-sm text-gray-700 dark:text-gray-300">
            <span className="font-semibold">Tier:</span> {tierLabels[summary.tier || 'unknown']} •{' '}
            <span className="font-semibold">Score:</span> {summary.score ?? '–'}
          </div>
          <div className="text-sm text-gray-700 dark:text-gray-300">
            <span className="font-semibold">Reasons:</span> {summary.reasons?.length ? summary.reasons.join(', ') : 'Not provided'}
          </div>
          {summary.decayAt && (
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Decays: {new Date(summary.decayAt).toLocaleString()}
            </div>
          )}
        </div>

        <div className="space-y-3 mb-6">
          {/* Simple node-link visual */}
          <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <svg ref={svgRef} width="100%" height="100" viewBox="0 0 100 20" preserveAspectRatio="none">
              {hops.map((hop, idx) => {
                const x = (idx / Math.max(hops.length - 1, 1)) * 100;
                const y = 10;
                const weight = linkWeights[idx] ?? 0.5;
                const stroke = weight >= 0.8 ? '#34d399' : weight >= 0.6 ? '#fbbf24' : '#f87171';
                const strokeWidth = 1 + weight * 2;
                return (
                  <g key={hop.id}>
                    {idx < hops.length - 1 && (
                      <line
                        x1={x + 2}
                        y1={y}
                        x2={((idx + 1) / Math.max(hops.length - 1, 1)) * 100 - 2}
                        y2={y}
                        stroke={stroke}
                        strokeWidth={strokeWidth}
                        strokeDasharray="1.5 2"
                      />
                    )}
                    <circle
                      cx={x}
                      cy={y}
                      r={2.8}
                      fill={summary.tier === 'low' ? '#fca5a5' : summary.tier === 'medium' ? '#fbbf24' : '#34d399'}
                      stroke="#1f2937"
                      strokeWidth="0.4"
                    />
                  </g>
                );
              })}
            </svg>
            <div className="grid grid-cols-1 gap-2 mt-3">
              {hops.map((hop, idx) => (
                <div key={hop.id} className="flex items-center justify-between text-sm text-gray-900 dark:text-gray-100">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-primary-100 dark:bg-primary-900/40 text-primary-700 dark:text-primary-200">
                      {idx + 1}
                    </span>
                    <span className="font-semibold">{hop.label}</span>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    Score: {hop.trust}
                    {idx < linkWeights.length && (
                      <span className="ml-2 text-[11px] text-gray-500 dark:text-gray-400">
                        Link weight: {linkWeights[idx].toFixed(2)}
                      </span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">Attestations</h4>
          {summary.attestations && summary.attestations.length > 0 ? (
            <div className="space-y-2">
              {summary.attestations.map((att, idx) => (
                <div
                  key={`${att.from}-${att.to}-${idx}`}
                  className="flex items-center justify-between rounded-md border border-gray-200 dark:border-gray-700 px-3 py-2 bg-white dark:bg-gray-800"
                >
                  <div>
                    <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {att.from} → {att.to}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      Weight: {att.weight} {att.reason ? `• ${att.reason}` : ''}
                    </div>
                  </div>
                  <div className="w-2 h-2 rounded-full bg-emerald-500" aria-hidden />
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-600 dark:text-gray-400">No attestations provided.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default TrustGraphDrawer;

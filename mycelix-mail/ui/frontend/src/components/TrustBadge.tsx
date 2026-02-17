import type { TrustSummary, TrustTier } from '@/store/trustStore';

interface TrustBadgeProps {
  summary: TrustSummary;
  compact?: boolean;
  className?: string;
}

const tierCopy: Record<TrustTier, { label: string; color: string; text: string; border: string }> = {
  high: {
    label: 'High trust',
    color: 'bg-emerald-100 dark:bg-emerald-900/30',
    text: 'text-emerald-700 dark:text-emerald-200',
    border: 'border-emerald-200 dark:border-emerald-800',
  },
  medium: {
    label: 'Neutral trust',
    color: 'bg-amber-100 dark:bg-amber-900/30',
    text: 'text-amber-700 dark:text-amber-200',
    border: 'border-amber-200 dark:border-amber-800',
  },
  low: {
    label: 'Low trust',
    color: 'bg-rose-100 dark:bg-rose-900/30',
    text: 'text-rose-700 dark:text-rose-200',
    border: 'border-rose-200 dark:border-rose-800',
  },
  unknown: {
    label: 'Unknown trust',
    color: 'bg-gray-100 dark:bg-gray-800',
    text: 'text-gray-700 dark:text-gray-300',
    border: 'border-gray-200 dark:border-gray-700',
  },
};

export function TrustBadge({ summary, compact = false, className = '' }: TrustBadgeProps) {
  const { tier, score, reasons } = summary;
  const copy = tierCopy[tier];
  const reasonText = reasons?.length ? reasons.join(', ') : 'No trust signals provided';
  const displayScore = typeof score === 'number' ? `${score}` : '–';

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium transition-colors ${copy.color} ${copy.text} ${copy.border} ${className}`}
      title={`${copy.label}${reasons?.length ? ` • ${reasonText}` : ''}`}
    >
      <span className="mr-1 h-2 w-2 rounded-full bg-current" aria-hidden />
      {!compact && <span className="mr-1">{copy.label}</span>}
      <span className="text-[11px] opacity-80">Score {displayScore}</span>
    </span>
  );
}

export default TrustBadge;

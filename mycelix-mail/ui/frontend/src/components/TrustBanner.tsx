interface TrustBannerProps {
  status: 'unreachable';
  message: string;
}

export function TrustBanner({ status, message }: TrustBannerProps) {
  if (status !== 'unreachable') return null;

  return (
    <div
      className="flex items-center space-x-2 px-3 py-2 bg-rose-50 dark:bg-rose-900/20 text-rose-700 dark:text-rose-100 border border-rose-200 dark:border-rose-800 rounded-md text-sm"
      role="status"
      aria-live="polite"
    >
      <span className="font-semibold">Trust provider unreachable</span>
      <span className="text-xs">{message}</span>
    </div>
  );
}

export default TrustBanner;

export default function TrustHowItWorks() {
  return (
    <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm space-y-2">
      <p className="text-base font-semibold text-gray-900 dark:text-gray-100">How trust works</p>
      <p className="text-gray-700 dark:text-gray-300">
        Trust summaries come from your MATL/Holochain provider (if configured) or a deterministic fallback. Scores map to tiers (high/medium/low) and drive badges, quarantine, and notifications.
      </p>
      <ul className="list-disc pl-5 space-y-1 text-gray-700 dark:text-gray-300">
        <li><span className="font-semibold">Policies:</span> Strict quarantines low-trust and suppresses notifications; Balanced (default) respects trust tier; Open minimizes auto-quarantine.</li>
        <li><span className="font-semibold">TTL:</span> Trust data is cached; adjust TTL in settings to refresh more/less often.</li>
        <li><span className="font-semibold">Overrides:</span> Allowlist senders to bypass low-trust; manage in Trust Center.</li>
        <li><span className="font-semibold">Attestations:</span> Show issuers/weights; request new attestations to promote quarantined mail.</li>
        <li><span className="font-semibold">Provider:</span> Set TRUST_PROVIDER_URL to enable live trust; health is shown here and in the inbox banner.</li>
      </ul>
    </div>
  );
}

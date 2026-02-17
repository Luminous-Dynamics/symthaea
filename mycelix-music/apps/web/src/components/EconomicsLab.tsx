import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Clipboard, Play } from 'lucide-react';
import { useToast } from './ToastProvider';

type ModuleKey = 'pay_per_stream' | 'gift' | 'dynamic' | 'loyalty';

type Module = {
  id: ModuleKey;
  name: string;
  accent: string;
  description: string;
};

const MODULES: Module[] = [
  {
    id: 'pay_per_stream',
    name: 'Pay Per Stream',
    accent: 'from-blue-500 to-cyan-400',
    description: 'Predictable per-play revenue, high signal for established catalogs.',
  },
  {
    id: 'gift',
    name: 'Gift Economy',
    accent: 'from-emerald-500 to-lime-400',
    description: 'Zero-floor access with voluntary tips and CGC rewards.',
  },
  {
    id: 'dynamic',
    name: 'Dynamic Pricing',
    accent: 'from-amber-500 to-orange-500',
    description: 'Surge/loyalty pricing that adapts to demand in real time.',
  },
  {
    id: 'loyalty',
    name: 'Loyalty Boost',
    accent: 'from-fuchsia-500 to-purple-500',
    description: 'Repeat listener multipliers and early-adopter bonuses.',
  },
];

type Split = { role: string; pct: number };

const BASE_SPLITS: Split[] = [
  { role: 'Artist', pct: 72 },
  { role: 'Collaborators', pct: 18 },
  { role: 'Protocol', pct: 10 },
];

function adjustSplits(modules: ModuleKey[]): Split[] {
  const splits = [...BASE_SPLITS];
  if (modules.includes('gift')) {
    splits[2].pct = 6;
    splits.push({ role: 'Community Vault', pct: 4 });
  }
  if (modules.includes('dynamic')) {
    splits.push({ role: 'Dynamic Buffer', pct: 5 });
  }
  const total = splits.reduce((s, r) => s + r.pct, 0);
  return splits.map((r) => ({ ...r, pct: Number(((r.pct / total) * 100).toFixed(2)) }));
}

function simulatePayment(modules: ModuleKey[], amount: number, loyaltyBoost: number, listeners: number) {
  const protocolFee = modules.includes('dynamic') ? 0.012 : 0.01;
  const loyaltyMultiplier = modules.includes('loyalty') ? loyaltyBoost : 1;
  const dynamicLift = modules.includes('dynamic') ? 1 + Math.min(0.25, Math.log(listeners + 1) / 10) : 1;
  const effectiveAmount = amount * loyaltyMultiplier * dynamicLift;
  const net = effectiveAmount * (1 - protocolFee);
  const splits = adjustSplits(modules).map((s) => ({
    ...s,
    payout: Number((net * (s.pct / 100)).toFixed(4)),
  }));
  return {
    gross: Number((amount * loyaltyMultiplier * dynamicLift).toFixed(4)),
    protocolFee: Number((amount * loyaltyMultiplier * dynamicLift * protocolFee).toFixed(4)),
    net: Number(net.toFixed(4)),
    splits,
    loyaltyMultiplier,
    dynamicLift: Number(dynamicLift.toFixed(3)),
  };
}

type OfferRule = {
  id: string;
  title: string;
  condition: string;
  action: string;
};

export function EconomicsLab() {
  const { show } = useToast();
  const [modules, setModules] = useState<ModuleKey[]>(['pay_per_stream', 'loyalty']);
  const [amount, setAmount] = useState(0.01);
  const [listeners, setListeners] = useState(1000);
  const [loyaltyBoost, setLoyaltyBoost] = useState(1.2);
  const [rules, setRules] = useState<OfferRule[]>([
    { id: 'r1', title: 'First 500 plays', condition: 'listener.plays < 500', action: 'price = 0' },
    { id: 'r2', title: 'Holders of NFT X', condition: 'listener.nfts.includes("0xNFTX")', action: 'price = 0.005; loyalty = 1.3x' },
  ]);
  const [dynamicLift, setDynamicLift] = useState(0.05);
  const [dynamicMax, setDynamicMax] = useState(0.2);
  const [dynamicCurve, setDynamicCurve] = useState(15);
  const [loyaltyLift, setLoyaltyLift] = useState(0.03);
  const [loyaltyMax, setLoyaltyMax] = useState(0.08);
  const [impacts, setImpacts] = useState<{ name: string; lift: number; netDelta?: number }[]>([]);
  const [previewResult, setPreviewResult] = useState<{ base: { totalPlays: number; totalNet: number }; simulated: { totalPlays: number; totalNet: number; delta: number } } | null>(null);
  const [perType, setPerType] = useState<{ payment_type: string; plays: number; net: number }[]>([]);

  const sim = useMemo(
    () => simulatePayment(modules, amount, loyaltyBoost, listeners),
    [modules, amount, loyaltyBoost, listeners],
  );

  function toggleModule(id: ModuleKey) {
    setModules((prev) => (prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]));
  }

  function addRule() {
    const next: OfferRule = {
      id: `r${Date.now()}`,
      title: 'New rule',
      condition: 'listener.plays >= 0',
      action: 'price = 0.01',
    };
    setRules((r) => [...r, next]);
  }

  function updateRule(id: string, patch: Partial<OfferRule>) {
    setRules((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }

  function deleteRule(id: string) {
    setRules((rs) => rs.filter((r) => r.id !== id));
  }

  const exportConfig = () => {
    const payload = {
      modules,
      pricing: {
        baseAmount: amount,
        loyaltyMultiplier: loyaltyBoost,
        dynamicLift,
        loyaltyLift,
        dynamicMax,
        dynamicCurve,
        loyaltyMax,
      },
      programmableOffers: rules.map(({ title, condition, action }) => ({ title, condition, action })),
      splits: sim.splits,
    };
    navigator.clipboard.writeText(JSON.stringify(payload, null, 2)).then(() => {
      show('Exported JSON to clipboard', 'success');
    }).catch(() => show('Failed to copy', 'error'));
  };

  const downloadConfig = () => {
    const payload = {
      modules,
      pricing: {
        baseAmount: amount,
        loyaltyMultiplier: loyaltyBoost,
        dynamicLift,
        loyaltyLift,
      },
      programmableOffers: rules.map(({ title, condition, action }) => ({ title, condition, action })),
      splits: sim.splits,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mycelix-strategy.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const triggerPreview = async () => {
    try {
      const resp = await fetch('/api/strategy-configs/latest');
      if (!resp.ok) {
        const msg = resp.status === 403 ? 'Preview restricted to admins' : 'No published config';
        throw new Error(msg);
      }
      const cfg = await resp.json();
      const preview = await fetch(`/api/strategy-configs/${cfg.id}/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days: 30, modules }),
      });
      if (!preview.ok) {
        const msg = preview.status === 403 ? 'Preview restricted to admins' : 'Preview failed';
        throw new Error(msg);
      }
      const data = await preview.json();
      const { base, simulated } = data.preview;
      const pct = base.totalNet > 0 ? (simulated.delta / base.totalNet) * 100 : 0;
      show(`Preview: base ${base.totalPlays} plays / ${Number(base.totalNet).toFixed(2)} net → simulated ${Number(simulated.totalNet).toFixed(2)} (${simulated.delta >= 0 ? '+' : ''}${simulated.delta.toFixed(2)} | ${pct.toFixed(1)}%)`, 'success');
      setImpacts(data.preview.impacts || []);
      setPreviewResult({ base, simulated });
      setPerType(data.preview.perType || []);
    } catch (e: any) {
      show(e.message || 'Preview failed', 'error');
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
      <div className="lg:col-span-1 space-y-4">
        <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
          <h2 className="text-xl font-semibold mb-3">Module Palette</h2>
          <p className="text-gray-600 text-sm mb-4">Compose your economic graph. Toggle modules and see payouts reflow instantly.</p>
          <div className="space-y-3">
            {MODULES.map((m) => (
              <button
                key={m.id}
                onClick={() => toggleModule(m.id)}
                className={`w-full flex items-start gap-3 p-4 rounded-xl border transition ${
                  modules.includes(m.id) ? 'border-purple-500 bg-purple-50' : 'border-gray-200 hover:border-purple-300'
                }`}
              >
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${m.accent}`} />
                <div className="text-left">
                  <div className="font-semibold">{m.name}</div>
                  <div className="text-sm text-gray-600">{m.description}</div>
                </div>
                <span className={`ml-auto text-xs px-2 py-1 rounded-full ${modules.includes(m.id) ? 'bg-purple-600 text-white' : 'bg-gray-100 text-gray-700'}`}>
                  {modules.includes(m.id) ? 'Active' : 'Add'}
                </span>
              </button>
            ))}
          </div>
        </div>

        {impacts.length > 0 && (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
            <h4 className="text-lg font-semibold mb-2">Preview Breakdown</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {impacts.map((imp) => (
                <div key={imp.name} className="p-3 rounded-lg border border-gray-200">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold capitalize">{imp.name}</span>
                    <span className="text-sm text-gray-500">{((imp.lift - 1) * 100).toFixed(2)}%</span>
                  </div>
                  {typeof imp.netDelta === 'number' && (
                    <div className="text-sm text-gray-600">
                      Net delta: {imp.netDelta >= 0 ? '+' : ''}{imp.netDelta.toFixed(2)}
                    </div>
                  )}
                  <div className="mt-2 h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-emerald-500"
                      style={{ width: `${Math.min(100, Math.max(0, (imp.lift - 1) * 400))}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {previewResult && (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
            <h4 className="text-lg font-semibold mb-2">Last Preview</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg border border-gray-200">
                <div className="text-sm text-gray-500">Base</div>
                <div className="text-lg font-semibold">{previewResult.base.totalPlays} plays</div>
                <div className="text-sm text-gray-600">{previewResult.base.totalNet.toFixed(2)} net</div>
              </div>
              <div className="p-3 rounded-lg border border-gray-200">
                <div className="text-sm text-gray-500">Simulated</div>
                <div className="text-lg font-semibold">{previewResult.simulated.totalPlays} plays</div>
                <div className="text-sm text-gray-600">{previewResult.simulated.totalNet.toFixed(2)} net</div>
              </div>
              <div className="p-3 rounded-lg border border-gray-200">
                <div className="text-sm text-gray-500">Delta</div>
                <div className="text-lg font-semibold">{previewResult.simulated.delta >= 0 ? '+' : ''}{previewResult.simulated.delta.toFixed(2)}</div>
              </div>
            </div>
            {perType.length > 0 && (
              <div className="mt-4">
                <div className="text-sm font-medium text-gray-700 mb-2">By payment type</div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {perType.map((row) => (
                    <div key={row.payment_type} className="p-3 rounded-lg border border-gray-200">
                      <div className="text-sm text-gray-500">{row.payment_type}</div>
                      <div className="text-lg font-semibold">{row.plays} plays</div>
                      <div className="text-sm text-gray-600">{Number(row.net || 0).toFixed(2)} net</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="bg-gray-900 text-white rounded-2xl shadow-lg p-6 border border-white/10">
          <h3 className="text-lg font-semibold mb-2">Signals</h3>
          <ul className="space-y-2 text-sm text-gray-200">
            <li>• Protocol fee auto-adjusts with dynamic pricing.</li>
            <li>• Loyalty boosts stack with surge demand (capped at 1.25x).</li>
            <li>• Community vault only funds when Gift module is active.</li>
          </ul>
        </div>
      </div>

      <div className="lg:col-span-2 space-y-6">
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-4">
            <div>
              <div className="text-sm uppercase tracking-wide text-gray-500">Economics Lab</div>
              <h2 className="text-2xl font-bold text-gray-900">Strategy Composer</h2>
              <p className="text-gray-600">Set payment assumptions, then watch splits ripple through the stack.</p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-500">Active Modules</span>
              <div className="flex flex-wrap gap-2">
                {modules.map((m) => {
                  const mod = MODULES.find((x) => x.id === m)!;
                  return (
                    <span key={m} className="px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs">{mod.name}</span>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-700">Payment per listener (FLOW)</label>
              <input
                type="number"
                min="0"
                step="0.001"
                value={amount}
                onChange={(e) => setAmount(Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-gray-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700">Active listeners (window)</label>
              <input
                type="number"
                min="0"
                step="10"
                value={listeners}
                onChange={(e) => setListeners(Number(e.target.value))}
                className="mt-1 w-full rounded-lg border border-gray-200 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <div className="col-span-1 md:col-span-2">
              <label className="text-sm font-medium text-gray-700 flex items-center justify-between">
                Loyalty multiplier
                <span className="text-xs text-gray-500">{loyaltyBoost.toFixed(2)}x</span>
              </label>
              <input
                type="range"
                min={1}
                max={1.5}
                step={0.01}
                value={loyaltyBoost}
                onChange={(e) => setLoyaltyBoost(Number(e.target.value))}
                className="w-full accent-purple-600 mt-2"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700 flex items-center justify-between">
                Dynamic lift (additional)
                <span className="text-xs text-gray-500">{(dynamicLift * 100).toFixed(1)}%</span>
              </label>
              <input
                type="range"
                min={0}
                max={0.2}
                step={0.01}
                value={dynamicLift}
                onChange={(e) => setDynamicLift(Number(e.target.value))}
                className="w-full accent-amber-500 mt-2"
              />
              <div className="mt-2 text-xs text-gray-500">Max surge {(dynamicMax * 100).toFixed(0)}%, curve {dynamicCurve}</div>
              <div className="grid grid-cols-2 gap-2 mt-2">
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.01}
                  value={dynamicMax}
                  onChange={(e) => setDynamicMax(Number(e.target.value))}
                  className="w-full rounded-md border border-gray-200 px-2 py-1 text-sm"
                  placeholder="Max surge"
                />
                <input
                  type="number"
                  min={1}
                  max={50}
                  step={1}
                  value={dynamicCurve}
                  onChange={(e) => setDynamicCurve(Number(e.target.value))}
                  className="w-full rounded-md border border-gray-200 px-2 py-1 text-sm"
                  placeholder="Curve"
                />
              </div>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700 flex items-center justify-between">
                Loyalty lift (additional)
                <span className="text-xs text-gray-500">{(loyaltyLift * 100).toFixed(1)}%</span>
              </label>
              <input
                type="range"
                min={0}
                max={0.1}
                step={0.005}
                value={loyaltyLift}
                onChange={(e) => setLoyaltyLift(Number(e.target.value))}
                className="w-full accent-emerald-500 mt-2"
              />
              <div className="mt-2 text-xs text-gray-500">Max loyalty {(loyaltyMax * 100).toFixed(0)}%</div>
              <input
                type="number"
                min={0}
                max={0.5}
                step={0.01}
                value={loyaltyMax}
                onChange={(e) => setLoyaltyMax(Number(e.target.value))}
                className="w-full rounded-md border border-gray-200 px-2 py-1 text-sm mt-2"
                placeholder="Max loyalty"
              />
            </div>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">
            <StatCard label="Gross (with surge/loyalty)" value={`${sim.gross.toFixed(4)} FLOW`} tone="purple" />
            <StatCard label="Protocol fee" value={`${sim.protocolFee.toFixed(4)} FLOW`} tone="gray" />
            <StatCard label="Net to distribute" value={`${sim.net.toFixed(4)} FLOW`} tone="emerald" />
          </div>
        </div>

        <div className="bg-gray-900 text-white rounded-2xl shadow-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold">Splits Preview</h3>
            <div className="flex items-center gap-2">
              <div className="text-sm text-gray-300">
                Loyalty: {sim.loyaltyMultiplier.toFixed(2)}x • Surge: {sim.dynamicLift.toFixed(2)}x
              </div>
              <button
                onClick={exportConfig}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/10 border border-white/20 text-xs hover:bg-white/20"
              >
                <Clipboard className="w-4 h-4" />
                Export JSON
              </button>
              <button
                onClick={downloadConfig}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/10 border border-white/20 text-xs hover:bg-white/20"
              >
                Download
              </button>
              <button
                onClick={triggerPreview}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-purple-600 text-white text-xs hover:bg-purple-700"
              >
                <Play className="w-4 h-4" />
                Preview (30d)
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {sim.splits.map((s) => (
              <motion.div
                key={s.role}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25 }}
                className="rounded-xl bg-white/5 border border-white/10 p-4"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm text-gray-300">{s.role}</div>
                    <div className="text-lg font-semibold text-white">{s.payout.toFixed(4)} FLOW</div>
                  </div>
                  <div className="text-sm text-emerald-300">{s.pct}%</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="text-sm uppercase tracking-wide text-gray-500">Programmable Offers</div>
              <h3 className="text-xl font-semibold">Rules</h3>
            </div>
            <button
              onClick={addRule}
              className="px-3 py-2 rounded-lg bg-purple-600 text-white text-sm hover:bg-purple-700"
            >
              Add rule
            </button>
          </div>
          <div className="space-y-3">
            {rules.map((r) => (
              <div key={r.id} className="rounded-lg border border-gray-200 p-4">
                <div className="flex items-center justify-between mb-2">
                  <input
                    value={r.title}
                    onChange={(e) => updateRule(r.id, { title: e.target.value })}
                    className="text-base font-semibold w-full border-0 focus:ring-0"
                  />
                  <button onClick={() => deleteRule(r.id)} className="text-sm text-red-500 ml-2">Delete</button>
                </div>
                <label className="text-xs uppercase text-gray-500">Condition</label>
                <input
                  value={r.condition}
                  onChange={(e) => updateRule(r.id, { condition: e.target.value })}
                  className="w-full mt-1 rounded-md border border-gray-200 px-3 py-2 text-sm"
                  placeholder='listener.nfts.includes("0x...")'
                />
                <label className="text-xs uppercase text-gray-500 mt-3 block">Action</label>
                <input
                  value={r.action}
                  onChange={(e) => updateRule(r.id, { action: e.target.value })}
                  className="w-full mt-1 rounded-md border border-gray-200 px-3 py-2 text-sm"
                  placeholder="price = 0.005; loyalty = 1.3x"
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, tone }: { label: string; value: string; tone: 'purple' | 'gray' | 'emerald' }) {
  const toneClass = tone === 'purple'
    ? 'bg-purple-600/10 text-purple-200'
    : tone === 'emerald'
      ? 'bg-emerald-600/10 text-emerald-200'
      : 'bg-white/5 text-gray-200';
  return (
    <div className={`rounded-xl p-4 border border-white/10 ${toneClass}`}>
      <div className="text-sm">{label}</div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  );
}

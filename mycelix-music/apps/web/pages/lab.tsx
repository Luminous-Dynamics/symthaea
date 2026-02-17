import Navigation from '../components/Navigation';
import { EconomicsLab } from '../src/components/EconomicsLab';

export default function LabPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-900 text-white">
      <Navigation />
      <main className="pt-20 max-w-6xl mx-auto px-4 pb-16">
        <div className="bg-white/5 border border-white/10 rounded-3xl p-8 shadow-2xl">
          <div className="mb-4">
            <p className="text-sm uppercase tracking-[0.2em] text-purple-200">Economics Lab</p>
            <h1 className="text-3xl font-bold">Strategy Composer & Simulation</h1>
            <p className="text-gray-200 max-w-3xl">
              Wire together payment primitives, loyalty boosts, and dynamic pricing—then preview how value flows to every stakeholder before you deploy on-chain.
            </p>
          </div>
          <EconomicsLab />
        </div>
      </main>
    </div>
  );
}

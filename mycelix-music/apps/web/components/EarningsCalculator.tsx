import { useState } from 'react';
import { TrendingUp, DollarSign, Music } from 'lucide-react';

export default function EarningsCalculator() {
  const [monthlyPlays, setMonthlyPlays] = useState(10000);

  // Calculate earnings for different platforms
  const spotifyEarnings = monthlyPlays * 0.003;
  const mycelixPayPerStream = monthlyPlays * 0.01;
  const mycelixFreemium = monthlyPlays * 0.007; // Assume 30% use free plays
  const mycelixPayWhatYouWant = monthlyPlays * 0.15; // Average $0.15 tip per play
  const mycelixPatronage = Math.floor(monthlyPlays / 100) * 10; // 1% become $10/mo patrons

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  return (
    <div className="bg-gradient-to-br from-purple-900/50 to-blue-900/50 rounded-2xl p-8 backdrop-blur-sm border border-white/10">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">
          💰 See What YOU Could Earn
        </h2>
        <p className="text-gray-300">
          Compare your potential earnings across platforms
        </p>
      </div>

      {/* Slider Input */}
      <div className="mb-8">
        <label className="block text-white text-lg font-semibold mb-4 text-center">
          Your Monthly Plays: {monthlyPlays.toLocaleString()}
        </label>
        <input
          type="range"
          min="1000"
          max="100000"
          step="1000"
          value={monthlyPlays}
          onChange={(e) => setMonthlyPlays(parseInt(e.target.value))}
          className="w-full h-3 bg-white/20 rounded-lg appearance-none cursor-pointer
                     [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-6
                     [&::-webkit-slider-thumb]:h-6 [&::-webkit-slider-thumb]:bg-purple-500
                     [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer
                     [&::-moz-range-thumb]:w-6 [&::-moz-range-thumb]:h-6
                     [&::-moz-range-thumb]:bg-purple-500 [&::-moz-range-thumb]:rounded-full
                     [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-0"
        />
        <div className="flex justify-between text-sm text-gray-400 mt-2">
          <span>1K plays</span>
          <span>50K plays</span>
          <span>100K plays</span>
        </div>
      </div>

      {/* Earnings Comparison */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Spotify */}
        <div className="bg-black/30 rounded-xl p-6 border border-red-500/30">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-white">Spotify</h3>
            <Music className="w-6 h-6 text-red-400" />
          </div>
          <div className="text-4xl font-bold text-red-400 mb-2">
            {formatCurrency(spotifyEarnings)}
          </div>
          <p className="text-gray-400 text-sm">$0.003 per stream</p>
          <p className="text-gray-400 text-sm mt-2">💸 90-day payment delay</p>
        </div>

        {/* Mycelix - Pay Per Stream */}
        <div className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl p-6 border border-blue-400/50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-white">Mycelix Pay-Per-Stream</h3>
            <TrendingUp className="w-6 h-6 text-blue-400" />
          </div>
          <div className="text-4xl font-bold text-blue-400 mb-2">
            {formatCurrency(mycelixPayPerStream)}
          </div>
          <p className="text-green-400 text-sm font-semibold">
            {((mycelixPayPerStream / spotifyEarnings - 1) * 100).toFixed(0)}% MORE than Spotify
          </p>
          <p className="text-gray-300 text-sm mt-2">⚡ Instant payment</p>
        </div>

        {/* Mycelix - Freemium */}
        <div className="bg-gradient-to-br from-orange-500/20 to-yellow-500/20 rounded-xl p-6 border border-orange-400/50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-white">Mycelix Freemium</h3>
            <DollarSign className="w-6 h-6 text-orange-400" />
          </div>
          <div className="text-4xl font-bold text-orange-400 mb-2">
            {formatCurrency(mycelixFreemium)}
          </div>
          <p className="text-green-400 text-sm font-semibold">
            {((mycelixFreemium / spotifyEarnings - 1) * 100).toFixed(0)}% MORE than Spotify
          </p>
          <p className="text-gray-300 text-sm mt-2">🎵 3 free plays, then $0.01</p>
        </div>

        {/* Mycelix - Pay What You Want */}
        <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl p-6 border border-green-400/50 relative overflow-hidden">
          <div className="absolute top-2 right-2 bg-yellow-500 text-black text-xs font-bold px-2 py-1 rounded">
            BEST VALUE
          </div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-white">Mycelix Pay-What-You-Want</h3>
            <DollarSign className="w-6 h-6 text-green-400" />
          </div>
          <div className="text-4xl font-bold text-green-400 mb-2">
            {formatCurrency(mycelixPayWhatYouWant)}
          </div>
          <p className="text-green-400 text-sm font-semibold">
            {((mycelixPayWhatYouWant / spotifyEarnings - 1) * 100).toFixed(0)}% MORE than Spotify! 🔥
          </p>
          <p className="text-gray-300 text-sm mt-2">💚 Fans tip avg $0.15</p>
        </div>
      </div>

      {/* Patronage Bonus */}
      {mycelixPatronage > 100 && (
        <div className="mt-6 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-6 border border-purple-400/50">
          <h3 className="text-xl font-bold text-white mb-2">
            💎 Patronage Bonus
          </h3>
          <p className="text-gray-300 mb-3">
            If just 1% of your listeners become patrons at $10/month:
          </p>
          <div className="text-3xl font-bold text-purple-400">
            +{formatCurrency(mycelixPatronage)}/month recurring
          </div>
          <p className="text-gray-400 text-sm mt-2">
            That's {formatCurrency(mycelixPatronage * 12)}/year in stable income!
          </p>
        </div>
      )}

      {/* Call to Action */}
      <div className="mt-8 text-center">
        <button className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-4 rounded-xl font-bold text-lg hover:from-purple-700 hover:to-pink-700 transition shadow-lg shadow-purple-500/50">
          Join the Waitlist - Early Artists Get Priority
        </button>
        <p className="text-gray-400 text-sm mt-3">
          🎸 First 100 artists get <span className="text-green-400 font-semibold">ZERO platform fees</span> for 6 months
        </p>
      </div>

      {/* Social Proof */}
      <div className="mt-6 flex items-center justify-center space-x-8 text-sm text-gray-400">
        <div>
          <span className="text-2xl font-bold text-white">$15,234</span>
          <p>Already paid to artists</p>
        </div>
        <div>
          <span className="text-2xl font-bold text-white">47</span>
          <p>Artists on platform</p>
        </div>
        <div>
          <span className="text-2xl font-bold text-white">1,234</span>
          <p>Active listeners</p>
        </div>
      </div>
    </div>
  );
}

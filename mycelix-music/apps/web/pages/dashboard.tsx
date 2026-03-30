// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useMemo } from 'react';
import { usePrivy } from '@privy-io/react-auth';
import {
  TrendingUp,
  DollarSign,
  Music,
  Users,
  Download,
  Edit,
  Trash2,
  Settings,
  ArrowUpRight,
  ArrowDownRight,
  Loader2
} from 'lucide-react';
import { type Song } from '../data/mockSongs';
import { PaymentModel } from '@/lib';
import { useArtistStats, useArtistSongs, useArtistAnalytics } from '../src/hooks/useArtistData';
import Navigation from '../components/Navigation';

export default function DashboardPage() {
  const { authenticated, user } = usePrivy();
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');

  // Get artist's wallet address from Privy
  const artistAddress = user?.wallet?.address;

  // Days mapping for API
  const daysMap = { '7d': 7, '30d': 30, '90d': 90, 'all': 365 };

  // Fetch artist data from API
  const { stats, isLoading: statsLoading } = useArtistStats(artistAddress);
  const { songs: artistSongs, isLoading: songsLoading } = useArtistSongs(artistAddress);
  const { analytics, isLoading: analyticsLoading } = useArtistAnalytics(artistAddress, daysMap[timeRange]);

  const isLoading = statsLoading || songsLoading || analyticsLoading;

  // Calculate stats from API data
  const totalEarnings = stats.totalEarnings;
  const totalPlays = stats.totalPlays;
  const avgPerPlay = totalPlays > 0 ? totalEarnings / totalPlays : 0;

  // Transform analytics timeseries to chart format
  const earningsData = useMemo(() => {
    if (!analytics.timeseries.length) {
      return [
        { day: 'Day 1', amount: 0 },
        { day: 'Day 2', amount: 0 },
      ];
    }
    return analytics.timeseries.map(t => ({
      day: new Date(t.day).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      amount: t.earnings,
    }));
  }, [analytics.timeseries]);

  const thisMonthEarnings = earningsData.length > 0 ? earningsData[earningsData.length - 1].amount : 0;
  const lastMonthEarnings = earningsData.length > 1 ? earningsData[0].amount : thisMonthEarnings;
  const earningsGrowth = lastMonthEarnings > 0 ? ((thisMonthEarnings - lastMonthEarnings) / lastMonthEarnings) * 100 : 0;

  // Economic model breakdown
  const modelStats = useMemo(() => [
    {
      model: 'Pay Per Stream',
      songs: artistSongs.filter((s: Song) => s.paymentModel === PaymentModel.PAY_PER_STREAM).length,
      earnings: artistSongs.filter((s: Song) => s.paymentModel === PaymentModel.PAY_PER_STREAM)
        .reduce((sum: number, s: Song) => sum + parseFloat(String(s.earnings).replace(',', '')), 0),
      color: 'blue'
    },
    {
      model: 'Freemium',
      songs: artistSongs.filter((s: Song) => s.paymentModel === PaymentModel.FREEMIUM).length,
      earnings: artistSongs.filter((s: Song) => s.paymentModel === PaymentModel.FREEMIUM)
        .reduce((sum: number, s: Song) => sum + parseFloat(String(s.earnings).replace(',', '')), 0),
      color: 'orange'
    },
    {
      model: 'Pay What You Want',
      songs: artistSongs.filter((s: Song) => s.paymentModel === PaymentModel.PAY_WHAT_YOU_WANT).length,
      earnings: artistSongs.filter((s: Song) => s.paymentModel === PaymentModel.PAY_WHAT_YOU_WANT)
        .reduce((sum: number, s: Song) => sum + parseFloat(String(s.earnings).replace(',', '')), 0),
      color: 'green'
    },
    {
      model: 'Patronage',
      songs: artistSongs.filter((s: Song) => s.paymentModel === PaymentModel.PATRONAGE).length,
      earnings: artistSongs.filter((s: Song) => s.paymentModel === PaymentModel.PATRONAGE)
        .reduce((sum: number, s: Song) => sum + parseFloat(String(s.earnings).replace(',', '')), 0),
      color: 'purple'
    },
  ].filter(stat => stat.songs > 0), [artistSongs]);

  if (!authenticated) {
    return (
      <>
        <Navigation />
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center p-4 pt-20">
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 max-w-md text-center">
            <h1 className="text-2xl font-bold text-white mb-4">Artist Dashboard</h1>
            <p className="text-gray-400 mb-6">Please connect your wallet to access your dashboard</p>
            <button className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition">
              Connect Wallet
            </button>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Navigation />
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 pt-20">
        <div className="max-w-7xl mx-auto px-4 py-12">

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">Artist Dashboard</h1>
              <p className="text-gray-400">Track your earnings, analyze performance, and manage your music</p>
            </div>
            {isLoading && (
              <div className="flex items-center space-x-2 text-purple-400">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span className="text-sm">Loading data...</span>
              </div>
            )}
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">

          {/* Total Earnings */}
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/20 rounded-lg">
                <DollarSign className="w-6 h-6 text-green-400" />
              </div>
              {earningsGrowth > 0 ? (
                <div className="flex items-center text-green-400 text-sm">
                  <ArrowUpRight className="w-4 h-4" />
                  <span>{earningsGrowth.toFixed(1)}%</span>
                </div>
              ) : (
                <div className="flex items-center text-red-400 text-sm">
                  <ArrowDownRight className="w-4 h-4" />
                  <span>{Math.abs(earningsGrowth).toFixed(1)}%</span>
                </div>
              )}
            </div>
            <p className="text-gray-400 text-sm mb-1">Total Earnings</p>
            <p className="text-3xl font-bold text-white">${totalEarnings.toFixed(2)}</p>
            <p className="text-gray-500 text-xs mt-2">This month: ${thisMonthEarnings.toFixed(2)}</p>
          </div>

          {/* Total Plays */}
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-500/20 rounded-lg">
                <Music className="w-6 h-6 text-blue-400" />
              </div>
              <div className="flex items-center text-blue-400 text-sm">
                <ArrowUpRight className="w-4 h-4" />
                <span>12%</span>
              </div>
            </div>
            <p className="text-gray-400 text-sm mb-1">Total Plays</p>
            <p className="text-3xl font-bold text-white">{totalPlays.toLocaleString()}</p>
            <p className="text-gray-500 text-xs mt-2">Last 30 days</p>
          </div>

          {/* Avg Per Play */}
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-500/20 rounded-lg">
                <TrendingUp className="w-6 h-6 text-purple-400" />
              </div>
            </div>
            <p className="text-gray-400 text-sm mb-1">Avg Per Play</p>
            <p className="text-3xl font-bold text-white">${avgPerPlay.toFixed(4)}</p>
            <p className="text-gray-500 text-xs mt-2">3.3x better than Spotify</p>
          </div>

          {/* Active Listeners */}
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-orange-500/20 rounded-lg">
                <Users className="w-6 h-6 text-orange-400" />
              </div>
              <div className="flex items-center text-orange-400 text-sm">
                <ArrowUpRight className="w-4 h-4" />
                <span>8%</span>
              </div>
            </div>
            <p className="text-gray-400 text-sm mb-1">Active Listeners</p>
            <p className="text-3xl font-bold text-white">1,234</p>
            <p className="text-gray-500 text-xs mt-2">Last 30 days</p>
          </div>
        </div>

        {/* Earnings Chart */}
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white">Earnings Over Time</h2>
            <div className="flex space-x-2">
              {(['7d', '30d', '90d', 'all'] as const).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition ${
                    timeRange === range
                      ? 'bg-purple-600 text-white'
                      : 'bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  {range === 'all' ? 'All Time' : range.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Simple bar chart */}
          <div className="relative h-64">
            <div className="absolute inset-0 flex items-end justify-around space-x-2">
              {earningsData.map((data, i) => {
                const maxAmount = Math.max(...earningsData.map(d => d.amount));
                const height = (data.amount / maxAmount) * 100;
                return (
                  <div key={i} className="flex-1 flex flex-col items-center">
                    <div className="relative w-full group">
                      <div
                        className="w-full bg-gradient-to-t from-purple-600 to-purple-400 rounded-t-lg transition-all hover:from-purple-500 hover:to-purple-300"
                        style={{ height: `${height * 2}px` }}
                      />
                      {/* Tooltip */}
                      <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition whitespace-nowrap">
                        <div className="bg-black/90 text-white text-xs px-2 py-1 rounded">
                          ${data.amount.toFixed(2)}
                        </div>
                      </div>
                    </div>
                    <p className="text-gray-500 text-xs mt-2">{data.day.split(' ')[1]}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-8">

          {/* Top Performing Songs */}
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
            <h2 className="text-xl font-bold text-white mb-4">Top Performing Songs</h2>
            <div className="space-y-4">
              {[...artistSongs].sort((a: Song, b: Song) => b.plays - a.plays).slice(0, 3).map((song: Song, i: number) => (
                <div key={song.id} className="flex items-center space-x-4">
                  <div className="text-2xl font-bold text-gray-500 w-8">{i + 1}</div>
                  <img
                    src={song.coverArt}
                    alt={song.title}
                    className="w-12 h-12 rounded object-cover"
                  />
                  <div className="flex-1">
                    <p className="text-white font-medium">{song.title}</p>
                    <p className="text-gray-400 text-sm">{song.plays.toLocaleString()} plays</p>
                  </div>
                  <p className="text-green-400 font-bold">${song.earnings}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Economic Model Performance */}
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
            <h2 className="text-xl font-bold text-white mb-4">Model Performance</h2>
            <div className="space-y-4">
              {modelStats.map((stat) => (
                <div key={stat.model}>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-white font-medium">{stat.model}</p>
                    <p className="text-gray-400 text-sm">{stat.songs} songs</p>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className={`h-full bg-${stat.color}-500 rounded-full`}
                        style={{ width: `${(stat.earnings / totalEarnings) * 100}%` }}
                      />
                    </div>
                    <p className="text-white font-bold min-w-[80px] text-right">
                      ${stat.earnings.toFixed(2)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <button className="bg-purple-600 hover:bg-purple-700 text-white rounded-xl p-6 transition flex items-center justify-center space-x-3">
            <Music className="w-6 h-6" />
            <span className="font-medium">Upload New Song</span>
          </button>

          <button className="bg-green-600 hover:bg-green-700 text-white rounded-xl p-6 transition flex items-center justify-center space-x-3">
            <Download className="w-6 h-6" />
            <span className="font-medium">Request Cashout</span>
          </button>

          <button className="bg-white/5 hover:bg-white/10 border border-white/10 text-white rounded-xl p-6 transition flex items-center justify-center space-x-3">
            <Settings className="w-6 h-6" />
            <span className="font-medium">Account Settings</span>
          </button>
        </div>

        {/* Song Management */}
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
          <h2 className="text-xl font-bold text-white mb-4">Manage Your Music</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left text-gray-400 font-medium py-3 px-2">Song</th>
                  <th className="text-left text-gray-400 font-medium py-3 px-2">Model</th>
                  <th className="text-left text-gray-400 font-medium py-3 px-2">Plays</th>
                  <th className="text-left text-gray-400 font-medium py-3 px-2">Earnings</th>
                  <th className="text-right text-gray-400 font-medium py-3 px-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {artistSongs.map((song: Song) => (
                  <tr key={song.id} className="border-b border-white/5 hover:bg-white/5 transition">
                    <td className="py-3 px-2">
                      <div className="flex items-center space-x-3">
                        <img
                          src={song.coverArt}
                          alt={song.title}
                          className="w-10 h-10 rounded object-cover"
                        />
                        <div>
                          <p className="text-white font-medium">{song.title}</p>
                          <p className="text-gray-400 text-sm">{song.genre}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-2">
                      <span className="inline-block px-2 py-1 bg-purple-500/20 border border-purple-500/30 rounded text-purple-400 text-xs">
                        {song.paymentModel === PaymentModel.PAY_PER_STREAM && 'Pay Per Stream'}
                        {song.paymentModel === PaymentModel.FREEMIUM && 'Freemium'}
                        {song.paymentModel === PaymentModel.PAY_WHAT_YOU_WANT && 'Pay What You Want'}
                        {song.paymentModel === PaymentModel.PATRONAGE && 'Patronage'}
                      </span>
                    </td>
                    <td className="py-3 px-2 text-white">{song.plays.toLocaleString()}</td>
                    <td className="py-3 px-2 text-green-400 font-bold">${song.earnings}</td>
                    <td className="py-3 px-2">
                      <div className="flex items-center justify-end space-x-2">
                        <button className="p-2 hover:bg-white/10 rounded transition">
                          <Edit className="w-4 h-4 text-gray-400" />
                        </button>
                        <button className="p-2 hover:bg-white/10 rounded transition">
                          <Trash2 className="w-4 h-4 text-red-400" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        </div>
      </div>
    </>
  );
}

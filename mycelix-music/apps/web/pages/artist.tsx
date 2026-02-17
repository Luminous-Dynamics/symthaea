import { motion } from 'framer-motion';
import { TrendingUp, DollarSign, Shield, Zap, Users, Clock } from 'lucide-react';
import Link from 'next/link';
import EarningsCalculator from '../components/EarningsCalculator';
import Navigation from '../components/Navigation';

export default function ArtistPage() {
  const comparisonData = [
    {
      platform: 'Spotify',
      perStream: '$0.003',
      for10kPlays: '$30',
      for100kPlays: '$300',
      waitTime: '90 days',
      color: 'text-gray-400',
    },
    {
      platform: 'Apple Music',
      perStream: '$0.007',
      for10kPlays: '$70',
      for100kPlays: '$700',
      waitTime: '90 days',
      color: 'text-gray-400',
    },
    {
      platform: 'Mycelix (Pay-per-stream)',
      perStream: '$0.01',
      for10kPlays: '$100',
      for100kPlays: '$1,000',
      waitTime: 'Instant',
      color: 'text-green-400',
      highlight: true,
    },
    {
      platform: 'Mycelix (Pay-what-you-want)',
      perStream: '$0.15 avg',
      for10kPlays: '$1,500',
      for100kPlays: '$15,000',
      waitTime: 'Instant',
      color: 'text-green-400',
      highlight: true,
    },
  ];

  const artistBenefits = [
    {
      icon: DollarSign,
      title: 'Instant Payments',
      description: 'Get paid immediately for every stream. No 90-day delays, no minimum payment thresholds.',
      stat: '0 seconds',
    },
    {
      icon: Shield,
      title: '100% Rights Ownership',
      description: 'You keep all rights to your music. No exploitative contracts, no giving up ownership.',
      stat: '100%',
    },
    {
      icon: TrendingUp,
      title: '10-50x More Earnings',
      description: 'Earn what your music is worth. Set your own price, choose your model.',
      stat: '10-50x',
    },
    {
      icon: Users,
      title: 'Direct Fan Relationship',
      description: 'Build direct connections with your fans. No platform intermediary.',
      stat: '∞',
    },
    {
      icon: Clock,
      title: 'No Minimums',
      description: 'Get paid from your first stream. No $100 minimum thresholds.',
      stat: '$0',
    },
    {
      icon: Zap,
      title: 'Full Control',
      description: 'You choose the economic model. Switch anytime. Your music, your rules.',
      stat: '4 models',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <Navigation />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 pt-24">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-20"
        >
          <h1 className="text-6xl font-bold text-white mb-6">
            Artists,
            <br />
            <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
              Get Paid What You Deserve
            </span>
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Stop getting pennies per stream. Mycelix Music puts you in control of your earnings,
            your rights, and your relationship with fans.
          </p>
          <Link
            href="/upload"
            className="inline-block px-10 py-4 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white rounded-lg text-lg font-medium transition shadow-lg shadow-green-500/30"
          >
            Upload Your Music →
          </Link>
        </motion.div>

        {/* Earnings Calculator */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <EarningsCalculator />
        </motion.div>

        {/* Platform Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-4xl font-bold text-white text-center mb-4">
            Compare Earnings Across Platforms
          </h2>
          <p className="text-xl text-gray-300 text-center mb-12 max-w-2xl mx-auto">
            See how much more you can earn on Mycelix vs traditional streaming
          </p>

          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left p-6 text-gray-400 font-medium">Platform</th>
                    <th className="text-right p-6 text-gray-400 font-medium">Per Stream</th>
                    <th className="text-right p-6 text-gray-400 font-medium">10K Plays</th>
                    <th className="text-right p-6 text-gray-400 font-medium">100K Plays</th>
                    <th className="text-right p-6 text-gray-400 font-medium">Payment Time</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonData.map((row, index) => (
                    <tr
                      key={index}
                      className={`border-b border-white/5 ${row.highlight ? 'bg-green-500/10' : ''}`}
                    >
                      <td className={`p-6 font-semibold ${row.color}`}>{row.platform}</td>
                      <td className={`p-6 text-right ${row.color}`}>{row.perStream}</td>
                      <td className={`p-6 text-right ${row.color}`}>{row.for10kPlays}</td>
                      <td className={`p-6 text-right ${row.color}`}>{row.for100kPlays}</td>
                      <td className={`p-6 text-right ${row.color}`}>{row.waitTime}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-8 p-6 bg-green-500/10 border border-green-500/20 rounded-xl">
            <p className="text-green-400 text-center text-lg">
              💰 <strong>With Mycelix, you could earn $1,500 from 10K streams</strong> vs just $30 on Spotify.
              That's 50x more!
            </p>
          </div>
        </motion.div>

        {/* Benefits Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-4xl font-bold text-white text-center mb-4">
            Why Artists Choose Mycelix
          </h2>
          <p className="text-xl text-gray-300 text-center mb-12 max-w-2xl mx-auto">
            Everything you need to build a sustainable music career
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {artistBenefits.map((benefit, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
                className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 hover:border-green-400/50 transition"
              >
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <benefit.icon className="w-10 h-10 text-green-400" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-white mb-2">{benefit.title}</h3>
                    <p className="text-gray-300 mb-3">{benefit.description}</p>
                    <div className="text-3xl font-bold text-green-400">{benefit.stat}</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Economic Models */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="text-4xl font-bold text-white text-center mb-4">
            Choose Your Economic Model
          </h2>
          <p className="text-xl text-gray-300 text-center mb-12 max-w-2xl mx-auto">
            Four proven models to monetize your music. Switch anytime.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                name: 'Pay Per Stream',
                price: '$0.01 per play',
                description: 'Simple and straightforward. Every play earns you a penny.',
                bestFor: 'Popular tracks with high play counts',
                color: 'from-blue-500 to-cyan-500',
              },
              {
                name: 'Freemium',
                price: '3 free plays, then $0.01',
                description: 'Let fans try before they buy. Perfect for discovery.',
                bestFor: 'New artists building an audience',
                color: 'from-orange-500 to-red-500',
              },
              {
                name: 'Pay What You Want',
                price: 'Free + tips ($0.15 avg)',
                description: 'True gift economy. Fans tip what they can afford.',
                bestFor: 'Artists with engaged, generous communities',
                color: 'from-green-500 to-emerald-500',
              },
              {
                name: 'Patronage',
                price: '$5-20/month subscription',
                description: 'Monthly support from dedicated fans. Steady income.',
                bestFor: 'Established artists with loyal fanbases',
                color: 'from-purple-500 to-pink-500',
              },
            ].map((model, index) => (
              <div
                key={index}
                className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:border-white/20 transition"
              >
                <div className={`inline-block px-4 py-2 bg-gradient-to-r ${model.color} rounded-full text-white text-sm font-medium mb-4`}>
                  {model.name}
                </div>
                <div className="text-3xl font-bold text-white mb-3">{model.price}</div>
                <p className="text-gray-300 mb-4 text-lg">{model.description}</p>
                <div className="text-sm text-gray-400">
                  <strong>Best for:</strong> {model.bestFor}
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="text-center bg-gradient-to-br from-green-600/20 to-emerald-600/20 border border-green-500/30 rounded-2xl p-12"
        >
          <h2 className="text-4xl font-bold text-white mb-4">Ready to Earn What You Deserve?</h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Join the revolution. Upload your music and start earning instantly.
          </p>
          <Link
            href="/upload"
            className="inline-block px-12 py-5 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white rounded-lg text-xl font-medium transition shadow-lg shadow-green-500/30"
          >
            Upload Your First Track →
          </Link>
        </motion.div>
      </div>
    </div>
  );
}

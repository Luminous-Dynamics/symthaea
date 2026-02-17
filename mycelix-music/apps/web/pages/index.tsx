import { useState } from 'react';
import { usePrivy } from '@privy-io/react-auth';
import { motion } from 'framer-motion';
import { Music, Sparkles, TrendingUp, Users, Play, Heart, DollarSign, Shield, Zap, Globe } from 'lucide-react';
import Link from 'next/link';
import Navigation from '../components/Navigation';

export default function HomePage() {
  const { login, authenticated, user } = usePrivy();
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  const listenerFeatures = [
    {
      icon: Sparkles,
      title: 'Try Before You Buy',
      description: 'Free plays on most tracks. No subscription required. Pay only for what you love.',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: Heart,
      title: 'Support Artists Directly',
      description: 'Your payment goes straight to the artist. No middlemen taking 70% cuts.',
      color: 'from-pink-500 to-rose-500',
    },
    {
      icon: Zap,
      title: 'Instant Access',
      description: 'Stream immediately. No ads, no waiting, no algorithmic manipulation.',
      color: 'from-purple-500 to-indigo-500',
    },
  ];

  const artistFeatures = [
    {
      icon: TrendingUp,
      title: '10x-50x More Earnings',
      description: 'Earn $100-$1,500 from 10K plays vs Spotify\'s $30. You set your price.',
      color: 'from-green-500 to-emerald-500',
    },
    {
      icon: DollarSign,
      title: 'Instant Payments',
      description: 'Get paid immediately for every stream. No 90-day delays, no minimum thresholds.',
      color: 'from-yellow-500 to-orange-500',
    },
    {
      icon: Shield,
      title: '100% Ownership',
      description: 'You keep all rights to your music. No exploitative contracts, no giving up control.',
      color: 'from-purple-500 to-pink-500',
    },
  ];

  const howItWorksSteps = [
    {
      number: '1',
      title: 'Choose Your Model',
      description: 'Artists pick from 4 economic models: Pay-per-stream ($0.01), Freemium (3 free plays), Tips (pay what you want), or Patronage (monthly subscription).',
    },
    {
      number: '2',
      title: 'Upload & Share',
      description: 'Upload your music to IPFS (decentralized storage) and set your price. Share your link with fans worldwide.',
    },
    {
      number: '3',
      title: 'Earn Instantly',
      description: 'Every stream or tip goes directly to your wallet. No delays, no minimums, no platform taking 30-70% cuts.',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <Navigation />

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 pt-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center"
        >
          <h1 className="text-6xl font-bold text-white mb-6">
            Music Streaming,
            <br />
            <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
              Reimagined
            </span>
          </h1>
          <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto">
            The first decentralized music platform where every artist is sovereign, every listener is
            valued, and every community designs its own economy.
          </p>
          <div className="flex justify-center space-x-4">
            <Link
              href="/discover"
              className="px-8 py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-lg font-medium transition"
            >
              Discover Music
            </Link>
            <Link
              href="/upload"
              className="px-8 py-4 border-2 border-purple-400 text-purple-400 hover:bg-purple-400/10 rounded-lg text-lg font-medium transition"
            >
              Upload Your Song
            </Link>
          </div>
        </motion.div>

        {/* For Listeners Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-32"
        >
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-white mb-4">For Listeners 🎧</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Discover amazing music and support artists directly—no subscriptions, no ads
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {listenerFeatures.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
                className="relative group"
              >
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${feature.color} rounded-xl opacity-0 group-hover:opacity-20 transition-opacity duration-300`}
                />
                <div className="relative bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 h-full hover:border-white/20 transition">
                  <feature.icon className="w-16 h-16 text-blue-400 mb-6" />
                  <h3 className="text-2xl font-semibold text-white mb-3">{feature.title}</h3>
                  <p className="text-gray-300 text-lg">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* For Artists Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-32"
        >
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-white mb-4">For Artists 🎵</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Keep what you earn. Own your music. Build direct relationships with fans.
            </p>
            <Link
              href="/artist"
              className="inline-block mt-6 px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white rounded-lg font-medium transition"
            >
              See Artist Earnings Calculator →
            </Link>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {artistFeatures.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
                className="relative group"
              >
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${feature.color} rounded-xl opacity-0 group-hover:opacity-20 transition-opacity duration-300`}
                />
                <div className="relative bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 h-full hover:border-white/20 transition">
                  <feature.icon className="w-16 h-16 text-green-400 mb-6" />
                  <h3 className="text-2xl font-semibold text-white mb-3">{feature.title}</h3>
                  <p className="text-gray-300 text-lg">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* How It Works Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="mt-32"
        >
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">How It Works ⚡</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Three simple steps to revolutionize music streaming
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {howItWorksSteps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                className="text-center"
              >
                <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 text-white text-3xl font-bold mb-6">
                  {step.number}
                </div>
                <h3 className="text-2xl font-semibold text-white mb-4">{step.title}</h3>
                <p className="text-gray-300 text-lg leading-relaxed">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.9 }}
          className="mt-32 text-center bg-gradient-to-br from-purple-600/20 to-pink-600/20 border border-purple-500/30 rounded-2xl p-12"
        >
          <h2 className="text-4xl font-bold text-white mb-4">Ready to Join the Revolution?</h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Whether you're an artist or a music lover, Mycelix Music gives you the power
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              href="/discover"
              className="px-10 py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-lg font-medium transition"
            >
              Start Listening
            </Link>
            <Link
              href="/upload"
              className="px-10 py-4 bg-green-600 hover:bg-green-700 text-white rounded-lg text-lg font-medium transition"
            >
              Upload Your Music
            </Link>
          </div>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.0 }}
          className="mt-20 grid md:grid-cols-3 gap-8 text-center"
        >
          <div>
            <div className="text-4xl font-bold text-white mb-2">4</div>
            <div className="text-gray-400">Economic Models</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-white mb-2">10-50x</div>
            <div className="text-gray-400">Better Artist Earnings</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-white mb-2">100%</div>
            <div className="text-gray-400">Artist Sovereignty</div>
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <footer className="border-t border-white/10 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-500">
            <p>© 2025 Mycelix Music. Built on Mycelix Protocol.</p>
            <p className="mt-2 text-sm">
              Powered by FLOW 💧 | CGC ✨ | TEND 🤲 | CIV 🏛️
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

'use client';

import { useState } from 'react';
import { Mail, CheckCircle, AlertCircle, Loader2, TrendingUp, Users, Globe } from 'lucide-react';
import { useAnalytics } from '@/lib/analytics';

interface EmailCaptureProps {
  source?: string;
  variant?: 'inline' | 'modal' | 'hero';
  projectId?: string;
}

export default function EmailCapture({ source = 'general', variant = 'inline', projectId }: EmailCaptureProps) {
  const analytics = useAnalytics();
  const [email, setEmail] = useState('');
  const [interestedAmount, setInterestedAmount] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [showDetails, setShowDetails] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('loading');

    try {
      const response = await fetch('/api/waitlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          interested_amount: interestedAmount ? parseFloat(interestedAmount) : null,
          source,
          project_id: projectId,
          utm_source: new URLSearchParams(window.location.search).get('utm_source'),
          utm_medium: new URLSearchParams(window.location.search).get('utm_medium'),
          utm_campaign: new URLSearchParams(window.location.search).get('utm_campaign'),
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setStatus('success');
        setMessage(data.message || 'Welcome to the Terra Atlas community!');
        setEmail('');
        setInterestedAmount('');
        
        // Track waitlist signup
        if (analytics) {
          const investmentInterest = interestedAmount ? parseFloat(interestedAmount) : undefined;
          analytics.trackWaitlistSignup(email, investmentInterest, source);
          
          // Track additional context
          analytics.trackEvent('waitlist_signup_complete', {
            source,
            project_id: projectId,
            has_investment_interest: !!interestedAmount,
            variant
          });
        }
      } else {
        setStatus('error');
        setMessage(data.error || 'Something went wrong. Please try again.');
        
        if (analytics) {
          analytics.trackEvent('waitlist_signup_error', {
            error: data.error,
            source
          });
        }
      }
    } catch (error) {
      setStatus('error');
      setMessage('Network error. Please check your connection and try again.');
    }
  };

  if (variant === 'hero') {
    return (
      <div className="bg-gradient-to-r from-green-600/20 to-blue-600/20 backdrop-blur-xl rounded-2xl p-4 sm:p-6 lg:p-8 border border-green-500/30">
        <div className="text-center mb-6">
          <h3 className="text-2xl sm:text-3xl font-bold text-white mb-2">
            Be First to Invest in the Clean Energy Revolution
          </h3>
          <p className="text-sm sm:text-base text-gray-300">
            Join 12,847 investors ready to deploy $2.3B into clean energy projects
          </p>
        </div>

        {status === 'success' ? (
          <div className="bg-green-500/20 border border-green-500/50 rounded-xl p-4 sm:p-6 text-center">
            <CheckCircle className="w-10 sm:w-12 h-10 sm:h-12 text-green-400 mx-auto mb-3" />
            <h4 className="text-lg sm:text-xl font-semibold text-green-100 mb-2">You're In!</h4>
            <p className="text-sm sm:text-base text-green-200">{message}</p>
            <div className="mt-6 pt-6 border-t border-green-500/30">
              <p className="text-gray-300 mb-4">What happens next:</p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-left">
                <div className="flex items-start gap-3">
                  <div className="bg-green-500/20 p-2 rounded-lg flex-shrink-0">
                    <Mail className="w-4 h-4 text-green-400" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-white">Weekly Updates</p>
                    <p className="text-xs text-gray-400">Project launches & insights</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-blue-500/20 p-2 rounded-lg flex-shrink-0">
                    <TrendingUp className="w-4 h-4 text-blue-400" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-white">Early Access</p>
                    <p className="text-xs text-gray-400">First to see new projects</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="bg-purple-500/20 p-2 rounded-lg flex-shrink-0">
                    <Users className="w-4 h-4 text-purple-400" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-white">Exclusive Events</p>
                    <p className="text-xs text-gray-400">Investor webinars & tours</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                required
                className="w-full px-6 py-4 bg-gray-800/50 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:border-green-500 text-lg"
              />
            </div>

            {showDetails && (
              <div className="animate-fadeIn">
                <label className="block text-gray-400 text-sm mb-2">
                  How much are you interested in investing? (Optional)
                </label>
                <select
                  value={interestedAmount}
                  onChange={(e) => setInterestedAmount(e.target.value)}
                  className="w-full px-6 py-4 bg-gray-800/50 border border-gray-700 rounded-xl text-white focus:outline-none focus:border-green-500"
                >
                  <option value="">Select amount...</option>
                  <option value="1000">$1,000 - $10,000</option>
                  <option value="10000">$10,000 - $50,000</option>
                  <option value="50000">$50,000 - $100,000</option>
                  <option value="100000">$100,000 - $500,000</option>
                  <option value="500000">$500,000+</option>
                </select>
              </div>
            )}

            {!showDetails && (
              <button
                type="button"
                onClick={() => setShowDetails(true)}
                className="text-sm text-gray-400 hover:text-gray-300"
              >
                + Add investment interest (optional)
              </button>
            )}

            {status === 'error' && (
              <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-3 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-red-400" />
                <p className="text-red-200 text-sm">{message}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={status === 'loading'}
              className="w-full px-8 py-4 bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-xl font-semibold text-lg hover:from-green-600 hover:to-blue-600 transition disabled:opacity-50 flex items-center justify-center gap-3"
            >
              {status === 'loading' ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Joining waitlist...
                </>
              ) : (
                <>
                  Join 12,847 Investors
                  <Globe className="w-5 h-5" />
                </>
              )}
            </button>

            <p className="text-center text-xs text-gray-500">
              No spam, ever. Unsubscribe anytime. By joining, you agree to our{' '}
              <a href="/privacy" className="text-blue-400 hover:text-blue-300">Privacy Policy</a>.
            </p>
          </form>
        )}
      </div>
    );
  }

  // Inline variant for embedding in pages
  return (
    <div className="bg-gray-800/50 backdrop-blur-xl rounded-xl p-6 border border-gray-700">
      <h3 className="text-lg font-semibold text-white mb-3">
        Get Early Access
      </h3>
      {status === 'success' ? (
        <div className="flex items-center gap-3 text-green-400">
          <CheckCircle className="w-5 h-5" />
          <p>{message}</p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-3">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="your@email.com"
            required
            className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
          {status === 'error' && (
            <p className="text-red-400 text-sm">{message}</p>
          )}
          <button
            type="submit"
            disabled={status === 'loading'}
            className="w-full px-4 py-2 bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-lg font-semibold hover:from-green-600 hover:to-blue-600 transition disabled:opacity-50"
          >
            {status === 'loading' ? 'Joining...' : 'Join Waitlist'}
          </button>
        </form>
      )}
    </div>
  );
}

// Modal variant for popups
export function EmailCaptureModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="relative max-w-md w-full">
        <button
          onClick={onClose}
          className="absolute -top-12 right-0 text-gray-400 hover:text-white"
        >
          ✕
        </button>
        <EmailCapture variant="hero" source="modal" />
      </div>
    </div>
  );
}
'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Mail, CheckCircle, AlertCircle, Loader2, ArrowRight, RefreshCw, Globe } from 'lucide-react'
import { supabase } from '@/lib/supabase'

export default function VerifyEmailPage() {
  const [email, setEmail] = useState<string | null>(null)
  const [isResending, setIsResending] = useState(false)
  const [resendSuccess, setResendSuccess] = useState(false)
  const [resendError, setResendError] = useState<string | null>(null)
  const [countdown, setCountdown] = useState(0)
  const router = useRouter()

  useEffect(() => {
    // Get email from session
    const getEmailFromSession = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (session?.user?.email) {
        setEmail(session.user.email)
        
        // Check if email is already verified
        if (session.user.email_confirmed_at) {
          router.push('/dashboard')
        }
      }
    }

    getEmailFromSession()

    // Check for email verification in URL (for users coming from email link)
    const { data: authListener } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'SIGNED_IN' && session?.user?.email_confirmed_at) {
        router.push('/dashboard')
      }
    })

    return () => {
      authListener.subscription.unsubscribe()
    }
  }, [router])

  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000)
      return () => clearTimeout(timer)
    }
  }, [countdown])

  const handleResendEmail = async () => {
    if (!email || countdown > 0) return

    setIsResending(true)
    setResendError(null)
    setResendSuccess(false)

    try {
      const { error } = await supabase.auth.resend({
        type: 'signup',
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      })

      if (error) {
        setResendError(error.message)
      } else {
        setResendSuccess(true)
        setCountdown(60) // 60 second cooldown
      }
    } catch (err) {
      setResendError('Failed to resend verification email')
    } finally {
      setIsResending(false)
    }
  }

  const handleRefreshStatus = async () => {
    const { data: { session } } = await supabase.auth.getSession()
    if (session?.user?.email_confirmed_at) {
      router.push('/dashboard')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white flex items-center justify-center p-6">
      {/* Background effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 right-1/3 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 left-1/3 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl"></div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10 w-full max-w-lg"
      >
        {/* Logo */}
        <Link href="/" className="flex items-center justify-center mb-8">
          <motion.div 
            whileHover={{ rotate: 360 }}
            transition={{ duration: 0.5 }}
            className="relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 blur-lg opacity-50"></div>
            <div className="relative bg-gradient-to-r from-blue-500 to-purple-500 w-16 h-16 rounded-xl flex items-center justify-center">
              <Globe className="w-10 h-10 text-white" />
            </div>
          </motion.div>
        </Link>

        {/* Main Container */}
        <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-8">
          {/* Icon */}
          <div className="flex justify-center mb-6">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, type: 'spring' }}
              className="relative"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 blur-2xl opacity-40"></div>
              <div className="relative w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                <Mail className="w-10 h-10 text-white" />
              </div>
            </motion.div>
          </div>

          {/* Content */}
          <h1 className="text-3xl font-bold text-center mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Verify Your Email
          </h1>
          
          <p className="text-gray-300 text-center mb-6">
            We&rsquo;ve sent a verification email to:
          </p>
          
          {email && (
            <div className="mb-8 p-3 bg-white/5 rounded-lg border border-white/10">
              <p className="text-center font-medium text-lg">{email}</p>
            </div>
          )}

          <p className="text-gray-400 text-center mb-8 text-sm">
            Please check your inbox and click the verification link to activate your account. 
            The link will expire in 24 hours.
          </p>

          {/* Resend Success Message */}
          {resendSuccess && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 bg-green-500/10 border border-green-500/30 rounded-lg flex items-start gap-3"
            >
              <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
              <div className="text-sm text-green-300">
                Verification email resent successfully! Please check your inbox.
              </div>
            </motion.div>
          )}

          {/* Resend Error Message */}
          {resendError && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-3"
            >
              <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
              <div className="text-sm text-red-300">{resendError}</div>
            </motion.div>
          )}

          {/* Actions */}
          <div className="space-y-4">
            {/* Resend Button */}
            <motion.button
              whileHover={{ scale: countdown > 0 ? 1 : 1.02 }}
              whileTap={{ scale: countdown > 0 ? 1 : 0.98 }}
              onClick={handleResendEmail}
              disabled={isResending || countdown > 0}
              className="w-full py-3 bg-white/10 border border-white/20 rounded-lg font-medium text-gray-300 hover:bg-white/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isResending ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Resending...
                </>
              ) : countdown > 0 ? (
                <>
                  <Mail className="w-5 h-5" />
                  Resend in {countdown}s
                </>
              ) : (
                <>
                  <Mail className="w-5 h-5" />
                  Resend Verification Email
                </>
              )}
            </motion.button>

            {/* Check Status Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleRefreshStatus}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-semibold text-white shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 transition-all flex items-center justify-center gap-2"
            >
              <RefreshCw className="w-5 h-5" />
              I&rsquo;ve Verified My Email
            </motion.button>
          </div>

          {/* Help Section */}
          <div className="mt-8 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <h3 className="text-sm font-semibold text-blue-300 mb-2">Didn&rsquo;t receive the email?</h3>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• Check your spam or junk folder</li>
              <li>• Make sure you entered the correct email address</li>
              <li>• Wait a few minutes for the email to arrive</li>
              <li>• Contact support if the problem persists</li>
            </ul>
          </div>

          {/* Footer Links */}
          <div className="mt-8 pt-6 border-t border-white/10 flex justify-between items-center text-sm">
            <Link href="/auth/login" className="text-gray-400 hover:text-white transition-colors">
              ← Back to Login
            </Link>
            <Link href="/support" className="text-blue-400 hover:text-blue-300 transition-colors">
              Need Help?
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
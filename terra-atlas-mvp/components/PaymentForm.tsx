'use client'

import { useState, useEffect } from 'react'
import { loadStripe, StripeElementsOptions } from '@stripe/stripe-js'
import {
  Elements,
  CardElement,
  useStripe,
  useElements
} from '@stripe/react-stripe-js'
import { Lock, CreditCard, AlertCircle, CheckCircle } from 'lucide-react'
import { useAuth } from '@/app/contexts/AuthContext'

// Load Stripe outside of component to avoid recreating on each render
const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY || 'YOUR_STRIPE_PUBLISHABLE_KEY')

interface PaymentFormProps {
  amount: number
  projectId: number
  projectName: string
  projectType?: string
  investmentTermMonths?: number
  onSuccess?: (paymentIntentId: string) => void
  onError?: (error: string) => void
  onCancel?: () => void
}

function PaymentFormInner({ 
  amount, 
  projectId, 
  projectName, 
  projectType = 'standard',
  investmentTermMonths = 12,
  onSuccess, 
  onError,
  onCancel 
}: PaymentFormProps) {
  const stripe = useStripe()
  const elements = useElements()
  const { session } = useAuth()
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [succeeded, setSucceeded] = useState(false)
  const [clientSecret, setClientSecret] = useState('')
  const authToken = session?.access_token

  useEffect(() => {
    // Create payment intent when parameters or auth change
    createPaymentIntent()
  }, [amount, projectId, authToken])

  const createPaymentIntent = async () => {
    if (!authToken) {
      setError('Please sign in to continue')
      onError?.('Authentication required')
      return
    }

    try {
      const response = await fetch('/api/stripe/create-payment-intent', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          amount,
          projectId,
          projectName,
          projectType,
          investmentTermMonths,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create payment intent')
      }

      setClientSecret(data.clientSecret)
    } catch (err: any) {
      setError(err.message)
      onError?.(err.message)
    }
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()

    if (!stripe || !elements || !clientSecret || !authToken) {
      if (!authToken) {
        setError('Please sign in to complete your investment.')
      }
      return
    }

    setProcessing(true)
    setError(null)

    const cardElement = elements.getElement(CardElement)
    
    if (!cardElement) {
      setError('Card details not found')
      setProcessing(false)
      return
    }

    // Confirm the payment with Stripe
    const { error: stripeError, paymentIntent } = await stripe.confirmCardPayment(
      clientSecret,
      {
        payment_method: {
          card: cardElement,
        },
      }
    )

    if (stripeError) {
      setError(stripeError.message || 'Payment failed')
      onError?.(stripeError.message || 'Payment failed')
      setProcessing(false)
      return
    }

    if (paymentIntent?.status === 'succeeded') {
      // Payment successful, now confirm with our backend
      try {
        const response = await fetch('/api/stripe/confirm-payment', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`
          },
          body: JSON.stringify({
            paymentIntentId: paymentIntent.id,
          }),
        })

        const data = await response.json()

        if (!response.ok) {
          throw new Error(data.error || 'Failed to confirm payment')
        }

        setSucceeded(true)
        onSuccess?.(paymentIntent.id)
      } catch (err: any) {
        // Payment succeeded but database update failed
        setError('Payment processed but investment record failed. Please contact support.')
        onError?.(err.message)
      }
    }

    setProcessing(false)
  }

  const formatAmount = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount)
  }

  // Card Element styling
  const cardElementOptions = {
    style: {
      base: {
        color: '#ffffff',
        fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        fontSize: '16px',
        '::placeholder': {
          color: '#6b7280',
        },
      },
      invalid: {
        color: '#ef4444',
        iconColor: '#ef4444',
      },
    },
  }

  if (succeeded) {
    return (
      <div className="text-center p-8">
        <CheckCircle className="h-16 w-16 text-emerald-400 mx-auto mb-4" />
        <h3 className="text-2xl font-bold text-white mb-2">Payment Successful!</h3>
        <p className="text-gray-400 mb-4">
          Your investment of {formatAmount(amount)} has been processed.
        </p>
        <p className="text-sm text-gray-500">
          You will receive a confirmation email shortly.
        </p>
      </div>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Amount Display */}
      <div className="bg-black/30 rounded-lg p-4 border border-gray-700">
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Investment Amount</span>
          <span className="text-2xl font-bold text-white">{formatAmount(amount)}</span>
        </div>
        <div className="flex justify-between items-center mt-2">
          <span className="text-sm text-gray-500">Project</span>
          <span className="text-sm text-gray-300">{projectName}</span>
        </div>
        <div className="flex justify-between items-center mt-1">
          <span className="text-sm text-gray-500">Term</span>
          <span className="text-sm text-gray-300">{investmentTermMonths} months</span>
        </div>
      </div>

      {/* Card Input */}
      <div>
        <label className="text-sm font-medium text-gray-300 mb-2 block">
          <CreditCard className="inline h-4 w-4 mr-1" />
          Card Details
        </label>
        <div className="bg-black/30 border border-gray-700 rounded-lg p-4 focus-within:border-emerald-500 transition">
          <CardElement options={cardElementOptions} />
        </div>
        <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
          <Lock className="h-3 w-3" />
          Your payment information is encrypted and secure
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-2">
          <AlertCircle className="h-5 w-5 text-red-400 mt-0.5" />
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            disabled={processing}
            className="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg font-medium text-gray-300 hover:bg-gray-700 transition disabled:opacity-50"
          >
            Cancel
          </button>
        )}
        <button
          type="submit"
          disabled={!stripe || processing || !clientSecret}
          className="flex-1 px-4 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-lg font-semibold text-white hover:shadow-lg hover:shadow-emerald-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {processing ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              Processing...
            </>
          ) : (
            <>
              <Lock className="h-4 w-4" />
              Complete Investment
            </>
          )}
        </button>
      </div>

      {/* Security Badges */}
      <div className="pt-4 border-t border-gray-800">
        <div className="flex items-center justify-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <Lock className="h-3 w-3" />
            Secure
          </span>
          <span className="flex items-center gap-1">
            <CreditCard className="h-3 w-3" />
            PCI Compliant
          </span>
          <span>Powered by Stripe</span>
        </div>
      </div>
    </form>
  )
}

export default function PaymentForm(props: PaymentFormProps) {
  const elementsOptions: StripeElementsOptions = {
    appearance: {
      theme: 'night',
      variables: {
        colorPrimary: '#10b981',
        colorBackground: '#000000',
        colorText: '#ffffff',
        colorDanger: '#ef4444',
        fontFamily: 'ui-sans-serif, system-ui, sans-serif',
      },
    },
  }

  return (
    <Elements stripe={stripePromise} options={elementsOptions}>
      <PaymentFormInner {...props} />
    </Elements>
  )
}

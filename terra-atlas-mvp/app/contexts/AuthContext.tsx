'use client'

import { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { User, Session } from '@supabase/supabase-js'
import { supabase, signIn, signUp, signOut, getUser, getSession } from '@/lib/supabase'
import { useRouter } from 'next/navigation'

interface Profile {
  id: string
  email: string
  username?: string | null
  full_name?: string | null
  avatar_url?: string | null
  role: 'user' | 'admin' | 'investor'
  kyc_status: 'pending' | 'verified' | 'rejected'
  total_invested: number
  total_returns: number
}

interface AuthContextType {
  user: User | null
  session: Session | null
  profile: Profile | null
  loading: boolean
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>
  signUp: (email: string, password: string, metadata?: any) => Promise<{ error: Error | null }>
  signOut: () => Promise<void>
  refreshProfile: () => Promise<void>
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  session: null,
  profile: null,
  loading: true,
  signIn: async () => ({ error: null }),
  signUp: async () => ({ error: null }),
  signOut: async () => {},
  refreshProfile: async () => {},
})

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [profile, setProfile] = useState<Profile | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  // Fetch user profile from database
  const fetchProfile = useCallback(async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .single()

      if (error) {
        console.error('Error fetching profile:', error)
        // If profile doesn't exist, create one
        if (error.code === 'PGRST116') {
          const { data: newProfile, error: createError } = await supabase
            .from('profiles')
            .insert([{
              id: userId,
              email: user?.email || '',
              role: 'user',
              kyc_status: 'pending',
              total_invested: 0,
              total_returns: 0,
            }])
            .select()
            .single()

          if (!createError && newProfile) {
            setProfile(newProfile as Profile)
          }
        }
      } else if (data) {
        setProfile(data as Profile)
      }
    } catch (error) {
      console.error('Error in fetchProfile:', error)
    }
  }, [user])

  // Initialize auth state
  useEffect(() => {
    const initAuth = async () => {
      try {
        // Get initial session
        const initialSession = await getSession()
        if (initialSession) {
          setSession(initialSession)
          setUser(initialSession.user)
          await fetchProfile(initialSession.user.id)
        }
      } catch (error) {
        console.error('Error initializing auth:', error)
      } finally {
        setLoading(false)
      }
    }

    initAuth()

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, currentSession) => {
        console.log('Auth state change:', event)
        
        if (currentSession) {
          setSession(currentSession)
          setUser(currentSession.user)
          await fetchProfile(currentSession.user.id)
        } else {
          setSession(null)
          setUser(null)
          setProfile(null)
        }

        // Handle auth events (removed automatic redirects)
        if (event === 'PASSWORD_RECOVERY') {
          router.push('/auth/reset-password')
        } else if (event === 'USER_UPDATED') {
          // Refresh profile when user is updated
          if (currentSession?.user) {
            await fetchProfile(currentSession.user.id)
          }
        }
        // Note: Removed automatic redirects on sign in/out to allow browsing without auth
      }
    )

    return () => {
      subscription.unsubscribe()
    }
  }, [fetchProfile, router])

  // Auth methods with error handling
  const handleSignIn = useCallback(async (email: string, password: string) => {
    try {
      setLoading(true)
      const { error } = await signIn(email, password)
      if (error) throw error
      return { error: null }
    } catch (error) {
      console.error('Sign in error:', error)
      return { error: error as Error }
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSignUp = useCallback(async (email: string, password: string, metadata?: any) => {
    try {
      setLoading(true)
      const { error } = await signUp(email, password, metadata)
      if (error) throw error
      return { error: null }
    } catch (error) {
      console.error('Sign up error:', error)
      return { error: error as Error }
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSignOut = useCallback(async () => {
    try {
      setLoading(true)
      const { error } = await signOut()
      if (error) throw error
      router.push('/')
    } catch (error) {
      console.error('Sign out error:', error)
    } finally {
      setLoading(false)
    }
  }, [router])

  const refreshProfile = useCallback(async () => {
    if (user) {
      await fetchProfile(user.id)
    }
  }, [user, fetchProfile])

  const value = {
    user,
    session,
    profile,
    loading,
    signIn: handleSignIn,
    signUp: handleSignUp,
    signOut: handleSignOut,
    refreshProfile,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// Protected route component
export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login')
    }
  }, [user, loading, router])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  return <>{children}</>
}
import { createClient } from '@supabase/supabase-js'
import { env } from './env'

// Environment variables for Supabase
const supabaseUrl = env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = env.NEXT_PUBLIC_SUPABASE_ANON_KEY

// Create a single supabase client for interacting with your database
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    flowType: 'pkce', // Proof Key for Code Exchange - more secure
    storage: typeof window !== 'undefined' ? window.localStorage : undefined,
  },
})

// Database types for TypeScript
export type Database = {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string
          email: string
          username: string | null
          full_name: string | null
          avatar_url: string | null
          created_at: string
          updated_at: string
          role: 'user' | 'admin' | 'investor'
          kyc_status: 'pending' | 'verified' | 'rejected'
          total_invested: number
          total_returns: number
        }
        Insert: {
          id: string
          email: string
          username?: string | null
          full_name?: string | null
          avatar_url?: string | null
          created_at?: string
          updated_at?: string
          role?: 'user' | 'admin' | 'investor'
          kyc_status?: 'pending' | 'verified' | 'rejected'
          total_invested?: number
          total_returns?: number
        }
        Update: {
          id?: string
          email?: string
          username?: string | null
          full_name?: string | null
          avatar_url?: string | null
          created_at?: string
          updated_at?: string
          role?: 'user' | 'admin' | 'investor'
          kyc_status?: 'pending' | 'verified' | 'rejected'
          total_invested?: number
          total_returns?: number
        }
      }
      investments: {
        Row: {
          id: string
          user_id: string
          project_id: string
          amount: number
          status: 'pending' | 'confirmed' | 'completed' | 'cancelled'
          transaction_hash: string | null
          created_at: string
          updated_at: string
          returns_paid: number
          estimated_returns: number
        }
        Insert: {
          id?: string
          user_id: string
          project_id: string
          amount: number
          status?: 'pending' | 'confirmed' | 'completed' | 'cancelled'
          transaction_hash?: string | null
          created_at?: string
          updated_at?: string
          returns_paid?: number
          estimated_returns?: number
        }
        Update: {
          id?: string
          user_id?: string
          project_id?: string
          amount?: number
          status?: 'pending' | 'confirmed' | 'completed' | 'cancelled'
          transaction_hash?: string | null
          created_at?: string
          updated_at?: string
          returns_paid?: number
          estimated_returns?: number
        }
      }
      projects: {
        Row: {
          id: string
          name: string
          type: 'solar' | 'wind' | 'hydro' | 'nuclear' | 'storage'
          location: string
          capacity_mw: number
          total_cost: number
          raised_amount: number
          status: 'planning' | 'funding' | 'construction' | 'operational'
          irr: number
          created_at: string
          updated_at: string
          latitude: number
          longitude: number
          description: string | null
        }
        Insert: {
          id?: string
          name: string
          type: 'solar' | 'wind' | 'hydro' | 'nuclear' | 'storage'
          location: string
          capacity_mw: number
          total_cost: number
          raised_amount?: number
          status?: 'planning' | 'funding' | 'construction' | 'operational'
          irr: number
          created_at?: string
          updated_at?: string
          latitude: number
          longitude: number
          description?: string | null
        }
        Update: {
          id?: string
          name?: string
          type?: 'solar' | 'wind' | 'hydro' | 'nuclear' | 'storage'
          location?: string
          capacity_mw?: number
          total_cost?: number
          raised_amount?: number
          status?: 'planning' | 'funding' | 'construction' | 'operational'
          irr?: number
          created_at?: string
          updated_at?: string
          latitude?: number
          longitude?: number
          description?: string | null
        }
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
  }
}

// Helper functions for auth
export const signUp = async (email: string, password: string, metadata?: any) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: metadata, // Additional user metadata
      emailRedirectTo: `${window.location.origin}/auth/callback`,
    },
  })
  return { data, error }
}

export const signIn = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  })
  return { data, error }
}

export const signOut = async () => {
  const { error } = await supabase.auth.signOut()
  return { error }
}

export const getUser = async () => {
  const { data: { user } } = await supabase.auth.getUser()
  return user
}

export const getSession = async () => {
  const { data: { session } } = await supabase.auth.getSession()
  return session
}

// Social auth
export const signInWithGoogle = async () => {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: `${window.location.origin}/auth/callback`,
    },
  })
  return { data, error }
}

export const signInWithGitHub = async () => {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'github',
    options: {
      redirectTo: `${window.location.origin}/auth/callback`,
    },
  })
  return { data, error }
}

// Password reset
export const resetPassword = async (email: string) => {
  const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${window.location.origin}/auth/reset-password`,
  })
  return { data, error }
}

export const updatePassword = async (newPassword: string) => {
  const { data, error } = await supabase.auth.updateUser({
    password: newPassword,
  })
  return { data, error }
}

import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      animation: {
        'gradient-flow': 'gradient 15s ease infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-slower': 'pulse 8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scroll-down': 'scroll-down 2s ease-in-out infinite',
        'fade-in-up': 'fade-in-up 0.6s ease-out',
        'shimmer': 'shimmer 3s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
        'tilt': 'tilt 10s ease-in-out infinite',
      },
      keyframes: {
        gradient: {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          }
        },
        'scroll-down': {
          '0%': {
            transform: 'translateY(0)',
            opacity: '1'
          },
          '100%': {
            transform: 'translateY(12px)',
            opacity: '0'
          }
        },
        'fade-in-up': {
          '0%': {
            opacity: '0',
            transform: 'translateY(20px)'
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)'
          }
        },
        'shimmer': {
          '0%': {
            backgroundPosition: '-200% center'
          },
          '100%': {
            backgroundPosition: '200% center'
          }
        },
        'float': {
          '0%, 100%': {
            transform: 'translateY(0px)'
          },
          '50%': {
            transform: 'translateY(-10px)'
          }
        },
        'glow-pulse': {
          '0%, 100%': {
            boxShadow: '0 0 20px rgba(16, 185, 129, 0.4), 0 0 40px rgba(16, 185, 129, 0.2)'
          },
          '50%': {
            boxShadow: '0 0 30px rgba(16, 185, 129, 0.6), 0 0 60px rgba(16, 185, 129, 0.3)'
          }
        },
        'tilt': {
          '0%, 100%': {
            transform: 'perspective(1000px) rotateX(0deg) rotateY(0deg)'
          },
          '25%': {
            transform: 'perspective(1000px) rotateX(1deg) rotateY(-1deg)'
          },
          '50%': {
            transform: 'perspective(1000px) rotateX(-1deg) rotateY(1deg)'
          },
          '75%': {
            transform: 'perspective(1000px) rotateX(1deg) rotateY(1deg)'
          }
        }
      },
      backgroundSize: {
        '200': '200% 200%',
      }
    },
  },
  plugins: [],
}

export default config
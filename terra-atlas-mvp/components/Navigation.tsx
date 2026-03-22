'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { 
  Globe, 
  Shield, 
  Brain, 
  Users, 
  TrendingUp, 
  Home,
  Calculator,
  Sparkles,
  ChevronRight
} from 'lucide-react';

export default function Navigation() {
  const pathname = usePathname();

  const navItems = [
    {
      href: '/',
      label: 'Home',
      icon: Home,
      description: 'Platform overview'
    },
    {
      href: '/trust',
      label: 'Trust Center',
      icon: Shield,
      description: 'Tax efficiency & transparency',
      highlight: true
    },
    {
      href: '/intelligence',
      label: 'AI Intelligence',
      icon: Brain,
      description: 'Smart investment scoring'
    },
    {
      href: '/community',
      label: 'Community',
      icon: Users,
      description: 'Ownership transition'
    },
    {
      href: '/portfolio',
      label: 'Portfolio',
      icon: TrendingUp,
      description: 'Your investments'
    }
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-gray-900/90 backdrop-blur-md border-b border-gray-800">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <Globe className="w-8 h-8 text-emerald-500" />
            <span className="text-xl font-bold text-white">Terra Atlas</span>
            <span className="hidden md:inline-block px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs font-semibold rounded-full ml-2">
              $72B Opportunities
            </span>
          </Link>

          {/* Navigation Items */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className="relative group"
                >
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={`
                      flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                      ${isActive 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                      }
                      ${item.highlight ? 'ring-1 ring-emerald-500/30' : ''}
                    `}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="font-medium">{item.label}</span>
                    {item.highlight && (
                      <Sparkles className="w-3 h-3 text-emerald-400 animate-pulse" />
                    )}
                  </motion.div>
                  
                  {/* Tooltip */}
                  <div className="absolute top-full mt-2 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
                    <div className="bg-gray-800 text-white text-xs px-3 py-1.5 rounded-lg whitespace-nowrap shadow-lg">
                      {item.description}
                      <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-gray-800 rotate-45" />
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>

          {/* CTA Button */}
          <div className="flex items-center gap-4">
            <Link
              href="/trust"
              className="hidden md:flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-lg font-semibold hover:from-emerald-600 hover:to-teal-600 transition-all shadow-lg shadow-emerald-500/20"
            >
              <Calculator className="w-4 h-4" />
              <span>See Tax Benefits</span>
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden border-t border-gray-800">
        <div className="flex items-center justify-around py-2">
          {navItems.slice(0, 4).map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`
                  flex flex-col items-center gap-1 px-3 py-1.5
                  ${isActive ? 'text-emerald-400' : 'text-gray-400'}
                `}
              >
                <Icon className="w-5 h-5" />
                <span className="text-xs">{item.label}</span>
                {item.highlight && (
                  <div className="absolute top-0 right-0 w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                )}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
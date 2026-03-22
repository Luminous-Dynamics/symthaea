'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Menu, X, Home, Globe, BookOpen, Calculator, Users, Mail, TrendingUp, Shield, ChevronRight } from 'lucide-react';
import { usePathname } from 'next/navigation';

export default function MobileNav() {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();

  // Close menu when route changes
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  // Prevent body scroll when menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  const navItems = [
    { href: '/', label: 'Home', icon: Home },
    { href: '/explore', label: 'Explore Projects', icon: Globe },
    { href: '/learn', label: 'Education', icon: BookOpen },
    { href: '/calculate', label: 'Impact Calculator', icon: Calculator },
    { href: '/community', label: 'Community', icon: Users },
    { href: '/about', label: 'About', icon: Shield },
  ];

  const quickActions = [
    { label: 'View Demo Projects', href: '/explore?demo=true', accent: 'green' },
    { label: 'Join Waitlist', href: '/#waitlist', accent: 'blue' },
    { label: 'Contact Team', href: '/contact', accent: 'purple' },
  ];

  return (
    <>
      {/* Mobile Header Bar */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-gray-900/95 backdrop-blur-lg border-b border-gray-800">
        <div className="flex items-center justify-between px-4 h-16">
          <Link href="/" className="flex items-center gap-2">
            <Globe className="w-7 h-7 text-green-400" />
            <span className="font-bold text-lg text-white">Terra Atlas</span>
          </Link>
          
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="p-2 rounded-lg hover:bg-gray-800 transition"
            aria-label="Toggle menu"
          >
            {isOpen ? (
              <X className="w-6 h-6 text-white" />
            ) : (
              <Menu className="w-6 h-6 text-white" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile Menu Overlay */}
      <div
        className={`lg:hidden fixed inset-0 z-40 transition-all duration-300 ${
          isOpen ? 'visible' : 'invisible'
        }`}
      >
        {/* Backdrop */}
        <div
          className={`absolute inset-0 bg-black transition-opacity duration-300 ${
            isOpen ? 'opacity-75' : 'opacity-0'
          }`}
          onClick={() => setIsOpen(false)}
        />

        {/* Menu Panel */}
        <div
          className={`absolute right-0 top-16 bottom-0 w-full sm:w-80 bg-gray-900 border-l border-gray-800 transition-transform duration-300 transform ${
            isOpen ? 'translate-x-0' : 'translate-x-full'
          }`}
        >
          <div className="h-full overflow-y-auto">
            {/* Navigation Links */}
            <nav className="p-4">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                Navigation
              </h3>
              <div className="space-y-1">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = pathname === item.href;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`flex items-center gap-3 px-4 py-3 rounded-lg transition ${
                        isActive
                          ? 'bg-gradient-to-r from-green-500/20 to-blue-500/20 text-white'
                          : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                      }`}
                    >
                      <Icon className="w-5 h-5 flex-shrink-0" />
                      <span className="flex-1">{item.label}</span>
                      {isActive && <ChevronRight className="w-4 h-4" />}
                    </Link>
                  );
                })}
              </div>
            </nav>

            {/* Quick Actions */}
            <div className="p-4 border-t border-gray-800">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                Quick Actions
              </h3>
              <div className="space-y-2">
                {quickActions.map((action) => (
                  <Link
                    key={action.label}
                    href={action.href}
                    className={`block px-4 py-3 rounded-lg text-center font-medium transition ${
                      action.accent === 'green'
                        ? 'bg-gradient-to-r from-green-600 to-green-500 text-white hover:from-green-500 hover:to-green-400'
                        : action.accent === 'blue'
                        ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white hover:from-blue-500 hover:to-blue-400'
                        : 'bg-gradient-to-r from-purple-600 to-purple-500 text-white hover:from-purple-500 hover:to-purple-400'
                    }`}
                  >
                    {action.label}
                  </Link>
                ))}
              </div>
            </div>

            {/* Stats */}
            <div className="p-4 border-t border-gray-800">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800/50 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-green-400">1,127</p>
                  <p className="text-xs text-gray-400">Projects</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-blue-400">230 GW</p>
                  <p className="text-xs text-gray-400">Capacity</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-purple-400">$72B</p>
                  <p className="text-xs text-gray-400">Opportunity</p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-yellow-400">11-14%</p>
                  <p className="text-xs text-gray-400">Target IRR</p>
                </div>
              </div>
            </div>

            {/* Demo Mode Notice */}
            <div className="p-4 border-t border-gray-800">
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <p className="text-xs text-yellow-400 font-medium">Demo Mode</p>
                <p className="text-xs text-gray-400 mt-1">
                  Exploring demonstration data. Real investments coming Q2 2025.
                </p>
              </div>
            </div>

            {/* Newsletter Signup */}
            <div className="p-4 border-t border-gray-800">
              <h3 className="text-sm font-semibold text-white mb-2">Get Updates</h3>
              <form className="flex gap-2">
                <input
                  type="email"
                  placeholder="Your email"
                  className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-green-500"
                />
                <button
                  type="submit"
                  className="px-4 py-2 bg-gradient-to-r from-green-600 to-blue-600 text-white text-sm font-medium rounded-lg hover:from-green-500 hover:to-blue-500 transition"
                >
                  Join
                </button>
              </form>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-gray-800 text-center">
              <p className="text-xs text-gray-500">
                © 2024 Terra Atlas. All rights reserved.
              </p>
              <div className="flex justify-center gap-4 mt-2">
                <Link href="/privacy" className="text-xs text-gray-400 hover:text-gray-300">
                  Privacy
                </Link>
                <Link href="/terms" className="text-xs text-gray-400 hover:text-gray-300">
                  Terms
                </Link>
                <Link href="/docs" className="text-xs text-gray-400 hover:text-gray-300">
                  Docs
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Spacer for fixed header */}
      <div className="lg:hidden h-16" />
    </>
  );
}
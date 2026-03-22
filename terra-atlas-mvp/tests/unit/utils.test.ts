import { describe, it, expect } from 'vitest'
import { cn } from '@/lib/utils'

describe('cn (class name utility)', () => {
  describe('basic functionality', () => {
    it('returns empty string for no inputs', () => {
      expect(cn()).toBe('')
    })

    it('returns single class name', () => {
      expect(cn('foo')).toBe('foo')
    })

    it('joins multiple class names', () => {
      expect(cn('foo', 'bar')).toBe('foo bar')
    })

    it('handles undefined and null values', () => {
      expect(cn('foo', undefined, 'bar', null)).toBe('foo bar')
    })

    it('handles boolean conditions', () => {
      expect(cn('foo', false && 'bar', 'baz')).toBe('foo baz')
      expect(cn('foo', true && 'bar', 'baz')).toBe('foo bar baz')
    })
  })

  describe('Tailwind merge behavior', () => {
    it('merges conflicting Tailwind classes', () => {
      // tailwind-merge should keep the last conflicting class
      expect(cn('p-4', 'p-8')).toBe('p-8')
      expect(cn('text-red-500', 'text-blue-500')).toBe('text-blue-500')
    })

    it('keeps non-conflicting classes', () => {
      expect(cn('p-4', 'm-4')).toBe('p-4 m-4')
    })

    it('merges padding conflicts correctly', () => {
      expect(cn('px-4', 'py-2', 'px-8')).toBe('py-2 px-8')
    })

    it('handles responsive variants', () => {
      expect(cn('md:p-4', 'md:p-8')).toBe('md:p-8')
    })

    it('handles hover states', () => {
      expect(cn('hover:bg-red-500', 'hover:bg-blue-500')).toBe('hover:bg-blue-500')
    })
  })

  describe('array and object inputs', () => {
    it('handles array of class names', () => {
      expect(cn(['foo', 'bar'])).toBe('foo bar')
    })

    it('handles object with boolean values', () => {
      expect(cn({ foo: true, bar: false, baz: true })).toBe('foo baz')
    })

    it('handles mixed inputs', () => {
      expect(cn('foo', ['bar', 'baz'], { qux: true })).toBe('foo bar baz qux')
    })
  })

  describe('real-world usage patterns', () => {
    it('handles button variant pattern', () => {
      const baseStyles = 'px-4 py-2 rounded font-medium'
      const variantStyles = 'bg-blue-500 text-white'
      const sizeOverride = 'px-6 py-3'

      // Size override should win
      const result = cn(baseStyles, variantStyles, sizeOverride)
      expect(result).toContain('px-6')
      expect(result).toContain('py-3')
      expect(result).not.toContain('px-4')
      expect(result).not.toContain('py-2')
    })

    it('handles conditional styling', () => {
      const isActive = true
      const isDisabled = false

      const result = cn(
        'button',
        isActive && 'bg-blue-500',
        isDisabled && 'opacity-50 cursor-not-allowed'
      )

      expect(result).toBe('button bg-blue-500')
    })
  })
})

'use client'

import { useEffect } from 'react'

type UseScrollRevealOptions = IntersectionObserverInit & {
  selector?: string
}

/**
 * Sets up an IntersectionObserver that toggles the `scroll-revealed` class
 * whenever elements matching the selector enter the viewport. The hook also
 * watches for lazily streamed DOM nodes so animations still trigger.
 */
export function useScrollReveal(options?: UseScrollRevealOptions) {
  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const {
      selector = '.scroll-reveal',
      threshold = 0.1,
      rootMargin = '0px 0px -100px 0px',
      ...observerOverrides
    } = options ?? {}

    const observed = new WeakSet<Element>()

    const observeNode = (node: Element) => {
      if (!observed.has(node)) {
        observed.add(node)
        observer.observe(node)
      }
    }

    const onIntersect: IntersectionObserverCallback = (entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('scroll-revealed')
          observer.unobserve(entry.target)
        }
      })
    }

    const observer = new IntersectionObserver(onIntersect, {
      threshold,
      rootMargin,
      ...observerOverrides,
    })

    const registerElements = (root: ParentNode | Document = document) => {
      if (root instanceof Element && root.matches(selector)) {
        observeNode(root)
      }

      root.querySelectorAll?.(selector).forEach((node) => {
        observeNode(node)
      })
    }

    registerElements()

    const mutationObserver = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (!(node instanceof HTMLElement)) {
            return
          }

          registerElements(node)
        })
      })
    })

    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    })

    return () => {
      mutationObserver.disconnect()
      observer.disconnect()
    }
  }, [options])
}

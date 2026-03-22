'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

export default function EarthGlobeCustom() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (!containerRef.current) return

    // Scene
    const scene = new THREE.Scene()
    
    // Camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000)
    camera.position.z = 2

    // Renderer
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(window.innerWidth, window.innerHeight)
    containerRef.current.appendChild(renderer.domElement)

    // Sphere (Earth)
    const geometry = new THREE.SphereGeometry(1, 64, 64)
    
    // Load Earth texture
    const loader = new THREE.TextureLoader()
    loader.load(
      '/globe-textures/earth-blue-marble.jpg',
      (texture) => {
        const material = new THREE.MeshStandardMaterial({ map: texture })
        const earth = new THREE.Mesh(geometry, material)
        scene.add(earth)
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
        directionalLight.position.set(5, 3, 5)
        scene.add(ambientLight, directionalLight)

        setReady(true)

        // Animation
        function animate() {
          requestAnimationFrame(animate)
          earth.rotation.y += 0.002
          renderer.render(scene, camera)
        }
        animate()
      },
      undefined,
      (err) => console.error('Texture load error:', err)
    )

    return () => {
      renderer.dispose()
      geometry.dispose()
    }
  }, [])

  return (
    <div ref={containerRef} className="absolute inset-0" style={{ border: '10px solid red' }}>
      {/* ULTRA VISIBLE MARKER */}
      <div className="absolute top-0 left-0 right-0 bg-red-600 text-white text-center py-4 text-2xl font-bold z-50">
        🌍 BRAND NEW EARTH GLOBE COMPONENT - RED MARKER 🌍
      </div>
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center text-white text-2xl">
          Loading new Earth globe...
        </div>
      )}
    </div>
  )
}

'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

interface ProfessionalGlobeProps {
  projects?: Array<{
    lat: number
    lng: number
    name: string
    type: string
    capacity: number
  }>
}

export default function ProfessionalGlobe({ projects = [] }: ProfessionalGlobeProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!containerRef.current) return

    let renderer: THREE.WebGLRenderer | null = null
    let animationId: number | null = null

    try {
      // Scene setup
      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x000000) // Black space background

      // Camera setup - closer for bigger globe
      const width = containerRef.current.clientWidth
      const height = containerRef.current.clientHeight
      const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000)
      camera.position.set(0, 0, 1.6)

      // Renderer setup with WebGL fallback
      try {
        renderer = new THREE.WebGLRenderer({
          alpha: true,
          antialias: true,
          powerPreference: 'high-performance'
        })
        renderer.setSize(width, height)
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
        containerRef.current.appendChild(renderer.domElement)
      } catch (webglError) {
        console.error('WebGL initialization failed:', webglError)
        setError('WebGL is not available. Please use a modern browser with WebGL support.')
        setLoading(false)
        return
      }

      // Earth geometry - higher detail
      const geometry = new THREE.SphereGeometry(1, 128, 128)

      // Create star field
      const starsGeometry = new THREE.BufferGeometry()
      const starsMaterial = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 2,
        sizeAttenuation: true
      })

      const starsVertices = []
      for (let i = 0; i < 10000; i++) {
        const x = (Math.random() - 0.5) * 2000
        const y = (Math.random() - 0.5) * 2000
        const z = (Math.random() - 0.5) * 2000
        starsVertices.push(x, y, z)
      }

      starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3))
      const starField = new THREE.Points(starsGeometry, starsMaterial)
      scene.add(starField)

      // Create atmosphere glow
      const atmosphereGeometry = new THREE.SphereGeometry(1.15, 64, 64)
      const atmosphereMaterial = new THREE.ShaderMaterial({
        vertexShader: `
          varying vec3 vNormal;
          void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          varying vec3 vNormal;
          void main() {
            float intensity = pow(0.6 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
            gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * intensity;
          }
        `,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        transparent: true
      })
      const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial)
      scene.add(atmosphere)

      // Load Earth texture with fallback
      const textureLoader = new THREE.TextureLoader()

      textureLoader.load(
        '/globe-textures/earth-blue-marble.jpg',
        (texture) => {
          // Success - create material with texture
          texture.anisotropy = renderer!.capabilities.getMaxAnisotropy()

          const material = new THREE.MeshStandardMaterial({
            map: texture,
            roughness: 0.9,
            metalness: 0.1,
            emissive: 0x112244,
            emissiveIntensity: 0.05
          })

          const earth = new THREE.Mesh(geometry, material)
          scene.add(earth)

          // Better lighting setup
          const ambientLight = new THREE.AmbientLight(0xffffff, 1.0)
          const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2)
          directionalLight.position.set(5, 3, 5)

          const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3)
          fillLight.position.set(-5, -3, -5)

          scene.add(ambientLight, directionalLight, fillLight)

          setLoading(false)

          // Animation loop with smooth rotation
          const animate = () => {
            if (!renderer) return

            animationId = requestAnimationFrame(animate)
            earth.rotation.y += 0.002
            atmosphere.rotation.y += 0.001
            starField.rotation.y += 0.0001
            renderer.render(scene, camera)
          }
          animate()
        },
        undefined,
        (err) => {
          console.error('Failed to load Earth texture:', err)

          // Fallback: use beautiful gradient if texture fails
          const fallbackMaterial = new THREE.MeshStandardMaterial({
            color: 0x1e40af,
            roughness: 0.9,
            metalness: 0.1,
            emissive: 0x1e3a8a,
            emissiveIntensity: 0.1
          })

          const earth = new THREE.Mesh(geometry, fallbackMaterial)
          scene.add(earth)

          // Better lighting setup
          const ambientLight = new THREE.AmbientLight(0xffffff, 1.0)
          const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2)
          directionalLight.position.set(5, 3, 5)

          const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3)
          fillLight.position.set(-5, -3, -5)

          scene.add(ambientLight, directionalLight, fillLight)

          setLoading(false)

          // Animation loop with smooth rotation
          const animate = () => {
            if (!renderer) return

            animationId = requestAnimationFrame(animate)
            earth.rotation.y += 0.002
            atmosphere.rotation.y += 0.001
            starField.rotation.y += 0.0001
            renderer.render(scene, camera)
          }
          animate()
        }
      )

      // Handle window resize
      const handleResize = () => {
        if (!containerRef.current || !renderer) return

        const width = containerRef.current.clientWidth
        const height = containerRef.current.clientHeight

        camera.aspect = width / height
        camera.updateProjectionMatrix()
        renderer.setSize(width, height)
      }

      window.addEventListener('resize', handleResize)

      // Cleanup
      return () => {
        window.removeEventListener('resize', handleResize)

        if (animationId !== null) {
          cancelAnimationFrame(animationId)
        }

        if (renderer) {
          renderer.dispose()
        }

        geometry.dispose()

        if (containerRef.current && renderer) {
          containerRef.current.removeChild(renderer.domElement)
        }
      }

    } catch (err) {
      console.error('Globe initialization error:', err)
      setError('Failed to initialize 3D globe. Please refresh the page.')
      setLoading(false)
    }
  }, [])

  if (error) {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-blue-950/50 to-black">
        <div className="text-center px-6">
          <div className="text-blue-400 text-6xl mb-4">🌍</div>
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="absolute inset-0">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-blue-950/50 to-black">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4 mx-auto"></div>
            <p className="text-gray-400 text-sm">Loading globe...</p>
          </div>
        </div>
      )}
    </div>
  )
}
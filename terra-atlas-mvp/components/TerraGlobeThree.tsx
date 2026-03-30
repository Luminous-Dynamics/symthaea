// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import logger from '@/lib/logger'

interface Project {
  lat: number
  lng: number
  type: string
  name: string
  capacity: number
  country?: string
}

interface TerraGlobeThreeProps {
  projects?: Project[]
}

// Fetch default project data
const getDefaultProjects = (): Project[] => {
  return [
    // Major solar projects
    { lat: 37.07, lng: -116.05, type: 'solar', name: 'Ivanpah Solar', capacity: 392, country: 'USA' },
    { lat: 34.82, lng: -118.39, type: 'solar', name: 'Solar Star', capacity: 579, country: 'USA' },
    { lat: 24.46, lng: 32.74, type: 'solar', name: 'Benban Solar Park', capacity: 1650, country: 'Egypt' },

    // Wind farms
    { lat: 40.15, lng: 95.68, type: 'wind', name: 'Gansu Wind Farm', capacity: 20000, country: 'China' },
    { lat: 53.88, lng: 1.75, type: 'wind', name: 'Hornsea 2', capacity: 1320, country: 'UK' },
    { lat: 35.08, lng: -118.33, type: 'wind', name: 'Alta Wind Energy', capacity: 1548, country: 'USA' },

    // Hydro
    { lat: 30.82, lng: 111.00, type: 'hydro', name: 'Three Gorges Dam', capacity: 22500, country: 'China' },
    { lat: -25.41, lng: -54.59, type: 'hydro', name: 'Itaipu Dam', capacity: 14000, country: 'Brazil' },
    { lat: 47.95, lng: -118.98, type: 'hydro', name: 'Grand Coulee', capacity: 6809, country: 'USA' },

    // Geothermal
    { lat: 38.78, lng: -122.75, type: 'geothermal', name: 'The Geysers', capacity: 1517, country: 'USA' },
    { lat: 43.64, lng: 10.80, type: 'geothermal', name: 'Larderello', capacity: 769, country: 'Italy' },
  ]
}

export default function TerraGlobeThree({ projects: initialProjects }: TerraGlobeThreeProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const globeRef = useRef<THREE.Mesh | null>(null)
  const markersRef = useRef<HTMLDivElement[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [projects, setProjects] = useState<Project[]>(initialProjects || getDefaultProjects())

  useEffect(() => {
    if (!containerRef.current) return

    logger.info('🌍 Initializing custom Three.js globe...')
    logger.info('📊 Projects to display:', projects.length)

    // Scene setup
    const scene = new THREE.Scene()
    sceneRef.current = scene

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      45,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    )
    camera.position.z = 2.5
    cameraRef.current = camera

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true
    })
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    containerRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(5, 3, 5)
    scene.add(directionalLight)

    // Create Earth sphere
    const geometry = new THREE.SphereGeometry(1, 64, 64)

    // Load Earth texture
    const textureLoader = new THREE.TextureLoader()
    const earthTexture = textureLoader.load(
      '/globe-textures/earth-blue-marble.jpg',
      () => {
        logger.info('✅ Earth texture loaded successfully!')
        setIsLoading(false)
      },
      undefined,
      (error) => {
        logger.error('❌ Failed to load Earth texture:', error)
        setIsLoading(false)
      }
    )

    // Optional: Load bump map for surface detail
    const bumpTexture = textureLoader.load('/globe-textures/earth-topology.png')

    const material = new THREE.MeshPhongMaterial({
      map: earthTexture,
      bumpMap: bumpTexture,
      bumpScale: 0.05,
      specular: new THREE.Color(0x333333),
      shininess: 10
    })

    const globe = new THREE.Mesh(geometry, material)
    scene.add(globe)
    globeRef.current = globe

    logger.info('✨ Globe mesh created and added to scene')

    // Animation loop
    let animationId: number
    const animate = () => {
      animationId = requestAnimationFrame(animate)

      // Slow rotation
      if (globeRef.current) {
        globeRef.current.rotation.y += 0.001
      }

      renderer.render(scene, camera)
      updateMarkerPositions()
    }
    animate()

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !camera || !renderer) return

      const width = containerRef.current.clientWidth
      const height = containerRef.current.clientHeight

      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }
    window.addEventListener('resize', handleResize)

    // Mouse controls
    let isDragging = false
    let previousMousePosition = { x: 0, y: 0 }

    const handleMouseDown = (e: MouseEvent) => {
      isDragging = true
      previousMousePosition = { x: e.clientX, y: e.clientY }
    }

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !globeRef.current) return

      const deltaX = e.clientX - previousMousePosition.x
      const deltaY = e.clientY - previousMousePosition.y

      globeRef.current.rotation.y += deltaX * 0.005
      globeRef.current.rotation.x += deltaY * 0.005

      // Limit vertical rotation
      globeRef.current.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, globeRef.current.rotation.x))

      previousMousePosition = { x: e.clientX, y: e.clientY }
    }

    const handleMouseUp = () => {
      isDragging = false
    }

    containerRef.current.addEventListener('mousedown', handleMouseDown)
    containerRef.current.addEventListener('mousemove', handleMouseMove)
    containerRef.current.addEventListener('mouseup', handleMouseUp)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (containerRef.current) {
        containerRef.current.removeEventListener('mousedown', handleMouseDown)
        containerRef.current.removeEventListener('mousemove', handleMouseMove)
        containerRef.current.removeEventListener('mouseup', handleMouseUp)
        containerRef.current.removeChild(renderer.domElement)
      }
      cancelAnimationFrame(animationId)
      geometry.dispose()
      material.dispose()
      renderer.dispose()
    }
  }, [])

  // Convert lat/lng to 3D coordinates
  const latLngToVector3 = (lat: number, lng: number, radius: number = 1) => {
    const phi = (90 - lat) * (Math.PI / 180)
    const theta = (lng + 180) * (Math.PI / 180)

    return new THREE.Vector3(
      -radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta)
    )
  }

  // Update marker positions to follow globe rotation
  const updateMarkerPositions = () => {
    if (!globeRef.current || !cameraRef.current || !containerRef.current) return

    markersRef.current.forEach((marker, index) => {
      const project = projects[index]
      if (!project || !marker) return

      const position = latLngToVector3(project.lat, project.lng, 1.02)
      position.applyQuaternion(globeRef.current!.quaternion)

      // Project 3D position to 2D screen
      const vector = position.clone()
      vector.project(cameraRef.current!)

      const x = (vector.x * 0.5 + 0.5) * containerRef.current!.clientWidth
      const y = (vector.y * -0.5 + 0.5) * containerRef.current!.clientHeight

      // Hide markers on back side of globe
      const isVisible = vector.z < 1

      marker.style.transform = `translate(-50%, -50%) translate(${x}px, ${y}px)`
      marker.style.opacity = isVisible ? '1' : '0'
      marker.style.pointerEvents = isVisible ? 'auto' : 'none'
    })
  }

  // Energy type icons and colors
  const getTypeIcon = (type: string): string => {
    const icons: Record<string, string> = {
      solar: '☀️',
      wind: '💨',
      hydro: '💧',
      geothermal: '🌋',
      nuclear: '⚛️',
      renewable: '⚡'
    }
    return icons[type] || '⚡'
  }

  const getTypeColor = (type: string): string => {
    const colors: Record<string, string> = {
      solar: '#fbbf24',
      wind: '#60a5fa',
      hydro: '#34d399',
      geothermal: '#f97316',
      nuclear: '#a855f7',
      renewable: '#06b6d4'
    }
    return colors[type] || '#a855f7'
  }

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 w-full h-full"
      style={{ cursor: 'grab', border: '5px solid lime' }}
    >
      {/* VERIFICATION MARKER - Remove after confirming deployment */}
      <div className="absolute top-4 left-4 bg-lime-500 text-black px-4 py-2 rounded font-bold z-50">
        ✅ CUSTOM THREE.JS GLOBE v2
      </div>

      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center text-white">
          <div className="text-center">
            <div className="text-2xl mb-2">🌍</div>
            <div>Loading Earth...</div>
          </div>
        </div>
      )}

      {/* HTML overlay markers */}
      <div className="absolute inset-0 pointer-events-none">
        {projects.map((project, index) => (
          <div
            key={`${project.lat}-${project.lng}-${index}`}
            ref={(el) => {
              if (el) markersRef.current[index] = el
            }}
            className="absolute pointer-events-auto transition-opacity duration-300"
            style={{
              transform: 'translate(-50%, -50%)',
              opacity: 0
            }}
          >
            <div
              className="px-2 py-1 rounded-lg text-xs font-semibold cursor-pointer hover:scale-125 transition-transform"
              style={{
                background: 'rgba(0, 0, 0, 0.8)',
                border: `1px solid ${getTypeColor(project.type)}`,
                color: getTypeColor(project.type),
                backdropFilter: 'blur(4px)',
                boxShadow: `0 2px 8px rgba(0, 0, 0, 0.3)`
              }}
              title={`${project.name}\n${project.capacity} MW\n${project.country || ''}`}
            >
              {getTypeIcon(project.type)}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
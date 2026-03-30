// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

export default function SimpleSpinningGlobe() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingProgress, setLoadingProgress] = useState(0)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const animationIdRef = useRef<number | null>(null)
  const rotationVelocityRef = useRef({ x: 0, y: 0 })

  useEffect(() => {
    if (!containerRef.current) return

    const initTimer = setTimeout(() => {
      try {
        const width = containerRef.current?.clientWidth || 800
        const height = containerRef.current?.clientHeight || 600

        console.log('🌍 Creating elegant minimalist globe')

        // Scene with very dark space background - makes holographic Earth pop
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0x000408) // Almost black

        // Camera - perfectly centered to show entire globe (both poles)
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000)
        camera.position.z = 4.2  // Balanced zoom - shows full globe
        camera.position.y = 0.0  // Perfectly centered vertically

        // Renderer
        const renderer = new THREE.WebGLRenderer({
          antialias: true,
          alpha: false
        })
        renderer.setSize(width, height)
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

        if (containerRef.current) {
          containerRef.current.appendChild(renderer.domElement)
          rendererRef.current = renderer
        }

        // 🌟 ENHANCED STAR FIELD - Multiple depth layers with parallax
        const starLayers = []

        // Layer 1: Distant stars (small, slow)
        const stars1Geometry = new THREE.BufferGeometry()
        const stars1Vertices = []
        const stars1Colors = []
        const stars1Sizes = []

        for (let i = 0; i < 800; i++) {
          const x = (Math.random() - 0.5) * 2000
          const y = (Math.random() - 0.5) * 2000
          const z = (Math.random() - 0.5) * 2000
          stars1Vertices.push(x, y, z)

          // Color variation - blue to white
          const colorVariation = Math.random()
          stars1Colors.push(0.8 + colorVariation * 0.2, 0.85 + colorVariation * 0.15, 1.0)
          stars1Sizes.push(0.8 + Math.random() * 0.5)
        }

        stars1Geometry.setAttribute('position', new THREE.Float32BufferAttribute(stars1Vertices, 3))
        stars1Geometry.setAttribute('color', new THREE.Float32BufferAttribute(stars1Colors, 3))
        stars1Geometry.setAttribute('size', new THREE.Float32BufferAttribute(stars1Sizes, 1))

        const stars1Material = new THREE.PointsMaterial({
          size: 1.0,
          sizeAttenuation: true,
          vertexColors: true,
          transparent: true,
          opacity: 0.5
        })

        const stars1 = new THREE.Points(stars1Geometry, stars1Material)
        scene.add(stars1)
        starLayers.push({ mesh: stars1, speed: 0.00002 })

        // Layer 2: Medium stars (brighter, medium speed)
        const stars2Geometry = new THREE.BufferGeometry()
        const stars2Vertices = []
        const stars2Colors = []
        const stars2Sizes = []

        for (let i = 0; i < 400; i++) {
          const x = (Math.random() - 0.5) * 1200
          const y = (Math.random() - 0.5) * 1200
          const z = (Math.random() - 0.5) * 1200
          stars2Vertices.push(x, y, z)

          const colorVariation = Math.random()
          stars2Colors.push(0.9 + colorVariation * 0.1, 0.95 + colorVariation * 0.05, 1.0)
          stars2Sizes.push(1.2 + Math.random() * 0.8)
        }

        stars2Geometry.setAttribute('position', new THREE.Float32BufferAttribute(stars2Vertices, 3))
        stars2Geometry.setAttribute('color', new THREE.Float32BufferAttribute(stars2Colors, 3))
        stars2Geometry.setAttribute('size', new THREE.Float32BufferAttribute(stars2Sizes, 1))

        const stars2Material = new THREE.PointsMaterial({
          size: 1.5,
          sizeAttenuation: true,
          vertexColors: true,
          transparent: true,
          opacity: 0.7
        })

        const stars2 = new THREE.Points(stars2Geometry, stars2Material)
        scene.add(stars2)
        starLayers.push({ mesh: stars2, speed: 0.00005 })

        // Layer 3: Bright nearby stars (largest, fastest, occasional twinkle)
        const stars3Geometry = new THREE.BufferGeometry()
        const stars3Vertices = []
        const stars3Colors = []
        const stars3Sizes = []

        for (let i = 0; i < 150; i++) {
          const x = (Math.random() - 0.5) * 800
          const y = (Math.random() - 0.5) * 800
          const z = (Math.random() - 0.5) * 800
          stars3Vertices.push(x, y, z)

          stars3Colors.push(1.0, 0.98, 1.0)
          stars3Sizes.push(1.5 + Math.random() * 1.0)
        }

        stars3Geometry.setAttribute('position', new THREE.Float32BufferAttribute(stars3Vertices, 3))
        stars3Geometry.setAttribute('color', new THREE.Float32BufferAttribute(stars3Colors, 3))
        stars3Geometry.setAttribute('size', new THREE.Float32BufferAttribute(stars3Sizes, 1))

        const stars3Material = new THREE.PointsMaterial({
          size: 2.0,
          sizeAttenuation: true,
          vertexColors: true,
          transparent: true,
          opacity: 0.8
        })

        const stars3 = new THREE.Points(stars3Geometry, stars3Material)
        scene.add(stars3)
        starLayers.push({ mesh: stars3, speed: 0.0001 })

        // Elegant Earth sphere
        const geometry = new THREE.SphereGeometry(1, 128, 128)

        // 🚀 PROGRESSIVE TEXTURE LOADING - Low-res instant, high-res background
        const textureLoader = new THREE.TextureLoader()
        let lowResLoaded = 0
        let highResLoaded = 0
        const totalTextures = 2

        // Track loading progress
        const updateProgress = (stage: 'low' | 'high') => {
          if (stage === 'low') {
            lowResLoaded++
            const progress = (lowResLoaded / totalTextures) * 50 // 0-50%
            setLoadingProgress(progress)
            if (lowResLoaded === totalTextures) {
              console.log('⚡ Low-res textures loaded - showing globe!')
              setTimeout(() => setLoading(false), 200) // Hide loading indicator
            }
          } else {
            highResLoaded++
            const progress = 50 + (highResLoaded / totalTextures) * 50 // 50-100%
            setLoadingProgress(progress)
            if (highResLoaded === totalTextures) {
              console.log('✨ High-res textures loaded - enhanced quality!')
            }
          }
        }

        // STEP 1: Load low-res WebP textures instantly (~22k total, <200ms)
        const earthTextureLow = textureLoader.load(
          '/textures/earth-blue-marble-low.webp',
          () => {
            console.log('⚡ Low-res Earth texture loaded (17k)')
            updateProgress('low')
          },
          undefined,
          (error) => {
            console.error('❌ Low-res Earth texture failed:', error)
            updateProgress('low')
          }
        )

        const boundariesTextureLow = textureLoader.load(
          '/textures/earth-topology-low.webp',
          () => {
            console.log('⚡ Low-res topology texture loaded (5k)')
            updateProgress('low')
          },
          undefined,
          (error) => {
            console.warn('❌ Low-res topology texture failed:', error)
            updateProgress('low')
          }
        )

        // Start with low-res textures
        let earthTexture = earthTextureLow
        let boundariesTexture = boundariesTextureLow

        // Dark glass reflective Earth shader - Purple land, Navy ocean
        const earthMaterial = new THREE.ShaderMaterial({
          vertexShader: `
            varying vec3 vNormal;
            varying vec2 vUv;
            varying vec3 vViewPosition;

            void main() {
              vNormal = normalize(normalMatrix * normal);
              vUv = uv;
              vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
              vViewPosition = -mvPosition.xyz;
              gl_Position = projectionMatrix * mvPosition;
            }
          `,
          fragmentShader: `
            uniform float time;
            uniform sampler2D earthTexture;
            uniform sampler2D boundariesTexture;
            varying vec3 vNormal;
            varying vec2 vUv;
            varying vec3 vViewPosition;

            void main() {
              // Sample Earth texture
              vec4 texColor = texture2D(earthTexture, vUv);

              // Improved land/ocean detection
              // Oceans are darker AND have blue dominance
              float brightness = (texColor.r + texColor.g + texColor.b) / 3.0;
              float blueDominance = texColor.b - ((texColor.r + texColor.g) * 0.5);

              // Combine brightness and blue channel for better ocean detection
              float isWater = smoothstep(0.05, 0.15, blueDominance);
              float isDark = smoothstep(0.20, 0.10, brightness);
              float isOcean = max(isWater, isDark);
              float isLand = 1.0 - isOcean;

              // 🌿 DARK EMERALD CONTINENTS - Healing planet aesthetic
              // Use texture to determine terrain type
              float terrainHeight = (texColor.r + texColor.g) * 0.5; // Mountains are brighter

              // Mountain ranges - emerald highlights
              float mountainRange = smoothstep(0.25, 0.30, terrainHeight) * (1.0 - smoothstep(0.30, 0.35, terrainHeight));
              vec3 mountainColor = vec3(0.015, 0.035, 0.025); // Brighter emerald peaks

              // Desert vs forest variation (using green channel)
              float forestness = texColor.g / (texColor.r + texColor.g + 0.01);
              vec3 desertColor = vec3(0.012, 0.020, 0.010); // Darker emerald, less green
              vec3 forestColor = vec3(0.006, 0.028, 0.018); // Deep emerald forests

              // DARK EMERALD LAND - Matches site's emerald-cyan palette
              vec3 landColor = mix(desertColor, forestColor, forestness);
              landColor = mix(landColor, mountainColor, mountainRange * 0.5);

              // 🌊 DARK OCEAN DEPTHS - Realistic nighttime ocean with depth variation
              // Ocean depth based on texture brightness (darker = deeper trenches)
              float oceanDepth = smoothstep(0.02, 0.12, brightness);

              // Enhanced ocean visibility - brighter blues against black space
              vec3 deepTrench = vec3(0.008, 0.020, 0.065);    // Visible deep trenches
              vec3 deepOcean = vec3(0.015, 0.030, 0.085);     // Deep midnight blue
              vec3 midOcean = vec3(0.020, 0.042, 0.105);      // Medium deep blue
              vec3 shallowOcean = vec3(0.030, 0.060, 0.140);  // Shallow coastal blue

              // Very gentle breathing effect - no jarring waves
              float oceanBreath = sin(time * 0.15) * 0.001 + 1.0;

              // Multi-level depth gradient for realism
              vec3 oceanColor;
              if (oceanDepth < 0.3) {
                // Deep trenches - almost black
                oceanColor = mix(deepTrench, deepOcean, oceanDepth / 0.3);
              } else if (oceanDepth < 0.7) {
                // Mid-depth ocean
                oceanColor = mix(deepOcean, midOcean, (oceanDepth - 0.3) / 0.4);
              } else {
                // Shallow coastal waters
                oceanColor = mix(midOcean, shallowOcean, (oceanDepth - 0.7) / 0.3);
              }

              oceanColor *= oceanBreath; // Gentle breathing

              // Subtle coastal glow - very faint emerald hint at shorelines
              float coastalGlow = smoothstep(0.10, 0.13, brightness) * (1.0 - smoothstep(0.13, 0.16, brightness));
              vec3 coastGlowColor = vec3(0.005, 0.012, 0.018); // Very subtle blue-teal
              oceanColor = mix(oceanColor, coastGlowColor, coastalGlow * 0.2);

              // Ice caps - barely visible
              float polarness = abs(vUv.y - 0.5) * 2.0;
              float iceCaps = smoothstep(0.88, 0.95, polarness);
              vec3 iceColor = vec3(0.015, 0.018, 0.020); // Very dark ice

              // Mix base colors
              vec3 baseColor = mix(oceanColor, landColor, isLand);
              baseColor = mix(baseColor, iceColor, iceCaps * isLand);

              // Political boundaries detection
              vec4 boundariesColor = texture2D(boundariesTexture, vUv);
              float boundaryBrightness = (boundariesColor.r + boundariesColor.g + boundariesColor.b) / 3.0;

              // Edge detection for borders
              float edgeWidth = 0.002;
              vec4 bRight = texture2D(boundariesTexture, vUv + vec2(edgeWidth, 0.0));
              vec4 bLeft = texture2D(boundariesTexture, vUv - vec2(edgeWidth, 0.0));
              vec4 bUp = texture2D(boundariesTexture, vUv + vec2(0.0, edgeWidth));
              vec4 bDown = texture2D(boundariesTexture, vUv - vec2(0.0, edgeWidth));

              float edgeH = length(bRight.rgb - bLeft.rgb);
              float edgeV = length(bUp.rgb - bDown.rgb);
              float edgeDetection = max(edgeH, edgeV);

              // Combine border detection methods
              float darkLines = smoothstep(0.4, 0.3, boundaryBrightness);
              float edges = smoothstep(0.08, 0.18, edgeDetection);
              float politicalBorder = max(darkLines, edges);

              // Calming emerald borders - match site palette
              vec3 borderColor = vec3(0.12, 0.28, 0.25); // Peaceful emerald-teal

              // Add gentle glowing borders to land
              baseColor = mix(baseColor, borderColor, politicalBorder * isLand * 0.5);

              // 💫 CALMING GRID LINES - Gentle resonance, no jarring flashes
              float gridSpacing = 20.0; // Clean spacing - 20 degrees between lines
              float latLines = fract(vUv.y * 180.0 / gridSpacing);
              float lonLines = fract(vUv.x * 360.0 / gridSpacing);

              // Softer, more refined grid lines
              float latGrid = smoothstep(0.985, 1.0, latLines) + smoothstep(0.015, 0.0, latLines);
              float lonGrid = smoothstep(0.985, 1.0, lonLines) + smoothstep(0.015, 0.0, lonLines);
              float techGrid = max(latGrid, lonGrid);

              // Very gentle breathing pulse - slow and calming
              float gridBreath = sin(time * 0.2) * 0.08 + 0.92;

              // Polar regions slightly brighter (peaceful gradient)
              float polarStrength = smoothstep(0.7, 0.95, abs(vUv.y - 0.5) * 2.0);

              // Calming emerald-cyan grid - matches site colors
              vec3 gridColor = vec3(0.10, 0.25, 0.30); // Peaceful emerald-cyan

              baseColor = mix(baseColor, gridColor, techGrid * 0.35 * gridBreath * (1.0 + polarStrength * 0.2));

              // 💎 CRYSTAL-CLEAR continent edges - much more visible
              float continentEdgeWidth = 0.003;

              // Multi-directional edge detection for clearer outlines
              vec4 texRight = texture2D(earthTexture, vUv + vec2(continentEdgeWidth, 0.0));
              vec4 texLeft = texture2D(earthTexture, vUv - vec2(continentEdgeWidth, 0.0));
              vec4 texUp = texture2D(earthTexture, vUv + vec2(0.0, continentEdgeWidth));
              vec4 texDown = texture2D(earthTexture, vUv - vec2(0.0, continentEdgeWidth));

              float brightRight = (texRight.r + texRight.g + texRight.b) / 3.0;
              float brightLeft = (texLeft.r + texLeft.g + texLeft.b) / 3.0;
              float brightUp = (texUp.r + texUp.g + texUp.b) / 3.0;
              float brightDown = (texDown.r + texDown.g + texDown.b) / 3.0;

              float landRight = step(0.12, brightRight);
              float landLeft = step(0.12, brightLeft);
              float landUp = step(0.12, brightUp);
              float landDown = step(0.12, brightDown);

              // Combine horizontal and vertical edges
              float continentEdgeH = abs(landRight - landLeft);
              float continentEdgeV = abs(landUp - landDown);
              float landEdge = max(continentEdgeH, continentEdgeV);

              // Calming continent edges - peaceful emerald-cyan
              vec3 edgeGlow = vec3(0.12, 0.28, 0.32); // Gentle emerald-cyan
              baseColor += edgeGlow * landEdge * isLand * 0.8;

              // 🌈 AURORA BOREALIS - Magical polar lights
              float auroraPolarStrength = smoothstep(0.75, 0.95, abs(vUv.y - 0.5) * 2.0);

              // Animated aurora waves
              float auroraWave1 = sin(vUv.x * 15.0 + time * 0.6) * 0.5 + 0.5;
              float auroraWave2 = sin(vUv.x * 20.0 - time * 0.8) * 0.5 + 0.5;
              float auroraPattern = (auroraWave1 + auroraWave2) * 0.5;

              // Green/blue aurora colors
              vec3 auroraGreen = vec3(0.1, 0.4, 0.2);
              vec3 auroraBlue = vec3(0.05, 0.2, 0.4);
              vec3 auroraColor = mix(auroraGreen, auroraBlue, auroraPattern);

              // Add aurora at poles with gentle wave motion
              float auroraIntensity = auroraPolarStrength * auroraPattern * 0.15;
              baseColor += auroraColor * auroraIntensity;

              // 🌍 SMOOTH POPULATION GLOW - Regional gradients, no texture
              // Define populated regions by latitude/longitude with smooth gradients

              // Northern Hemisphere population centers (North America, Europe, Asia)
              float northernLat = smoothstep(0.35, 0.55, vUv.y); // 30-50° North
              float northernFalloff = smoothstep(0.75, 0.55, vUv.y); // Fade above 50°N
              float northernBand = northernLat * northernFalloff;

              // Eastern Hemisphere (Europe, Asia) - longitude 0° to 180°E
              float easternHemisphere = smoothstep(0.0, 0.25, vUv.x) * smoothstep(0.75, 0.5, vUv.x);

              // Western Hemisphere (Americas) - longitude 180°W to 0°
              float westernHemisphere = smoothstep(0.0, 0.1, vUv.x) + smoothstep(1.0, 0.85, vUv.x);
              westernHemisphere = min(westernHemisphere, 1.0);

              // Asian glow - Eastern + Northern
              float asiaGlow = easternHemisphere * northernBand * 0.6;

              // European glow - Centered around 0-30°E
              float europeGlow = smoothstep(0.05, 0.15, vUv.x) * smoothstep(0.25, 0.15, vUv.x) * northernBand * 0.5;

              // North American glow - Western + Northern
              float northAmericaGlow = westernHemisphere * northernBand * 0.5;

              // Coastal enhancement - population tends to cluster near coasts
              float coastalProximity = smoothstep(0.10, 0.15, brightness) * (1.0 - smoothstep(0.15, 0.20, brightness));

              // Combine all regional glows
              float populationDensity = (asiaGlow + europeGlow + northAmericaGlow) * isLand;
              populationDensity += coastalProximity * isLand * 0.3; // Coastal boost

              // Calming emerald/cyan population glow - matches site palette
              vec3 populationColor = mix(
                vec3(0.5, 0.9, 0.7),  // Soft emerald
                vec3(0.4, 0.85, 0.95), // Soft cyan
                smoothstep(0.3, 0.7, vUv.x) // Smooth east-west color gradient
              );

              // Apply smooth population glow
              vec3 populationGlow = populationColor * populationDensity * 0.4;

              // Gentle breathing pulse - calming rhythm
              float glowPulse = sin(time * 0.5) * 0.08 + 0.92;
              baseColor += populationGlow * glowPulse;

              // NIGHTTIME: Minimal reflections - let city lights shine
              vec3 viewDir = normalize(vViewPosition);
              float fresnelTerm = pow(1.0 - abs(dot(viewDir, vNormal)), 3.0);

              // Very subtle reflection for nighttime
              vec3 reflectionColor = vec3(0.02, 0.04, 0.05); // Barely visible
              baseColor = mix(baseColor, reflectionColor, fresnelTerm * 0.2 * isLand);

              // Minimal specular - nighttime Earth doesn't need bright highlights
              vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
              vec3 halfVector = normalize(lightDir + viewDir);
              float specular = pow(max(dot(vNormal, halfVector), 0.0), 60.0);
              baseColor += vec3(0.03, 0.08, 0.10) * specular * isLand * 0.3; // Very subtle

              // Subtle rim lighting for edge definition
              float rimPower = 2.5;
              float rim = 1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0)));
              float rimIntensity = pow(rim, rimPower);

              // Soft cyan rim light - just enough to see the sphere shape
              vec3 rimColor = vec3(0.08, 0.18, 0.22) * rimIntensity * 0.5;

              // Combine all elements
              vec3 finalColor = baseColor + rimColor;

              gl_FragColor = vec4(finalColor, 1.0);
            }
          `,
          uniforms: {
            time: { value: 0 },
            earthTexture: { value: earthTexture },
            boundariesTexture: { value: boundariesTexture }
          }
        })

        const earth = new THREE.Mesh(geometry, earthMaterial)
        // Initial rotation to show Americas/Atlantic view
        earth.rotation.y = -0.5
        scene.add(earth)

        // STEP 2: Load high-res WebP textures in background (~312k total, <1s)
        // These swap in seamlessly once loaded for enhanced quality
        textureLoader.load(
          '/textures/earth-blue-marble.webp',
          (texture) => {
            console.log('✨ High-res Earth texture loaded (256k)')
            updateProgress('high')
            // Swap to high-res seamlessly
            earthMaterial.uniforms.earthTexture.value = texture
            earthMaterial.needsUpdate = true
          },
          undefined,
          (error) => {
            console.error('❌ High-res Earth texture failed:', error)
            updateProgress('high')
          }
        )

        textureLoader.load(
          '/textures/earth-topology.webp',
          (texture) => {
            console.log('✨ High-res topology texture loaded (56k)')
            updateProgress('high')
            // Swap to high-res seamlessly
            earthMaterial.uniforms.boundariesTexture.value = texture
            earthMaterial.needsUpdate = true
          },
          undefined,
          (error) => {
            console.warn('❌ High-res topology texture failed:', error)
            updateProgress('high')
          }
        )

        // NIGHTTIME: Subtle atmospheric glow - doesn't compete with city lights
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
            uniform float time;
            varying vec3 vNormal;

            void main() {
              // Very subtle pulse
              float pulse = sin(time * 0.3) * 0.08 + 0.92;

              // Enhanced glow - more visible against black space
              float intensity = pow(0.75 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.5);

              // Brighter cyan atmospheric color - matches site palette
              vec3 atmosphereColor = vec3(0.05, 0.15, 0.25);

              gl_FragColor = vec4(atmosphereColor, 1.0) * intensity * 0.50 * pulse;
            }
          `,
          uniforms: {
            time: { value: 0 }
          },
          blending: THREE.AdditiveBlending,
          side: THREE.BackSide,
          transparent: true
        })

        const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial)
        scene.add(atmosphere)

        // NIGHTTIME: Very subtle outer atmosphere layer
        const atmosphere2Geometry = new THREE.SphereGeometry(1.22, 64, 64)
        const atmosphere2Material = new THREE.ShaderMaterial({
          vertexShader: `
            varying vec3 vNormal;
            void main() {
              vNormal = normalize(normalMatrix * normal);
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
          fragmentShader: `
            uniform float time;
            varying vec3 vNormal;

            void main() {
              // Enhanced outer glow - more visible
              float pulse = sin(time * 0.2 + 1.5) * 0.08 + 0.92;
              float intensity = pow(0.82 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.2);

              // Brighter cyan outer glow - creates depth
              vec3 color = vec3(0.03, 0.10, 0.18);

              gl_FragColor = vec4(color, 1.0) * intensity * 0.35 * pulse;
            }
          `,
          uniforms: {
            time: { value: 0 }
          },
          blending: THREE.AdditiveBlending,
          side: THREE.BackSide,
          transparent: true
        })

        const atmosphere2 = new THREE.Mesh(atmosphere2Geometry, atmosphere2Material)
        scene.add(atmosphere2)

        // Soft ambient lighting
        const ambientLight = new THREE.AmbientLight(0x667788, 0.6)
        const directionalLight = new THREE.DirectionalLight(0x99aacc, 0.5)
        directionalLight.position.set(5, 3, 5)
        scene.add(ambientLight, directionalLight)

        // Mouse interaction for rotation
        let isDragging = false
        let previousMousePosition = { x: 0, y: 0 }

        const handleMouseDown = (e: MouseEvent) => {
          isDragging = true
          previousMousePosition = { x: e.clientX, y: e.clientY }
          if (containerRef.current) {
            containerRef.current.style.cursor = 'grabbing'
          }
        }

        const handleMouseMove = (e: MouseEvent) => {
          if (isDragging) {
            const deltaX = e.clientX - previousMousePosition.x
            const deltaY = e.clientY - previousMousePosition.y

            // Update rotation velocity based on drag - SLOWER for calming feel
            rotationVelocityRef.current.x += deltaY * 0.002
            rotationVelocityRef.current.y += deltaX * 0.002

            previousMousePosition = { x: e.clientX, y: e.clientY }
          }
        }

        const handleMouseUp = () => {
          isDragging = false
          if (containerRef.current) {
            containerRef.current.style.cursor = 'grab'
          }
        }

        // Add event listeners
        if (containerRef.current) {
          containerRef.current.style.cursor = 'grab'
          containerRef.current.addEventListener('mousedown', handleMouseDown)
          containerRef.current.addEventListener('mousemove', handleMouseMove)
          containerRef.current.addEventListener('mouseup', handleMouseUp)
          containerRef.current.addEventListener('mouseleave', handleMouseUp)
        }

        // Gentle animation
        let time = 0
        const animate = () => {
          if (!rendererRef.current) return

          animationIdRef.current = requestAnimationFrame(animate)
          time += 0.008

          // Update shader times
          earthMaterial.uniforms.time.value = time
          atmosphereMaterial.uniforms.time.value = time
          atmosphere2Material.uniforms.time.value = time

          // Interactive Earth rotation with mouse drag
          if (Math.abs(rotationVelocityRef.current.x) > 0.0001) {
            earth.rotation.x += rotationVelocityRef.current.x
            rotationVelocityRef.current.x *= 0.92 // Stronger damping - stops faster
          }
          if (Math.abs(rotationVelocityRef.current.y) > 0.0001) {
            earth.rotation.y += rotationVelocityRef.current.y
            rotationVelocityRef.current.y *= 0.92 // Stronger damping - stops faster
          }

          // Very gentle auto-rotation when not dragging
          if (!isDragging && Math.abs(rotationVelocityRef.current.y) < 0.001) {
            earth.rotation.y += 0.0002 // Slower auto-rotation
          }

          // Slow atmospheric rotation
          atmosphere.rotation.y += 0.0003
          atmosphere2.rotation.y -= 0.0002

          // Parallax star layers - each rotates at different speed
          starLayers.forEach((layer, index) => {
            layer.mesh.rotation.y += layer.speed
            layer.mesh.rotation.x += layer.speed * 0.5

            // Subtle pulsing for each layer
            const layerPulse = Math.sin(time * 0.2 + index * 2.0) * 0.15 + 0.85
            layer.mesh.material.opacity = [0.5, 0.7, 0.8][index] * layerPulse
          })

          // Occasional shooting star effect (very rare)
          if (Math.random() < 0.0005) {
            // Could add shooting star particle here
            console.log('✨ Shooting star!')
          }

          // Very subtle camera sway - doesn't interfere with interaction
          camera.position.x = Math.sin(time * 0.08) * 0.01
          camera.position.y = 0.0 + Math.cos(time * 0.1) * 0.01  // Perfectly centered vertically

          renderer.render(scene, camera)
        }
        animate()

        // Handle resize
        const handleResize = () => {
          if (!containerRef.current || !renderer) return

          const width = containerRef.current.clientWidth
          const height = containerRef.current.clientHeight

          camera.aspect = width / height
          camera.updateProjectionMatrix()
          renderer.setSize(width, height)
        }
        window.addEventListener('resize', handleResize)

        return () => {
          // Cleanup event listeners
          window.removeEventListener('resize', handleResize)
          if (containerRef.current) {
            containerRef.current.removeEventListener('mousedown', handleMouseDown)
            containerRef.current.removeEventListener('mousemove', handleMouseMove)
            containerRef.current.removeEventListener('mouseup', handleMouseUp)
            containerRef.current.removeEventListener('mouseleave', handleMouseUp)
          }
        }
      } catch (err) {
        console.error('❌ Globe initialization error:', err)
        setError('Failed to initialize 3D globe')
      }
    }, 100)

    return () => {
      clearTimeout(initTimer)

      if (animationIdRef.current !== null) {
        cancelAnimationFrame(animationIdRef.current)
      }

      if (rendererRef.current) {
        rendererRef.current.dispose()
        if (containerRef.current && rendererRef.current.domElement) {
          try {
            containerRef.current.removeChild(rendererRef.current.domElement)
          } catch (e) {
            console.error('Cleanup error:', e)
          }
        }
      }
    }
  }, [])

  if (error) {
    return (
      <div className="absolute inset-0 bg-black flex items-center justify-center">
        <div className="text-center">
          <div className="text-blue-400 text-6xl mb-4">🌍</div>
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="absolute inset-0 min-h-[600px]" />

      {/* Loading overlay with smooth fade out */}
      {loading && (
        <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center
                        transition-opacity duration-500 z-10">
          <div className="text-center">
            {/* Spinning Earth emoji */}
            <div className="text-7xl mb-4 animate-spin" style={{ animationDuration: '3s' }}>
              🌍
            </div>

            {/* Loading text */}
            <p className="text-emerald-400 text-lg font-medium mb-3">
              Loading Earth...
            </p>

            {/* Progress bar */}
            <div className="w-48 h-1 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-300"
                style={{ width: `${loadingProgress}%` }}
              />
            </div>

            {/* Progress percentage */}
            <p className="text-gray-400 text-sm mt-2">
              {Math.round(loadingProgress)}%
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

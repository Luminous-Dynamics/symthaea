'use client';

import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { motion } from 'framer-motion';

interface Project {
  id: string;
  name: string;
  type: string;
  capacity_mw: number;
  irr: number;
  latitude: number;
  longitude: number;
  status: string;
}

export default function AnimatedGlobe() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [projectCount, setProjectCount] = useState(0);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const composerRef = useRef<EffectComposer>();

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      45,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 3;

    // Renderer with enhanced settings
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true,
      powerPreference: "high-performance"
    });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Post-processing
    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);
    
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(mountRef.current.clientWidth, mountRef.current.clientHeight),
      0.5, // bloom strength
      0.4, // radius
      0.85  // threshold
    );
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.5;
    controls.minDistance = 1.5;
    controls.maxDistance = 5;

    // Lights with better positioning
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);
    
    const sunLight = new THREE.DirectionalLight(0xffffff, 1);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    const rimLight = new THREE.DirectionalLight(0x4a9eff, 0.3);
    rimLight.position.set(-5, -3, -5);
    scene.add(rimLight);

    // Earth with better textures
    const geometry = new THREE.SphereGeometry(1, 64, 64);
    
    // Create Earth with procedural texture instead of loading external
    const textureLoader = new THREE.TextureLoader();
    
    // Create a proper Earth texture with continents
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d')!;
    
    // Create ocean base
    const oceanGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    oceanGradient.addColorStop(0, '#0a2342');
    oceanGradient.addColorStop(0.5, '#1e4d8b');
    oceanGradient.addColorStop(1, '#0a2342');
    ctx.fillStyle = oceanGradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add some texture variation to ocean
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * canvas.width;
      const y = Math.random() * canvas.height;
      const radius = Math.random() * 50 + 10;
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
      gradient.addColorStop(0, 'rgba(30, 77, 139, 0.2)');
      gradient.addColorStop(1, 'rgba(30, 77, 139, 0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
    }
    
    // Draw continents (simplified shapes, but more realistic)
    ctx.fillStyle = '#2d5016';
    
    // Africa
    ctx.beginPath();
    ctx.ellipse(1100, 500, 150, 220, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Europe
    ctx.beginPath();
    ctx.ellipse(1050, 250, 100, 80, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Asia
    ctx.beginPath();
    ctx.ellipse(1400, 300, 250, 180, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // North America
    ctx.beginPath();
    ctx.ellipse(500, 300, 180, 200, -0.3, 0, Math.PI * 2);
    ctx.fill();
    
    // South America
    ctx.beginPath();
    ctx.ellipse(600, 650, 100, 180, 0.2, 0, Math.PI * 2);
    ctx.fill();
    
    // Australia
    ctx.beginPath();
    ctx.ellipse(1500, 700, 80, 60, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Antarctica
    ctx.fillStyle = '#e0e0e0';
    ctx.fillRect(0, 950, canvas.width, 74);
    
    // Add some detail to continents
    ctx.fillStyle = 'rgba(45, 80, 22, 0.5)';
    for (let i = 0; i < 50; i++) {
      const x = Math.random() * canvas.width;
      const y = Math.random() * canvas.height;
      const radius = Math.random() * 20 + 5;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    
    const earthTexture = new THREE.CanvasTexture(canvas);
    earthTexture.needsUpdate = true;

    const material = new THREE.MeshPhongMaterial({
      map: earthTexture,
      bumpScale: 0.05,
      specular: new THREE.Color(0x333333),
      shininess: 15,
      emissive: new THREE.Color(0x001122),
      emissiveIntensity: 0.1
    });

    const earth = new THREE.Mesh(geometry, material);
    scene.add(earth);

    // Atmosphere glow
    const atmosphereGeometry = new THREE.SphereGeometry(1.15, 64, 64);
    const atmosphereMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
          vec3 atmosphere = vec3(0.3, 0.6, 1.0) * intensity * 1.5;
          gl_FragColor = vec4(atmosphere, intensity * 0.8);
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true
    });
    const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    scene.add(atmosphere);

    // Create project points with connections
    const projectGroup = new THREE.Group();
    const projects: Project[] = [];
    
    // Generate sample projects
    for (let i = 0; i < 500; i++) {
      const lat = (Math.random() - 0.5) * 180;
      const lng = (Math.random() - 0.5) * 360;
      const project: Project = {
        id: `project-${i}`,
        name: `Energy Project ${i}`,
        type: ['solar', 'wind', 'hydro', 'nuclear', 'storage'][Math.floor(Math.random() * 5)],
        capacity_mw: Math.round(10 + Math.random() * 490),
        irr: 8 + Math.random() * 12,
        latitude: lat,
        longitude: lng,
        status: ['operational', 'construction', 'planning'][Math.floor(Math.random() * 3)]
      };
      projects.push(project);
    }

    // Convert lat/lng to 3D positions and create points
    const positions: number[] = [];
    const colors: number[] = [];
    const sizes: number[] = [];
    
    projects.forEach((project) => {
      const phi = (90 - project.latitude) * (Math.PI / 180);
      const theta = (project.longitude + 180) * (Math.PI / 180);
      
      const x = -(1.02) * Math.sin(phi) * Math.cos(theta);
      const y = (1.02) * Math.cos(phi);
      const z = (1.02) * Math.sin(phi) * Math.sin(theta);
      
      positions.push(x, y, z);
      
      // Color by type
      const color = new THREE.Color();
      switch(project.type) {
        case 'solar': color.setHex(0xffaa00); break;
        case 'wind': color.setHex(0x00aaff); break;
        case 'hydro': color.setHex(0x00ffaa); break;
        case 'nuclear': color.setHex(0xff00aa); break;
        case 'storage': color.setHex(0xaa00ff); break;
        default: color.setHex(0xffffff);
      }
      colors.push(color.r, color.g, color.b);
      
      // Size by capacity
      sizes.push(Math.min(project.capacity_mw / 100, 3));
    });

    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    pointGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    pointGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

    const pointMaterial = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 }
      },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        uniform float time;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          float pulse = 1.0 + 0.3 * sin(time * 2.0 + position.x * 10.0);
          gl_PointSize = size * pulse * 300.0 / -mvPosition.z;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
          if (dist > 0.5) discard;
          float opacity = 1.0 - dist * 2.0;
          gl_FragColor = vec4(vColor, opacity);
        }
      `,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true
    });

    const points = new THREE.Points(pointGeometry, pointMaterial);
    projectGroup.add(points);

    // Create connections between nearby projects
    const connectionGeometry = new THREE.BufferGeometry();
    const connectionPositions: number[] = [];
    const connectionColors: number[] = [];
    
    for (let i = 0; i < projects.length; i++) {
      for (let j = i + 1; j < projects.length; j++) {
        const distance = Math.sqrt(
          Math.pow(projects[i].latitude - projects[j].latitude, 2) +
          Math.pow(projects[i].longitude - projects[j].longitude, 2)
        );
        
        if (distance < 20 && Math.random() > 0.95) { // Connect nearby projects randomly
          const phi1 = (90 - projects[i].latitude) * (Math.PI / 180);
          const theta1 = (projects[i].longitude + 180) * (Math.PI / 180);
          const x1 = -(1.02) * Math.sin(phi1) * Math.cos(theta1);
          const y1 = (1.02) * Math.cos(phi1);
          const z1 = (1.02) * Math.sin(phi1) * Math.sin(theta1);
          
          const phi2 = (90 - projects[j].latitude) * (Math.PI / 180);
          const theta2 = (projects[j].longitude + 180) * (Math.PI / 180);
          const x2 = -(1.02) * Math.sin(phi2) * Math.cos(theta2);
          const y2 = (1.02) * Math.cos(phi2);
          const z2 = (1.02) * Math.sin(phi2) * Math.sin(theta2);
          
          connectionPositions.push(x1, y1, z1, x2, y2, z2);
          connectionColors.push(0.2, 0.5, 1.0, 0.2, 0.5, 1.0);
        }
      }
    }

    connectionGeometry.setAttribute('position', new THREE.Float32BufferAttribute(connectionPositions, 3));
    connectionGeometry.setAttribute('color', new THREE.Float32BufferAttribute(connectionColors, 3));
    
    const connectionMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending
    });
    
    const connections = new THREE.LineSegments(connectionGeometry, connectionMaterial);
    projectGroup.add(connections);
    
    scene.add(projectGroup);

    // Animate project counter
    let displayCount = 0;
    const targetCount = projects.length;
    const countInterval = setInterval(() => {
      if (displayCount < targetCount) {
        displayCount += Math.ceil((targetCount - displayCount) / 10);
        setProjectCount(Math.min(displayCount, targetCount));
      } else {
        clearInterval(countInterval);
      }
    }, 50);

    // Animation loop
    const clock = new THREE.Clock();
    const animate = () => {
      requestAnimationFrame(animate);
      
      const elapsedTime = clock.getElapsedTime();
      
      // Update uniforms
      if (pointMaterial.uniforms.time) {
        pointMaterial.uniforms.time.value = elapsedTime;
      }
      
      // Rotate earth
      earth.rotation.y += 0.001;
      atmosphere.rotation.y += 0.0005;
      projectGroup.rotation.y += 0.0002;
      
      // Pulse connections
      connectionMaterial.opacity = 0.2 + 0.1 * Math.sin(elapsedTime * 2);
      
      controls.update();
      composer.render();
    };

    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
      composer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      clearInterval(countInterval);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-full" />
      
      {/* Project Counter */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1 }}
        className="absolute top-4 left-4 bg-black/50 backdrop-blur-md rounded-lg px-4 py-2 border border-white/10"
      >
        <div className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
          {projectCount.toLocaleString()}
        </div>
        <div className="text-sm text-gray-400">Energy Projects Mapped</div>
      </motion.div>

      {/* Legend */}
      <motion.div 
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-4 right-4 bg-black/50 backdrop-blur-md rounded-lg p-3 border border-white/10"
      >
        <div className="text-xs font-semibold text-gray-400 mb-2">PROJECT TYPES</div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
            <span className="text-xs text-gray-300">Solar</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-500"></div>
            <span className="text-xs text-gray-300">Wind</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span className="text-xs text-gray-300">Hydro</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-pink-500"></div>
            <span className="text-xs text-gray-300">Nuclear</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-purple-500"></div>
            <span className="text-xs text-gray-300">Storage</span>
          </div>
        </div>
      </motion.div>

      {/* Instructions */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2 }}
        className="absolute top-4 right-4 bg-black/30 backdrop-blur-sm rounded-lg px-3 py-2 border border-white/5"
      >
        <div className="text-xs text-gray-400">
          <div className="flex items-center gap-1">
            <span>🖱️</span> Drag to rotate
          </div>
          <div className="flex items-center gap-1">
            <span>🔍</span> Scroll to zoom
          </div>
        </div>
      </motion.div>
    </div>
  );
}
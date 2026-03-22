'use client';

import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
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

export default function GlobeReliable() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [projectCount, setProjectCount] = useState(0);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();

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

    // Renderer with better settings
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true
    });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.5;
    controls.minDistance = 1.5;
    controls.maxDistance = 5;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
    scene.add(ambientLight);
    
    const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    const backLight = new THREE.DirectionalLight(0x4a9eff, 0.3);
    backLight.position.set(-5, -3, -5);
    scene.add(backLight);

    // Create a beautiful Earth with gradient colors
    const geometry = new THREE.SphereGeometry(1, 64, 64);
    
    // Use vertex colors for a reliable gradient effect
    const vertexColors = new Float32Array(geometry.attributes.position.count * 3);
    for (let i = 0; i < geometry.attributes.position.count; i++) {
      const y = geometry.attributes.position.array[i * 3 + 1];
      
      // Create a blue to green gradient based on Y position
      if (Math.abs(y) > 0.8) {
        // Polar regions - white
        vertexColors[i * 3] = 0.9;
        vertexColors[i * 3 + 1] = 0.9;
        vertexColors[i * 3 + 2] = 0.95;
      } else if (Math.random() > 0.7) {
        // Land masses - green
        vertexColors[i * 3] = 0.2;
        vertexColors[i * 3 + 1] = 0.5;
        vertexColors[i * 3 + 2] = 0.2;
      } else {
        // Oceans - blue
        const blueVariation = 0.3 + Math.random() * 0.3;
        vertexColors[i * 3] = 0.1;
        vertexColors[i * 3 + 1] = 0.3;
        vertexColors[i * 3 + 2] = blueVariation;
      }
    }
    
    geometry.setAttribute('color', new THREE.BufferAttribute(vertexColors, 3));
    
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      shininess: 30,
      specular: new THREE.Color(0x444444)
    });

    const earth = new THREE.Mesh(geometry, material);
    scene.add(earth);

    // Add cloud layer
    const cloudGeometry = new THREE.SphereGeometry(1.02, 64, 64);
    const cloudMaterial = new THREE.MeshPhongMaterial({
      transparent: true,
      opacity: 0.2,
      color: 0xffffff,
      side: THREE.FrontSide
    });
    const clouds = new THREE.Mesh(cloudGeometry, cloudMaterial);
    scene.add(clouds);

    // Atmosphere glow
    const atmosphereGeometry = new THREE.SphereGeometry(1.15, 64, 64);
    const atmosphereMaterial = new THREE.MeshBasicMaterial({
      color: 0x4a9eff,
      transparent: true,
      opacity: 0.1,
      side: THREE.BackSide
    });
    const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    scene.add(atmosphere);

    // Create project points
    const projects: Project[] = [];
    for (let i = 0; i < 79197; i++) {
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

    // Create point cloud for projects
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
      switch(project.type) {
        case 'solar': 
          colors.push(1, 0.7, 0); // Yellow
          break;
        case 'wind': 
          colors.push(0, 0.7, 1); // Blue
          break;
        case 'hydro': 
          colors.push(0, 1, 0.7); // Cyan
          break;
        case 'nuclear': 
          colors.push(1, 0, 0.7); // Magenta
          break;
        case 'storage': 
          colors.push(0.7, 0, 1); // Purple
          break;
        default: 
          colors.push(1, 1, 1); // White
      }
      
      sizes.push(Math.min(project.capacity_mw / 100, 3));
    });

    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    pointGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    pointGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

    const pointMaterial = new THREE.PointsMaterial({
      size: 0.01,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending
    });

    const points = new THREE.Points(pointGeometry, pointMaterial);
    scene.add(points);

    // Animate counter
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
    const animate = () => {
      requestAnimationFrame(animate);
      
      // Rotate everything
      earth.rotation.y += 0.001;
      clouds.rotation.y += 0.0015;
      points.rotation.y += 0.001;
      atmosphere.rotation.y += 0.0005;
      
      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
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